// Point lights lift the occlusion that would cut them (TOTK_AO_LIGHT percent; with TOTK_AO=gtao, applies in play).
// TotK's half-size lighting writes an RGBA8 image whose red and blue channels (the occlusion) the full-size composite
// multiplies its point light by (1.4.3: the composite's fragment shaders read it at texture slot 16 and the point
// lights' own half-size image, two layers of R11G11B10F, at slot 28). The blue channel goes through a steep ramp first,
// so wherever occlusion is below about three quarters a point light adds nothing, however bright: GTAO's wide
// occlusion shows it as dark rings around a wall lamp. The game's own thin occlusion rarely did.
// The occlusion image is learned from the eight-colour-target lighting pass (target 0 RGBA8 at half the G-buffer's
// size; targets 2 and 3 are layers 0 and 1 of the point lights' image) and is written several times a frame; only the
// last write before the composite is read. A render pass that writes it makes ours out of date (the swap ends there),
// and once the point lights have been drawn, before the first pass at the G-buffer's size that follows a write,
// runtime/shaders/light_ao.comp copies it into an image of ours with red and blue moved towards 1 where the point
// lights are bright. Passes at the G-buffer's size then sample ours in its place (vk_lightao_swap_img and
// vk_lightao_swap_view, applied by tex_info), until the next write. A pixel reads only its own texels of both images,
// which share their size, so light is not spread across a depth edge.
// Only the first G-buffer pass after a present starts a new frame, as in vk_lightnorm.c.
#include "vk.h"

#include <string.h>

#include "cfg.h"
#include "host.h"
#include "light_ao_spv.h"
#include "vk_exec.h"
#include "vk_rec.h"

typedef struct {
    VkImage img;
    VmaAllocation alloc;
    VkImageView view;
} Img;

const VkTexImage *vk_lightao_swap_img;
VkImageView vk_lightao_swap_view;

// The point lights' light (layer 1, linear luminance) that lifts nothing and that lifts the occlusion completely.
#define LIGHT_LO 0.03f
#define LIGHT_HI 0.15f

static struct {
    int tried, on, ok, init_tried, logged;
    unsigned cfg_ver;
    float strength;                     // 0..1
    uint64_t frame, gb_frame;           // presents so far; the frame whose G-buffer was seen last
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;          // sampled at 0 and 1, storage at 2
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler point;
    Img out, old[8];                    // old: earlier sizes, which recorded work may still use (freed at release)
    int nold, fresh;                    // fresh: the image is still in UNDEFINED layout
    uint32_t w, h;                      // the occlusion image's size, ours
    uint32_t gw;                        // this frame's G-buffer width, 0 before its G-buffer pass
    const VkTexImage *occ, *lights;     // the game's occlusion image and the point lights' image, this frame
    int dirty;                          // the occlusion image was written since ours was made
    int lit;                            // a pass has drawn the point lights since the frame's lighting pass
} A;

static void swap_off(void) {
    if (!vk_lightao_swap_img) return;
    vk_lightao_swap_img = NULL, vk_lightao_swap_view = VK_NULL_HANDLE;
    vk_exec_draw_new_cb();   // a draw with the same bindings as the one before must not keep the swapped view
}

static void update_mode(void) {
    unsigned v = cfg_version();
    if (A.tried && v == A.cfg_ver) return;
    int pct = cfg_int("TOTK_AO_LIGHT");
    A.strength = (float)(pct < 0 ? 0 : pct > 100 ? 100 : pct) / 100.0f;
    int on = !strcmp(cfg_value("TOTK_AO"), "gtao") && A.strength > 0, first = !A.tried;
    A.tried = 1, A.cfg_ver = v;
    if (on == A.on && !first) return;
    A.on = on;
    if (!on) swap_off();
    rt_log(on ? "vk: point lights lift the occlusion over them, up to %.0f %%"
              : "vk: point lights are cut by the game's occlusion",
           A.strength * 100.0f);
}

static int init(VkCtx *c) {
    A.init_tried = 1;
    A.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[3];
    for (uint32_t i = 0; i < 3; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                      : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 3, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &A.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof light_ao_spv, .pCode = light_ao_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!A.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &A.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &A.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &A.point) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = A.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &A.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    return A.ok = 1;
fail:
    rt_log("vk: point lights lifting occlusion off: the pipeline could not be created");
    return 0;
}

static void free_image(VkCtx *c, Img *m) {
    if (m->view) vkDestroyImageView(c->device, m->view, NULL);
    if (m->img) vmaDestroyImage(c->vma, m->img, m->alloc);
    memset(m, 0, sizeof *m);
}

static int ensure_image(VkCtx *c, uint32_t w, uint32_t h) {
    if (A.out.img && A.w == w && A.h == h) return 1;
    if (A.out.img) {
        if (A.nold == 8) return 0;
        A.old[A.nold++] = A.out;
        memset(&A.out, 0, sizeof A.out);
    }
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vmaCreateImage(c->vma, &ci, &ai, &A.out.img, &A.out.alloc, NULL) != VK_SUCCESS) {
        A.out.img = VK_NULL_HANDLE;
        goto fail;
    }
    vi.image = A.out.img;
    if (vkCreateImageView(c->device, &vi, NULL, &A.out.view) != VK_SUCCESS) goto fail;
    A.w = w, A.h = h, A.fresh = 1;
    return 1;
fail:
    free_image(c, &A.out);
    rt_log("vk: point lights lifting occlusion off: a %ux%u image could not be created", w, h);
    A.ok = 0;
    return 0;
}

static int full_size(const VkExec *ex, const VkExecTarget *t) {
    return t->image && !t->level && t->width == ex->pass_w && t->height == ex->pass_h;
}

// A G-buffer pass: a D32F depth, at least four colour targets and an R32F one (the linear depth) at the pass's size.
static int gbuffer(const VkExec *ex) {
    if (ex->pass_noatt || ex->pass_w < 640 || ex->depth.format != VK_FORMAT_D32_SFLOAT || ex->depth.level ||
        ex->pass_ncolor < 4)
        return 0;
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++)
        if (full_size(ex, &ex->color[i]) && ex->color[i].format == VK_FORMAT_R32_SFLOAT) return 1;
    return 0;
}

// The lighting pass: six or more colour targets at half the G-buffer's size, target 0 the RGBA8 occlusion image and
// targets 2 and 3 layers 0 and 1 of one R11G11B10F image, the point lights'.
static int lighting(const VkExec *ex) {
    if (!A.gw || ex->pass_ncolor < 6 || ex->pass_w * 2 + 1 < A.gw || ex->pass_w * 2 > A.gw + 1) return 0;
    const VkExecTarget *o = &ex->color[0], *l0 = &ex->color[2], *l1 = &ex->color[3];
    return full_size(ex, o) && o->img && o->format == VK_FORMAT_R8G8B8A8_UNORM && full_size(ex, l0) && l0->img &&
           l0->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && l0->img == l1->img && l0->base_layer == 0 &&
           l1->base_layer == 1;
}

static void flush(VkExec *ex);

void vk_lightao_present(uint64_t frame) { __atomic_store_n(&A.frame, frame, __ATOMIC_RELAXED); }

void vk_lightao_pass_begin(VkExec *ex) {
    update_mode();
    if (!A.on) return;
    if (gbuffer(ex)) {
        uint64_t f = __atomic_load_n(&A.frame, __ATOMIC_RELAXED);
        if (f != A.gb_frame || !A.gw) {   // a new frame's G-buffer: what was learned of the last one is gone
            A.gb_frame = f, A.gw = ex->pass_w;
            A.occ = A.lights = NULL;
            A.dirty = A.lit = 0;
            swap_off();
        }
        return;
    }
    if (!A.occ && lighting(ex)) A.occ = ex->color[0].img, A.lights = ex->color[2].img;
    // the first pass at the G-buffer's size after the occlusion image was written and the point lights drawn: the
    // composite
    if (A.occ && A.dirty && A.lit && ex->pass_w >= A.gw) flush(ex);
}

// The pass that ends: a write of the occlusion image makes ours out of date; a pass whose first target is the point
// lights' image has drawn them.
void vk_lightao_pass_end(VkExec *ex) {
    if (!A.on || !A.occ) return;
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++) {
        const VkExecTarget *t = &ex->color[i];
        if (!t->img) continue;
        if (t->img == A.occ) A.dirty = 1, swap_off();
        else if (i == 0 && t->img == A.lights) A.lit = 1;
    }
}

static void flush(VkExec *ex) {
    if (ex->in_pass) return;
    A.dirty = 0;
    VkCtx *c = vk_ctx();
    if (!A.init_tried) init(c);
    if (!A.ok || !A.occ || !A.lights || !ensure_image(c, A.occ->width, A.occ->height)) return swap_off();
    VkImageView o = vk_exec_target_view(A.occ, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D,
                                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView l = vk_exec_target_view(A.lights, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2, 0);
    if (!o || !l) return swap_off();
    vk_exec_barrier(ex);   // the game's writes; images in the feedback layout go back to GENERAL
    VkCommandBuffer cb = ex->cb;
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,   // earlier draws read ours
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                .oldLayout = A.fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = A.out.img,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                           .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    A.fresh = 0;
    VkDescriptorImageInfo ii[3] = {{A.point, o, VK_IMAGE_LAYOUT_GENERAL}, {A.point, l, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, A.out.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[3];
    for (uint32_t i = 0; i < 3; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i,
                                      .descriptorCount = 1,
                                      .descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .pImageInfo = &ii[i]};
    struct { int32_t size[2]; float strength, lo, hi; } pc = {{(int32_t)A.w, (int32_t)A.h}, A.strength, LIGHT_LO, LIGHT_HI};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.pipe);
    rec_push(A.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.layout, 0, 3, w);
    vkCmdPushConstants(cb, A.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (A.w + 7) / 8, (A.h + 7) / 8, 1);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT};
    VkDependencyInfo md = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &md);
    ex->recorded += 5;
    if (!A.logged) {
        A.logged = 1;
        rt_log("vk: point lights lift the occlusion image at %ux%u (pass %u)", A.w, A.h, ex->pass_serial);
    }
    if (vk_dump_armed)   // TOTK_DUMP: ours
        vk_dump_image(ex, A.out.img, VK_FORMAT_R8G8B8A8_UNORM, A.w, A.h, "light_ao");
    // the game's bound compute pipeline and descriptors were replaced: bind them again; from here the composite
    // samples ours
    vk_exec_draw_new_cb();
    vk_lightao_swap_img = A.occ, vk_lightao_swap_view = A.out.view;
}

void vk_lightao_release(void) {
    VkCtx *c = vk_ctx();
    vk_lightao_swap_img = NULL, vk_lightao_swap_view = VK_NULL_HANDLE;
    if (!c || !A.tried) return;
    for (int i = 0; i < A.nold; i++) free_image(c, &A.old[i]);
    free_image(c, &A.out);
    if (A.pipe) vkDestroyPipeline(c->device, A.pipe, NULL);
    if (A.point) vkDestroySampler(c->device, A.point, NULL);
    if (A.layout) vkDestroyPipelineLayout(c->device, A.layout, NULL);
    if (A.dsl) vkDestroyDescriptorSetLayout(c->device, A.dsl, NULL);
    memset(&A, 0, sizeof A);
}
