// Averaged normals for the half-resolution lighting (TOTK_LIGHT_NORMALS: game, average; applies in play).
// TotK lights its point lights at half the screen resolution. Before that, one draw copies the G-buffer's normals
// (target 3, RGBA8, sphere-map encoded) to a half-size RGBA8 image with a single nearest read per pixel, so every
// half-size normal is one of the four full-resolution normals under it. Normal-map detail finer than two pixels then
// lights differently with every small camera move: everything a point light reaches shimmers, near and far (1.4.3: the
// copy's fragment shader is SPIR-V 7e93281b21b94126, the point lights draw light volumes into a half-size R11G11B10
// pair afterwards and the composite reads that with one bilinear sample).
// With average, the draw is found as the first one after a frame's G-buffer that writes a single RGBA8 target of half
// the G-buffer's size and samples the G-buffer's normals (vk_lightnorm_sampled, from vk_exec_draw.c tex_info). Before
// the next pass, runtime/shaders/light_normals.comp writes the average of those of the four normals on the same
// surface into an image of our own, and the lighting samples that in place of the game's copy (vk_lightnorm_swap_img
// and vk_lightnorm_swap_view, applied by tex_info) until the next frame's G-buffer. The game's copy is not written
// again after its draw (the lighting passes bind it as a target and leave it unchanged), so the swap loses nothing.
// A frame can draw into its G-buffer again between the lighting passes (1.4.3, the Depths), so only the first G-buffer
// pass after a present starts a new frame.
#include "vk.h"

#include <string.h>

#include "cfg.h"
#include "host.h"
#include "light_normals_spv.h"
#include "vk_exec.h"
#include "vk_rec.h"

typedef struct {
    VkImage img;
    VmaAllocation alloc;
    VkImageView view;
} Img;

const VkTexImage *vk_lightnorm_swap_img;
VkImageView vk_lightnorm_swap_view;
int vk_lightnorm_watch;

static struct {
    int tried, on, ok, init_tried, logged;
    unsigned cfg_ver;
    uint64_t frame, gb_frame;           // presents so far; the frame whose G-buffer was seen last
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;          // sampled at 0, 1, 2; storage at 3
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler point;
    Img out, old[8];                    // old: earlier sizes, which recorded work may still use (freed at release)
    int nold, fresh;                    // fresh: the image is still in UNDEFINED layout
    uint32_t w, h;                      // half the G-buffer's size
    const VkTexImage *gnormal, *depth;  // this frame's G-buffer normals and linear depth
    uint32_t dw, dh;
    const VkTexImage *half;             // the game's half-size copy of the normals
    int done, pending;                  // the copy was found this frame; ours is due before the next pass
} L;

static void swap_off(void) {
    if (!vk_lightnorm_swap_img) return;
    vk_lightnorm_swap_img = NULL, vk_lightnorm_swap_view = VK_NULL_HANDLE;
    vk_exec_draw_new_cb();   // a draw with the same bindings as the one before must not keep the swapped view
}

static void update_mode(void) {
    unsigned v = cfg_version();
    if (L.tried && v == L.cfg_ver) return;
    int on = !strcmp(cfg_value("TOTK_LIGHT_NORMALS"), "average"), first = !L.tried;
    L.tried = 1, L.cfg_ver = v;
    if (on == L.on && !first) return;
    L.on = on;
    if (!on) swap_off(), vk_lightnorm_watch = 0;
    rt_log(on ? "vk: point lights shade from averaged half-resolution normals"
              : "vk: point lights shade from the game's half-resolution normals");
}

static int init(VkCtx *c) {
    L.init_tried = 1;
    L.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < 4; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                      : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 4, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &L.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof light_normals_spv, .pCode = light_normals_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!L.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &L.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &L.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &L.point) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = L.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &L.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    return L.ok = 1;
fail:
    rt_log("vk: averaged light normals off: the pipeline could not be created");
    return 0;
}

static void free_image(VkCtx *c, Img *m) {
    if (m->view) vkDestroyImageView(c->device, m->view, NULL);
    if (m->img) vmaDestroyImage(c->vma, m->img, m->alloc);
    memset(m, 0, sizeof *m);
}

static int ensure_image(VkCtx *c, uint32_t w, uint32_t h) {
    if (L.out.img && L.w == w && L.h == h) return 1;
    if (L.out.img) {
        if (L.nold == 8) {   // a full list of retired sizes: wait for the GPU, free them (as vk_dlaa.c does)
            vkDeviceWaitIdle(c->device);
            for (int i = 0; i < L.nold; i++) free_image(c, &L.old[i]);
            L.nold = 0;
        }
        L.old[L.nold++] = L.out;
        memset(&L.out, 0, sizeof L.out);
    }
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vmaCreateImage(c->vma, &ci, &ai, &L.out.img, &L.out.alloc, NULL) != VK_SUCCESS) {
        L.out.img = VK_NULL_HANDLE;
        goto fail;
    }
    vi.image = L.out.img;
    if (vkCreateImageView(c->device, &vi, NULL, &L.out.view) != VK_SUCCESS) goto fail;
    L.w = w, L.h = h, L.fresh = 1;
    rt_log("vk: averaged light normals at %ux%u from %ux%u", w, h, L.dw, L.dh);
    return 1;
fail:
    free_image(c, &L.out);
    rt_log("vk: averaged light normals off: a %ux%u image could not be created", w, h);
    L.ok = 0;
    return 0;
}

// A G-buffer pass: a D32F depth and, at the pass's size, an R32F target (the linear depth) and an RGBA8 target 3 (the
// normals).
static int gbuffer(const VkExec *ex, const VkTexImage **normals, const VkTexImage **depth) {
    *normals = *depth = NULL;
    if (ex->pass_noatt || ex->pass_w < 640 || ex->depth.format != VK_FORMAT_D32_SFLOAT || ex->depth.level ||
        ex->pass_ncolor < 4)
        return 0;
    const VkExecTarget *n = &ex->color[3];
    if (n->image && !n->level && n->width == ex->pass_w && n->height == ex->pass_h &&
        n->format == VK_FORMAT_R8G8B8A8_UNORM)
        *normals = n->img;
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++) {
        const VkExecTarget *t = &ex->color[i];
        if (t->image && !t->level && t->width == ex->pass_w && t->height == ex->pass_h &&
            t->format == VK_FORMAT_R32_SFLOAT)
            *depth = t->img;
    }
    return *normals && *depth;
}

void vk_lightnorm_present(uint64_t frame) { __atomic_store_n(&L.frame, frame, __ATOMIC_RELAXED); }

void vk_lightnorm_pass_begin(VkExec *ex) {
    update_mode();
    if (L.on) {
        const VkTexImage *n, *d;
        uint64_t f = __atomic_load_n(&L.frame, __ATOMIC_RELAXED);
        if (gbuffer(ex, &n, &d) && (f != L.gb_frame || !L.gnormal)) {   // a new frame's G-buffer
            L.gb_frame = f, L.gnormal = n, L.depth = d, L.dw = ex->pass_w, L.dh = ex->pass_h;
            L.done = L.pending = 0;
            swap_off();
        }
    }
    vk_lightnorm_flush(ex);
}

// A draw about to be recorded, its pass open: a candidate for the copy while it writes one RGBA8 target of half the
// G-buffer's size; tex_info then reports what it samples.
void vk_lightnorm_draw(VkExec *ex) {
    update_mode();
    vk_lightnorm_watch = 0;
    if (!L.on || L.done || !L.gnormal || ex->pass_ncolor != 1) return;
    const VkExecTarget *t = &ex->color[0];
    if (!t->img || t->level || t->format != VK_FORMAT_R8G8B8A8_UNORM) return;
    if (t->width * 2 + 1 < L.dw || t->width * 2 > L.dw + 1 || t->height * 2 + 1 < L.dh || t->height * 2 > L.dh + 1) return;
    vk_lightnorm_watch = 1;
}

void vk_lightnorm_sampled(VkExec *ex, const VkTexImage *img) {
    if (!vk_lightnorm_watch || !img || img != L.gnormal) return;
    vk_lightnorm_watch = 0;
    L.done = 1, L.half = ex->color[0].img;
    if (!L.init_tried) init(vk_ctx());
    if (L.ok) L.pending = 1;
    if (!L.logged) {
        L.logged = 1;
        rt_log("vk: the game's half-size light normals found: pass %u, %ux%u", ex->pass_serial, ex->color[0].width,
               ex->color[0].height);
    }
}

void vk_lightnorm_flush(VkExec *ex) {
    if (!L.pending || ex->in_pass) return;
    L.pending = 0;
    if (!L.ok || !L.half || !L.gnormal || !L.depth) return;
    VkCtx *c = vk_ctx();
    if (!ensure_image(c, L.half->width, L.half->height)) return;
    VkImageView n = vk_exec_target_view(L.gnormal, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D,
                                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView d = vk_exec_target_view(L.depth, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                        0, 0, 1, 0);
    VkImageView h = vk_exec_target_view(L.half, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D,
                                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    if (!n || !d || !h) return;
    vk_exec_barrier(ex);   // the copy's writes; images in the feedback layout go back to GENERAL
    VkCommandBuffer cb = ex->cb;
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,   // the frame before read it
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                .oldLayout = L.fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = L.out.img,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                           .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    L.fresh = 0;
    VkDescriptorImageInfo ii[4] = {{L.point, n, VK_IMAGE_LAYOUT_GENERAL}, {L.point, d, VK_IMAGE_LAYOUT_GENERAL},
                                   {L.point, h, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, L.out.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[4];
    for (uint32_t i = 0; i < 4; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i,
                                      .descriptorCount = 1,
                                      .descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .pImageInfo = &ii[i]};
    int32_t pc[4] = {(int32_t)L.w, (int32_t)L.h, (int32_t)L.dw, (int32_t)L.dh};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, L.pipe);
    rec_push(L.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, L.layout, 0, 4, w);
    vkCmdPushConstants(cb, L.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (L.w + 7) / 8, (L.h + 7) / 8, 1);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT};
    VkDependencyInfo md = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &md);
    ex->recorded += 6;
    if (vk_dump_armed) {   // TOTK_DUMP: the game's copy and ours
        vk_dump_image(ex, L.half->image, VK_FORMAT_R8G8B8A8_UNORM, L.half->width, L.half->height, "lightnorm_game");
        vk_dump_image(ex, L.out.img, VK_FORMAT_R8G8B8A8_UNORM, L.w, L.h, "lightnorm_average");
    }
    // the game's bound compute pipeline and descriptors were replaced: bind them again; from here the lighting samples
    // our normals
    vk_exec_draw_new_cb();
    vk_lightnorm_swap_img = L.half, vk_lightnorm_swap_view = L.out.view;
}

void vk_lightnorm_release(void) {
    VkCtx *c = vk_ctx();
    vk_lightnorm_swap_img = NULL, vk_lightnorm_swap_view = VK_NULL_HANDLE, vk_lightnorm_watch = 0;
    if (!c || !L.tried) return;
    for (int i = 0; i < L.nold; i++) free_image(c, &L.old[i]);
    free_image(c, &L.out);
    if (L.pipe) vkDestroyPipeline(c->device, L.pipe, NULL);
    if (L.point) vkDestroySampler(c->device, L.point, NULL);
    if (L.layout) vkDestroyPipelineLayout(c->device, L.layout, NULL);
    if (L.dsl) vkDestroyDescriptorSetLayout(c->device, L.dsl, NULL);
    memset(&L, 0, sizeof L);
}
