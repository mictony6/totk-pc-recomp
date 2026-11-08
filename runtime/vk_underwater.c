// Full-resolution view through water and ice (TOTK_UNDERWATER_UPSAMPLE: game, smooth; applies in play).
// TotK draws its water and ice into the G-buffer, and what they show through themselves (the riverbed, the scene behind
// the ice) is a half-size R11G11B10F image of the lit scene without them, which a half-size pass of the lighting writes
// before the G-buffer's water draws (1.4.3: target 5 of an eight-target pass, with its linear depth as an R32F target).
// Sampled bilinearly at half size, everything under water looks blurred and stair-stepped.
// With smooth:
// - the image is found as the first half-size R11G11B10F 2D image a draw in a full-size G-buffer pass samples (the
//   water's and the ice's draws; vk_underwater_sampled, from vk_exec_draw.c tex_info while vk_underwater_watch is set),
//   and remembered by address. The game reuses memory for other images (the title screen's frames put another image
//   there), so a frame whose G-buffer follows one in which no pass wrote an R11G11B10F image at that address
//   forgets it and finds it again;
// - a render pass that writes it (vk_underwater_pass_end) makes ours out of date: the swap ends there, and before the
//   next full-size G-buffer pass (where the water and ice are drawn; 1.4.3 writes the image three times a frame, and
//   only the write before that pass is read at full size) runtime/shaders/scene_upsample.comp upsamples the latest
//   write to the G-buffer's size into an image of our own, guided by the writing pass's half-size linear depth and the
//   G-buffer's;
// - draws in full-size passes then sample ours in its place (vk_underwater_swap_img and vk_underwater_swap_view,
//   applied by tex_info); half-size passes keep the game's. Ours is never older than the game's, which
//   TOTK_LIGHT_NORMALS=average got wrong (it swapped once a frame and missed a rewrite).
// Only the first G-buffer pass after a present starts a new frame, as in vk_lightnorm.c.
#include "vk.h"

#include <string.h>

#include "cfg.h"
#include "host.h"
#include "scene_upsample_spv.h"
#include "vk_exec.h"
#include "vk_rec.h"

typedef struct {
    VkImage img;
    VmaAllocation alloc;
    VkImageView view;
} Img;

const VkTexImage *vk_underwater_swap_img;
VkImageView vk_underwater_swap_view;
int vk_underwater_watch;   // the open pass is a full-size G-buffer pass and the scene is not found yet

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
    uint32_t w, h;                      // the G-buffer's size, ours
    uint64_t src_addr;                  // the game's half-size scene image, by address; 0 until found
    const VkTexImage *gdepth;           // this frame's G-buffer linear depth
    const VkTexImage *src, *hdepth;     // the game's image and its half-size depth, from the pass that wrote it
    int pending;                        // ours is due before the next full-size G-buffer pass
    int written;                        // this frame wrote the image at src_addr
    int relearned;                      // times src_addr was forgotten (the first few are logged)
    int debug;                          // TOTK_UNDERWATER_DEBUG: ours is solid magenta, to see where it is shown
} U;

static void swap_off(void) {
    if (!vk_underwater_swap_img) return;
    vk_underwater_swap_img = NULL, vk_underwater_swap_view = VK_NULL_HANDLE;
    vk_exec_draw_new_cb();   // a draw with the same bindings as the one before must not keep the swapped view
}

static void update_mode(void) {
    unsigned v = cfg_version();
    if (U.tried && v == U.cfg_ver) return;
    int on = !strcmp(cfg_value("TOTK_UNDERWATER_UPSAMPLE"), "smooth"), first = !U.tried;
    U.tried = 1, U.cfg_ver = v;
    const char *dbg = cfg_get("TOTK_UNDERWATER_DEBUG");   // a diagnostic: no settings row, unset is off
    U.debug = dbg && *dbg && strcmp(dbg, "0");
    if (on == U.on && !first) return;
    U.on = on;
    if (!on) swap_off(), U.pending = 0, vk_underwater_watch = 0;
    rt_log(on ? "vk: water and ice show the scene behind them upsampled to full size"
              : "vk: water and ice show the game's half-size scene");
}

static int init(VkCtx *c) {
    U.init_tried = 1;
    U.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < 4; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                      : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 4, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &U.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof scene_upsample_spv, .pCode = scene_upsample_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!U.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &U.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &U.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &U.point) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = U.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &U.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    return U.ok = 1;
fail:
    rt_log("vk: underwater upsample off: the pipeline could not be created");
    return 0;
}

static void free_image(VkCtx *c, Img *m) {
    if (m->view) vkDestroyImageView(c->device, m->view, NULL);
    if (m->img) vmaDestroyImage(c->vma, m->img, m->alloc);
    memset(m, 0, sizeof *m);
}

static int ensure_image(VkCtx *c, uint32_t w, uint32_t h) {
    if (U.out.img && U.w == w && U.h == h) return 1;
    if (U.out.img) {
        if (U.nold == 8) {   // a full list of retired sizes: wait for the GPU, free them (as vk_dlaa.c does)
            vkDeviceWaitIdle(c->device);
            for (int i = 0; i < U.nold; i++) free_image(c, &U.old[i]);
            U.nold = 0;
        }
        U.old[U.nold++] = U.out;
        memset(&U.out, 0, sizeof U.out);
    }
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R16G16B16A16_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1,
                            .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vmaCreateImage(c->vma, &ci, &ai, &U.out.img, &U.out.alloc, NULL) != VK_SUCCESS) {
        U.out.img = VK_NULL_HANDLE;
        goto fail;
    }
    vi.image = U.out.img;
    if (vkCreateImageView(c->device, &vi, NULL, &U.out.view) != VK_SUCCESS) goto fail;
    U.w = w, U.h = h, U.fresh = 1;
    rt_log("vk: underwater upsample at %ux%u", w, h);
    return 1;
fail:
    free_image(c, &U.out);
    rt_log("vk: underwater upsample off: a %ux%u image could not be created", w, h);
    U.ok = 0;
    return 0;
}

static int full_size(const VkExec *ex, const VkExecTarget *t) {
    return t->image && !t->level && t->width == ex->pass_w && t->height == ex->pass_h;
}

// A G-buffer pass: a D32F depth, at least four colour targets and an R32F one (the linear depth) at the pass's size.
static const VkTexImage *gbuffer_depth(const VkExec *ex) {
    if (ex->pass_noatt || ex->pass_w < 640 || ex->depth.format != VK_FORMAT_D32_SFLOAT || ex->depth.level ||
        ex->pass_ncolor < 4)
        return NULL;
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++)
        if (full_size(ex, &ex->color[i]) && ex->color[i].format == VK_FORMAT_R32_SFLOAT) return ex->color[i].img;
    return NULL;
}

static void vk_underwater_flush(VkExec *ex);

void vk_underwater_present(uint64_t frame) { __atomic_store_n(&U.frame, frame, __ATOMIC_RELAXED); }

void vk_underwater_pass_begin(VkExec *ex) {
    update_mode();
    vk_underwater_watch = 0;
    if (!U.on) return;
    const VkTexImage *d = gbuffer_depth(ex);
    if (!d) return;
    uint64_t f = __atomic_load_n(&U.frame, __ATOMIC_RELAXED);
    if (f != U.gb_frame || !U.gdepth) {   // a new frame's G-buffer: what was learned of the last one is gone
        if (U.src_addr && !U.written && U.gdepth) {   // the last frame did not write it: another image lives there now
            if (U.relearned++ < 4)
                rt_log("vk: the half-size scene behind water was not written at %#llx last frame: finding it again",
                       (unsigned long long)U.src_addr);
            U.src_addr = 0;
        }
        U.gb_frame = f, U.gdepth = d, U.written = 0;
        U.src = U.hdepth = NULL;
        U.pending = 0;
        swap_off();
    }
    vk_underwater_watch = !U.src_addr;
    vk_underwater_flush(ex);   // a later G-buffer pass of the frame: the water's
}

// The pass that ends: when it wrote the game's half-size scene, ours is due.
void vk_underwater_pass_end(VkExec *ex) {
    if (!U.on || !U.src_addr || !U.gdepth) return;
    const VkTexImage *src = NULL, *hd = NULL;
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++) {
        const VkExecTarget *t = &ex->color[i];
        if (!full_size(ex, t) || !t->img) continue;
        if (t->img->addr == U.src_addr && t->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) src = t->img;
        else if (t->format == VK_FORMAT_R32_SFLOAT) hd = t->img;
    }
    if (!src) return;
    U.src = src, U.written = 1;
    if (hd) U.hdepth = hd;   // a later write without its depth keeps the depth of the frame's earlier one
    swap_off();              // ours shows the write before this one
    if (U.hdepth) U.pending = 1;
}

// A draw in a full-size G-buffer pass samples img: the first half-size R11G11B10F 2D image found so is the scene.
void vk_underwater_sampled(VkExec *ex, const VkTexImage *img) {
    if (U.src_addr || !img || img->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || img->type != VK_IMAGE_TYPE_2D)
        return;
    if (img->width * 2 + 1 < ex->pass_w || img->width * 2 > ex->pass_w + 1 || img->height * 2 + 1 < ex->pass_h ||
        img->height * 2 > ex->pass_h + 1)
        return;
    U.src_addr = img->addr, U.written = 1;   // written before this read, too early to be seen
    vk_underwater_watch = 0;
    if (U.logged < 4) {
        U.logged++;
        rt_log("vk: the game's half-size scene behind water found: %#llx, %ux%u, pass %u",
               (unsigned long long)img->addr, img->width, img->height, ex->pass_serial);
    }
}

static void vk_underwater_flush(VkExec *ex) {
    if (!U.pending || ex->in_pass) return;
    U.pending = 0;
    if (!U.on || !U.src || !U.hdepth || !U.gdepth) return;
    VkCtx *c = vk_ctx();
    if (!U.init_tried) init(c);
    if (!U.ok || !ensure_image(c, U.gdepth->width, U.gdepth->height)) return swap_off();
    VkImageView s = vk_exec_target_view(U.src, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_VIEW_TYPE_2D,
                                        VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView hd = vk_exec_target_view(U.hdepth, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
                                         VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView fd = vk_exec_target_view(U.gdepth, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
                                         VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    if (!s || !hd || !fd) return swap_off();
    vk_exec_barrier(ex);   // the game's writes; images in the feedback layout go back to GENERAL
    VkCommandBuffer cb = ex->cb;
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,   // earlier draws read ours
                                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                .oldLayout = U.fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = U.out.img,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                           .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    U.fresh = 0;
    VkDescriptorImageInfo ii[4] = {{U.point, s, VK_IMAGE_LAYOUT_GENERAL}, {U.point, hd, VK_IMAGE_LAYOUT_GENERAL},
                                   {U.point, fd, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, U.out.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[4];
    for (uint32_t i = 0; i < 4; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i,
                                      .descriptorCount = 1,
                                      .descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      .pImageInfo = &ii[i]};
    int32_t pc[2] = {(int32_t)U.w, (int32_t)U.h};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, U.pipe);
    rec_push(U.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, U.layout, 0, 4, w);
    vkCmdPushConstants(cb, U.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (U.w + 7) / 8, (U.h + 7) / 8, 1);
    if (U.debug) {   // magenta over the upsample: every pixel that shows ours shows magenta
        VkMemoryBarrier2 cb2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT};
        VkDependencyInfo cd = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                               .pMemoryBarriers = &cb2};
        vkCmdPipelineBarrier2(cb, &cd);
        VkClearColorValue mag = {.float32 = {4.0f, 0.0f, 4.0f, 1.0f}};
        VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, U.out.img, VK_IMAGE_LAYOUT_GENERAL, &mag, 1, &rr);
        ex->recorded += 2;
    }
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT};
    VkDependencyInfo md = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &md);
    ex->recorded += 6;
    if (vk_dump_armed)   // TOTK_DUMP: ours
        vk_dump_image(ex, U.out.img, VK_FORMAT_R16G16B16A16_SFLOAT, U.w, U.h, "underwater_upsample");
    // the game's bound compute pipeline and descriptors were replaced: bind them again; from here full-size passes
    // sample ours
    vk_exec_draw_new_cb();
    vk_underwater_swap_img = U.src, vk_underwater_swap_view = U.out.view;
}

void vk_underwater_release(void) {
    VkCtx *c = vk_ctx();
    vk_underwater_swap_img = NULL, vk_underwater_swap_view = VK_NULL_HANDLE, vk_underwater_watch = 0;
    if (!c || !U.tried) return;
    for (int i = 0; i < U.nold; i++) free_image(c, &U.old[i]);
    free_image(c, &U.out);
    if (U.pipe) vkDestroyPipeline(c->device, U.pipe, NULL);
    if (U.point) vkDestroySampler(c->device, U.point, NULL);
    if (U.layout) vkDestroyPipelineLayout(c->device, U.layout, NULL);
    if (U.dsl) vkDestroyDescriptorSetLayout(c->device, U.dsl, NULL);
    memset(&U, 0, sizeof U);
}
