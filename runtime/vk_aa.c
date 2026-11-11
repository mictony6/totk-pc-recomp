// Anti-aliasing (TOTK_AA: game, off, smaa). TotK tonemaps its scene colour in place (a full-size B10G11R11 target, one
// draw), then, with FXAA on (ModelSceneExtensionParam EnableFXAA, docked), copies it to a second full-size image and
// runs FXAA from there back into the scene colour, and then draws the HUD into that same image. The draws are told
// apart by the guest code of their fragment shader (vk_exec_fs_hash), which does not change with the translator:
//   tonemap 7e114c4330dc212a, FXAA copy 974a5ded2d5d2b19, FXAA ca674e9d11936eed (1.4.3)
// game leaves all of it alone. off skips the two FXAA draws. smaa skips them too and, once the tonemap draw ran, runs
// SMAA 1x (runtime/shaders/smaa_*.comp, smaa/SMAA.hlsl, MIT, Jorge Jimenez et al.) on the scene colour when the next
// pass begins or a dispatch or copy comes first: before the HUD, so its text stays sharp. SMAA reads an sRGB-encoded
// RGBA8 copy (smaa_prep: luma edges in gamma space, blending in linear light through an sRGB view), keeps edges and
// weights in images of its own, and writes the blended pixels back; the scene images always get STORAGE usage
// (vk_texture.c create_tex), so the setting applies in play: it is read again whenever the settings change
// (cfg_version). SMAA's pipelines and images are made the first time it is on and kept until release.
//
// TOTK_AA_LOG=<frame>: every draw of that frame into a full-size B10G11R11 target is logged with its fragment shader's
// guest hash, in replay order, to find those shaders in another game version.
#include "vk.h"

#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "shaders/smaa/AreaTex.h"
#include "shaders/smaa/SearchTex.h"
#include "smaa_blend_spv.h"
#include "smaa_edges_spv.h"
#include "smaa_prep_spv.h"
#include "smaa_weights_spv.h"
#include "vk_exec.h"
#include "vk_rec.h"

#define FS_TONEMAP 0x7e114c4330dc212aull
#define FS_FXAA_COPY 0x974a5ded2d5d2b19ull
#define FS_FXAA 0xca674e9d11936eedull

enum { AA_GAME, AA_OFF, AA_SMAA };
enum { P_PREP, P_EDGES, P_WEIGHTS, P_BLEND, NPASS };

typedef struct {
    VkImage img;
    VmaAllocation alloc;
    VkImageView view, view_srgb;   // view_srgb: the copy only
} Img;

static struct {
    int tried, mode, ok, logged_fxaa;
    int smaa_tried;                // init_smaa ran (ok: it succeeded)
    unsigned cfg_ver;              // cfg_version when mode was read
    uint64_t log_frame;            // TOTK_AA_LOG
    uint64_t frame;                // presents so far (vk_aa_present)
    uint32_t logged;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe[NPASS];
    VkSampler linear, point;
    Img area, search, copy, edges, weights;
    VkBuffer staging;              // the lookup textures' bytes until release
    VmaAllocation staging_alloc;
    int uploaded, fresh;           // fresh: copy, edges and weights still in UNDEFINED layout
    Img old[8][3];                 // images of earlier sizes; recorded work may still use them, freed at release
    int nold;
    uint32_t w, h;
    const VkTexImage *scene;       // the tonemapped scene colour waiting for SMAA
    int pending;
} A;

static int mode(void) {
    const char *m = cfg_value("TOTK_AA");
    return !strcmp(m, "smaa") ? AA_SMAA : !strcmp(m, "off") ? AA_OFF : AA_GAME;
}

static int make_image(VkCtx *c, Img *m, VkFormat f, uint32_t w, uint32_t h, VkImageUsageFlags usage, int srgb) {
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .flags = srgb ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0, .imageType = VK_IMAGE_TYPE_2D,
                            .format = f, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &ai, &m->img, &m->alloc, NULL) != VK_SUCCESS) return m->img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = m->img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &m->view) != VK_SUCCESS) return 0;
    if (srgb) {   // sampled only: an sRGB format has no storage support
        VkImageViewUsageCreateInfo u = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                        .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
        vi.pNext = &u;
        vi.format = VK_FORMAT_R8G8B8A8_SRGB;
        if (vkCreateImageView(c->device, &vi, NULL, &m->view_srgb) != VK_SUCCESS) return 0;
    }
    return 1;
}

static void free_image(VkCtx *c, Img *m) {
    if (m->view) vkDestroyImageView(c->device, m->view, NULL);
    if (m->view_srgb) vkDestroyImageView(c->device, m->view_srgb, NULL);
    if (m->img) vmaDestroyImage(c->vma, m->img, m->alloc);
    memset(m, 0, sizeof *m);
}

static int init_smaa(VkCtx *c) {
    A.smaa_tried = 1;
    A.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
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
                                     .pSetLayouts = &A.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
                              .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    if (!A.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &A.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &A.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &A.linear) != VK_SUCCESS)
        goto fail;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    if (vkCreateSampler(c->device, &si, NULL, &A.point) != VK_SUCCESS) goto fail;
    static const struct { const uint32_t *code; size_t size; } spv[NPASS] = {
        {smaa_prep_spv, sizeof smaa_prep_spv}, {smaa_edges_spv, sizeof smaa_edges_spv},
        {smaa_weights_spv, sizeof smaa_weights_spv}, {smaa_blend_spv, sizeof smaa_blend_spv}};
    for (int i = 0; i < NPASS; i++) {
        VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spv[i].size,
                                       .pCode = spv[i].code};
        VkShaderModule m = VK_NULL_HANDLE;
        if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) goto fail;
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                          .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                          .layout = A.layout};
        VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &A.pipe[i]);
        vkDestroyShaderModule(c->device, m, NULL);
        if (r != VK_SUCCESS) goto fail;
    }
    // SMAA's lookup textures, uploaded by the first SMAA run
    VkImageUsageFlags lu = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!make_image(c, &A.area, VK_FORMAT_R8G8_UNORM, AREATEX_WIDTH, AREATEX_HEIGHT, lu, 0) ||
        !make_image(c, &A.search, VK_FORMAT_R8_UNORM, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, lu, 0))
        goto fail;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = AREATEX_SIZE + SEARCHTEX_SIZE,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &A.staging, &A.staging_alloc, &info) != VK_SUCCESS || !info.pMappedData)
        goto fail;
    memcpy(info.pMappedData, areaTexBytes, AREATEX_SIZE);
    memcpy((uint8_t *)info.pMappedData + AREATEX_SIZE, searchTexBytes, SEARCHTEX_SIZE);
    vmaFlushAllocation(c->vma, A.staging_alloc, 0, VK_WHOLE_SIZE);
    return A.ok = 1;
fail:
    rt_log("vk: SMAA off: its pipelines or lookup textures could not be created");
    return 0;
}

// Reads TOTK_AA at the first draw and after each settings change. A change in the middle of a frame is fine: a
// tonemap already caught still gets its SMAA run, and the FXAA draws are skipped or not from the next one on.
static void update_mode(void) {
    unsigned v = cfg_version();
    if (A.tried && v == A.cfg_ver) return;
    if (!A.tried) {
        const char *e = cfg_get("TOTK_AA_LOG");
        A.log_frame = e && *e ? strtoull(e, NULL, 0) : 0;
    }
    int m = mode(), first = !A.tried;
    A.tried = 1, A.cfg_ver = v;
    if (m == AA_SMAA && !A.smaa_tried) init_smaa(vk_ctx());
    if (m == A.mode && !first) return;
    A.mode = m, A.logged_fxaa = 0;
    if (m == AA_SMAA)
        rt_log(A.ok ? "vk: anti-aliasing SMAA 1x (high preset) in place of the game's FXAA, before the HUD"
                    : "vk: anti-aliasing SMAA unavailable; the game's FXAA is skipped");
    else
        rt_log(m == AA_OFF ? "vk: anti-aliasing off (the game's FXAA is skipped)" : "vk: anti-aliasing: the game's own (FXAA unless a mod turns it off)");
}

static int ensure_images(VkCtx *c, uint32_t w, uint32_t h) {
    if (A.copy.img && A.w == w && A.h == h) return 1;
    if (A.copy.img) {
        if (A.nold == 8) {   // a full list of retired sizes: wait for the GPU, free them (as vk_dlaa.c does)
            vkDeviceWaitIdle(c->device);
            for (int i = 0; i < A.nold; i++)
                for (int k = 0; k < 3; k++) free_image(c, &A.old[i][k]);
            A.nold = 0;
        }
        A.old[A.nold][0] = A.copy, A.old[A.nold][1] = A.edges, A.old[A.nold][2] = A.weights, A.nold++;
        memset(&A.copy, 0, sizeof A.copy), memset(&A.edges, 0, sizeof A.edges), memset(&A.weights, 0, sizeof A.weights);
    }
    VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!make_image(c, &A.copy, VK_FORMAT_R8G8B8A8_UNORM, w, h, u, 1) ||
        !make_image(c, &A.edges, VK_FORMAT_R8G8_UNORM, w, h, u, 0) ||
        !make_image(c, &A.weights, VK_FORMAT_R8G8B8A8_UNORM, w, h, u, 0)) {
        free_image(c, &A.copy), free_image(c, &A.edges), free_image(c, &A.weights);
        rt_log("vk: SMAA off: its %ux%u images could not be created", w, h);
        A.ok = 0;
        return 0;
    }
    A.w = w, A.h = h, A.fresh = 1;
    rt_log("vk: SMAA on a %ux%u scene", w, h);
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

static void mem_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &d);
}

static void upload_lookup(VkCommandBuffer cb) {
    barrier(cb, A.area.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    barrier(cb, A.search.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r = {.bufferOffset = 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                           .imageExtent = {AREATEX_WIDTH, AREATEX_HEIGHT, 1}};
    vkCmdCopyBufferToImage(cb, A.staging, A.area.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    r.bufferOffset = AREATEX_SIZE, r.imageExtent = (VkExtent3D){SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1};
    vkCmdCopyBufferToImage(cb, A.staging, A.search.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    barrier(cb, A.area.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, A.search.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
}

// One SMAA pass: nin sampled inputs (view, sampler) at bindings 0.. and a storage output at binding 3.
static void run(VkCommandBuffer cb, int p, int nin, const VkImageView *in, const VkSampler *smp, VkImageView out) {
    VkDescriptorImageInfo ii[4];
    VkWriteDescriptorSet w[4];
    for (int i = 0; i <= nin; i++) {
        int st = i == nin;   // the last one is the output
        ii[i] = (VkDescriptorImageInfo){st ? VK_NULL_HANDLE : smp[i], st ? out : in[i], VK_IMAGE_LAYOUT_GENERAL};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                      .dstBinding = st ? 3u : (uint32_t)i, .descriptorCount = 1,
                                      .descriptorType = st ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                           : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    float pc[4] = {1.0f / (float)A.w, 1.0f / (float)A.h, (float)A.w, (float)A.h};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.pipe[p]);
    rec_push(A.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.layout, 0, (uint32_t)nin + 1, w);
    vkCmdPushConstants(cb, A.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (A.w + 7) / 8, (A.h + 7) / 8, 1);
    mem_barrier(cb);
}

// The tonemap ran: SMAA on its scene colour, outside any pass.
void vk_aa_flush(VkExec *ex) {
    if (!A.pending) return;
    A.pending = 0;
    if (ex->in_pass || !A.ok) return;
    VkCtx *c = vk_ctx();
    if (!ensure_images(c, A.scene->width, A.scene->height)) return;
    VkImageView scene = vk_exec_target_view(A.scene, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_VIEW_TYPE_2D,
                                            VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    if (!scene) return;
    VkCommandBuffer cb = ex->cb;
    int n = 1;
    mem_barrier(cb);   // the tonemap's writes, before SMAA reads them
    if (!A.uploaded) upload_lookup(cb), A.uploaded = 1, n += 6;
    if (A.fresh) {
        barrier(cb, A.copy.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        barrier(cb, A.edges.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        barrier(cb, A.weights.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        A.fresh = 0, n += 3;
    }
    run(cb, P_PREP, 1, &scene, &A.point, A.copy.view);
    run(cb, P_EDGES, 1, &A.copy.view, &A.linear, A.edges.view);
    VkImageView wi[3] = {A.edges.view, A.area.view, A.search.view};
    VkSampler ws[3] = {A.linear, A.linear, A.point};
    run(cb, P_WEIGHTS, 3, wi, ws, A.weights.view);
    VkImageView bi[2] = {A.copy.view_srgb, A.weights.view};
    VkSampler bs[2] = {A.linear, A.linear};
    run(cb, P_BLEND, 2, bi, bs, scene);
    ex->recorded += n + 4 * 5;
    vk_exec_draw_new_cb(ex);   // the game's bound compute pipeline and descriptors were replaced: bind them again
}

void vk_aa_present(uint64_t frame) { __atomic_store_n(&A.frame, frame, __ATOMIC_RELAXED); }

static void log_draw(VkExec *ex) {
    if (__atomic_load_n(&A.frame, __ATOMIC_RELAXED) + 1 != A.log_frame || A.logged >= 1500) return;
    const VkExecTarget *t = &ex->color[0];
    if (ex->pass_ncolor != 1 || !t->image || t->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || t->width < 1280) return;
    uint32_t len = 0;
    uint64_t h = vk_exec_fs_hash(ex, &len);
    A.logged++;
    rt_log("vk_aa: frame %llu pass %u draw into %ux%u addr %#llx (depth fmt %d): fs %016llx (%u bytes)",
           (unsigned long long)A.log_frame, ex->pass_serial, t->width, t->height,
           (unsigned long long)(uintptr_t)t->tex->mem, (int)ex->depth.format, (unsigned long long)h, len);
}

// A draw about to be recorded, its pass open. Returns 1 when the draw is to be skipped.
int vk_aa_draw(VkExec *ex) {
    update_mode();
    if (A.log_frame) log_draw(ex);
    if (A.mode == AA_GAME) return 0;
    uint32_t len;
    uint64_t h = vk_exec_fs_hash(ex, &len);
    // The FXAA shader also does the scene-to-output scaling pass when the scene is smaller than the output; that pass
    // has no depth attachment and is the only thing that puts the 3D picture on screen (vk_dlaa.c vk_dlaa_skip_draw).
    if (h == FS_FXAA_COPY || (h == FS_FXAA && ex->depth.image)) {
        if (!A.logged_fxaa)
            A.logged_fxaa = 1, rt_log("vk: the game's FXAA pass is skipped (TOTK_AA=%s)", cfg_value("TOTK_AA"));
        return 1;
    }
    if (h != FS_TONEMAP || A.mode != AA_SMAA || !A.ok) return 0;
    const VkExecTarget *t = &ex->color[0];
    if (ex->pass_ncolor != 1 || !t->img || t->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || t->level ||
        t->width < 640 || !(t->img->usage & VK_IMAGE_USAGE_STORAGE_BIT))
        return 0;
    A.scene = t->img, A.pending = 1;
    return 0;
}

void vk_aa_release(void) {
    VkCtx *c = vk_ctx();
    if (!c || !A.tried) return;
    for (int i = 0; i < A.nold; i++)
        for (int k = 0; k < 3; k++) free_image(c, &A.old[i][k]);
    free_image(c, &A.copy), free_image(c, &A.edges), free_image(c, &A.weights);
    free_image(c, &A.area), free_image(c, &A.search);
    if (A.staging) vmaDestroyBuffer(c->vma, A.staging, A.staging_alloc);
    for (int i = 0; i < NPASS; i++)
        if (A.pipe[i]) vkDestroyPipeline(c->device, A.pipe[i], NULL);
    if (A.linear) vkDestroySampler(c->device, A.linear, NULL);
    if (A.point) vkDestroySampler(c->device, A.point, NULL);
    if (A.layout) vkDestroyPipelineLayout(c->device, A.layout, NULL);
    if (A.dsl) vkDestroyDescriptorSetLayout(c->device, A.dsl, NULL);
    memset(&A, 0, sizeof A);
}
