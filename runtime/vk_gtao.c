// Ambient occlusion (TOTK_AO: game, gtao; TOTK_AO_STRENGTH percent, TOTK_AO_RADIUS metres, TOTK_AO_FOLIAGE minimal or
// full, TOTK_AO_GRASS percent of the occlusion vegetation keeps; all apply in play).
// TotK computes its SSAO at a quarter of the screen per axis, in one draw that writes an RGBA8 mask: sun shadow in red,
// a distance term in green, the AO in blue, a screen-space shadow term in alpha. Two passes then blur the mask and
// bend the AO through a contrast curve, and the half-resolution lighting reads the mask with one bilinear sample at a
// screen position. The game's AO is a thin outline along edges with little occlusion in corners and under objects, and
// at a quarter of the screen it is blocky and unsteady when the camera moves.
// With gtao, once the game finished its mask and before the next pass, three compute passes make a half-resolution
// mask, and the lighting samples it in place of the game's (vk_gtao_swap_img and vk_gtao_swap_view, applied by
// vk_exec_draw.c tex_info until the next G-buffer):
//   runtime/shaders/gtao_main.comp     ground-truth AO (XeGTAO, MIT) at half resolution from the G-buffer's full-resolution
//                                      linear depth, its normal turned towards the shading normals of target 3 by at
//                                      most about 18 degrees, which also darken by up to 10 % where the normal maps
//                                      tilt; vegetation (TOTK_AO_FOLIAGE=minimal) from its material class, green
//                                      of target 0 (32), and its base colour, target 1, as the field grass writes the
//                                      terrain class; characters (class 253) keep the game's AO
//   runtime/shaders/gtao_denoise.comp  an edge-aware blur, in place of the game's two
//   runtime/shaders/gtao_compose.comp  a second blur, and the game's finished mask with GTAO in its blue channel
// The camera comes from the fragment stage's uniform block c4 as in vk_contact.c: projection scales c4[7].x and
// c4[8].y, near and far planes c4[14].x and c4[14].y. The G-buffer's linear depth is (view z - near) / (far - near), as
// its fragment shaders write it (1.4.3: G-buffer fragment shader 6c08a63ccc131487).
// The game's AO draws are told apart by their pass, not their shader, which has variants (1.4.3: fragment shader guest
// hash ebaf90f76afc748b in the field, 1b096b4b3e0a804a in shrines): the raw AO draw is the first draw after the G-buffer
// into a pass of five or more colour targets whose target 0 is RGBA8 at a quarter of the G-buffer's size per axis; a
// blur writes another image; the finished mask is the next draw, in a later pass, into the raw draw's image.
//
// TOTK_AO_TRACE=<frames>: one log line per frame for that many frames after GTAO first runs: the G-buffer passes and
// resets, the raw AO draw's and the finished mask's pass, whether GTAO ran (else why not), and how many sampled
// descriptors of the game's mask were given GTAO's (vk_gtao_swap_hits, counted in vk_exec_draw.c tex_info).
//
// TOTK_AO_LOG=<frame>: every draw of that frame into an RGBA8 target at most a third as wide as the widest pass seen,
// in a pass with five or more colour targets, is logged with its fragment shader's guest hash, to check the AO passes
// in another scene or game version.
#include "vk.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gtao_compose_spv.h"
#include "gtao_denoise_spv.h"
#include "gtao_main_spv.h"
#include "host.h"
#include "vk_exec.h"
#include "vk_rec.h"

enum { AO_GAME, AO_GTAO };
enum { P_MAIN, P_DENOISE, P_COMPOSE, NPIPE };
enum { I_AO, I_AO2, I_EDGES, I_OUT, NIMG };   // RG8 visibility and kept flag (two), R8 edges, RGBA8 mask for the lighting

typedef struct {
    VkImage img;
    VmaAllocation alloc;
    VkImageView view;
} Img;

const VkTexImage *vk_gtao_swap_img;
VkImageView vk_gtao_swap_view;
uint32_t vk_gtao_swap_hits;

static struct {   // TOTK_AO_TRACE, this frame
    uint32_t left, gbuf, resets, raw_pass, fin_pass, ran;
    const char *why;
} T;

static struct {
    int tried, mode, ok, init_tried;
    unsigned cfg_ver;
    float strength, radius, thin, fade; // thin: XeGTAO's thin occluder compensation, fade: foliage fade (TOTK_AO_FOLIAGE)
    float grass;                        // how much AO vegetation receives (TOTK_AO_GRASS, percent, 100)
    float char_occ;                     // how much characters occlude what is around them (TOTK_AO_CHAR, percent, 25)
    int denoise;                        // denoise passes, the compose's included (TOTK_AO_DENOISE)
    float slices, steps;                // horizon directions and samples per direction (TOTK_AO_SLICES, TOTK_AO_STEPS)
    float micro;                        // most the normal maps darken by themselves (TOTK_AO_MICRO, percent of 0.10, 100)
    float tilt;                         // radians the search normal may turn towards the shading normal (TOTK_AO_TILT, degrees, 18)
    float crease;                       // crease emphasis (TOTK_AO_CREASE, percent: 0 none, 100 a half-visible corner to a quarter)
    float kept_ao;                      // the AO characters get: 108/255, as every character pixel of the game's
                                        // finished mask (1.4.3); TOTK_AO_KEPT (0..255) overrides it for tests
    uint64_t log_frame, frame, gb_frame;   // gb_frame: the frame whose G-buffer was seen last
    uint32_t logged, max_w;            // max_w: the widest pass seen (TOTK_AO_LOG)
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;          // sampled at 0, 1, 2, 5, 6; storage at 3, 4 (each pass uses some)
    VkPipelineLayout layout;
    VkPipeline pipe[NPIPE];
    VkSampler point, linear;
    Img img[NIMG], old[8][NIMG];        // old: earlier sizes, which recorded work may still use (freed at release)
    int nold, fresh;                    // fresh: the images are still in UNDEFINED layout
    uint32_t w, h;                      // GTAO size: half the G-buffer's
    const VkTexImage *depth;            // the G-buffer's linear depth
    const VkTexImage *material;         // the G-buffer's target 0 (RGBA8: green is the material class); NULL: none seen
    const VkTexImage *albedo;           // its target 1 (RGBA8 sRGB: base colour)
    const VkTexImage *gnormal;          // its target 3 (RGBA8: shading normal); NULL: normals from depth
    uint32_t dw, dh;
    uint64_t c4;                        // the camera block's guest address
    const VkTexImage *mask;             // the game's AO mask: the raw AO draw's image
    uint32_t raw_pass;                  // the raw AO draw's pass (VkExec.pass_serial)
    int raw_seen, done;                 // this frame's raw AO draw was found; its finished mask too (reset at a G-buffer)
    int pending;                        // the mask is finished: GTAO before the next pass
} G;

static void swap_off(void) {
    if (!vk_gtao_swap_img) return;
    vk_gtao_swap_img = NULL, vk_gtao_swap_view = VK_NULL_HANDLE;
    vk_exec_draw_new_cb();   // a draw with the same bindings as the one before must not keep the swapped view
}

static int read_mode(void) { return !strcmp(cfg_value("TOTK_AO"), "gtao") ? AO_GTAO : AO_GAME; }

static void update_mode(void) {
    unsigned v = cfg_version();
    if (G.tried && v == G.cfg_ver) return;
    if (!G.tried) {
        const char *e = cfg_get("TOTK_AO_LOG");
        G.log_frame = e && *e ? strtoull(e, NULL, 0) : 0;
        const char *tr = cfg_get("TOTK_AO_TRACE");
        T.left = tr && *tr ? (uint32_t)strtoul(tr, NULL, 0) : 0;
    }
    int m = read_mode(), first = !G.tried;
    G.tried = 1, G.cfg_ver = v;
    int s = cfg_int("TOTK_AO_STRENGTH");
    G.strength = (float)(s < 0 ? 0 : s > 300 ? 300 : s) / 100.0f;
    double r = cfg_double("TOTK_AO_RADIUS");
    G.radius = (float)(r < 0.25 ? 0.25 : r > 5 ? 5 : r);
    const char *sl = cfg_get("TOTK_AO_SLICES"), *st = cfg_get("TOTK_AO_STEPS");
    int nsl = sl && *sl ? atoi(sl) : 3, nst = st && *st ? atoi(st) : 6;   // 6 steps: smooth with per-pixel noise
    G.slices = (float)(nsl < 1 ? 1 : nsl > 16 ? 16 : nsl), G.steps = (float)(nst < 1 ? 1 : nst > 32 ? 32 : nst);
    const char *dn = cfg_get("TOTK_AO_DENOISE");
    int nd = dn && *dn ? atoi(dn) : 6;   // 6: no visible grain from the per-pixel noise (2 left it, 4 nearly)
    G.denoise = nd < 1 ? 1 : nd > 8 ? 8 : nd;
    int cr = cfg_int("TOTK_AO_CREASE");
    G.crease = (float)(cr < 0 ? 0 : cr > 100 ? 100 : cr) / 100.0f;
    int tl = cfg_int("TOTK_AO_TILT");
    G.tilt = (float)(tl < 0 ? 0 : tl > 90 ? 90 : tl) * 0.017453293f;
    int mc = cfg_int("TOTK_AO_MICRO");
    G.micro = 0.10f * (float)(mc < 0 ? 0 : mc > 300 ? 300 : mc) / 100.0f;
    int cp = cfg_int("TOTK_AO_CHAR");
    G.char_occ = (float)(cp < 0 ? 0 : cp > 100 ? 100 : cp) / 100.0f;
    int gr = cfg_int("TOTK_AO_GRASS");
    G.grass = (float)(gr < 0 ? 0 : gr > 100 ? 100 : gr) / 100.0f;
    const char *kv = cfg_get("TOTK_AO_KEPT");
    int ka = kv && *kv ? atoi(kv) : 108;
    G.kept_ao = (float)(ka < 0 ? 0 : ka > 255 ? 255 : ka) / 255.0f;
    // minimal: close to the game's own AO, which leaves vegetation untouched: thin occluders count less (the top of
    // XeGTAO's range), and vegetation (the G-buffer's material class) mostly neither receives nor casts occlusion
    // (gtao_main.comp). full: GTAO's plain result.
    int minimal = strcmp(cfg_value("TOTK_AO_FOLIAGE"), "full") != 0;
    G.thin = minimal ? 0.7f : 0.0f, G.fade = minimal ? 0.85f : 0.0f;
    if (m == G.mode && !first) return;
    G.mode = m;
    if (m != AO_GTAO) swap_off();
    rt_log(m == AO_GTAO ? "vk: ambient occlusion GTAO at half resolution in place of the game's SSAO, strength %.0f%%, "
                          "radius %.2f m, foliage %s"
                        : "vk: ambient occlusion: the game's own",
           G.strength * 100.0f, G.radius, G.thin > 0 ? "minimal" : "full");
}

static int make_image(VkCtx *c, Img *m, VkFormat f, uint32_t w, uint32_t h) {
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = f, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &ai, &m->img, &m->alloc, NULL) != VK_SUCCESS) return m->img = VK_NULL_HANDLE, 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = m->img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, &m->view) == VK_SUCCESS;
}

static void free_image(VkCtx *c, Img *m) {
    if (m->view) vkDestroyImageView(c->device, m->view, NULL);
    if (m->img) vmaDestroyImage(c->vma, m->img, m->alloc);
    memset(m, 0, sizeof *m);
}

// The pipelines and samplers: once, the first time GTAO runs.
static int init(VkCtx *c) {
    G.init_tried = 1;
    G.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[7];
    static const uint32_t binding[7] = {0, 1, 2, 5, 6, 3, 4};
    for (int i = 0; i < 7; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = binding[i], .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .descriptorType = i < 5 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                                      : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 7, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 96};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &G.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    if (!G.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &G.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &G.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &G.point) != VK_SUCCESS)
        goto fail;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    if (vkCreateSampler(c->device, &si, NULL, &G.linear) != VK_SUCCESS) goto fail;
    static const struct { const uint32_t *code; size_t size; } spv[NPIPE] = {
        {gtao_main_spv, sizeof gtao_main_spv}, {gtao_denoise_spv, sizeof gtao_denoise_spv},
        {gtao_compose_spv, sizeof gtao_compose_spv}};
    for (int i = 0; i < NPIPE; i++) {
        VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spv[i].size,
                                       .pCode = spv[i].code};
        VkShaderModule m = VK_NULL_HANDLE;
        if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) goto fail;
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                          .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                          .layout = G.layout};
        VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &G.pipe[i]);
        vkDestroyShaderModule(c->device, m, NULL);
        if (r != VK_SUCCESS) goto fail;
    }
    return G.ok = 1;
fail:
    rt_log("vk: GTAO off: its pipelines could not be created");
    return 0;
}

static int ensure_images(VkCtx *c, uint32_t w, uint32_t h) {
    static const VkFormat fmt[NIMG] = {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8_UNORM,
                                       VK_FORMAT_R8G8B8A8_UNORM};
    if (G.img[0].img && G.w == w && G.h == h) return 1;
    if (G.img[0].img) {
        if (G.nold == 8) return 0;
        memcpy(G.old[G.nold++], G.img, sizeof G.img);
        memset(G.img, 0, sizeof G.img);
    }
    for (int i = 0; i < NIMG; i++)
        if (!make_image(c, &G.img[i], fmt[i], w, h)) {
            for (int k = 0; k < NIMG; k++) free_image(c, &G.img[k]);
            rt_log("vk: GTAO off: its %ux%u images could not be created", w, h);
            G.ok = 0;
            return 0;
        }
    G.w = w, G.h = h, G.fresh = 1;
    rt_log("vk: GTAO at %ux%u from a %ux%u depth", w, h, G.dw, G.dh);
    return 1;
}

static void to_general(VkCommandBuffer cb, VkImage img) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
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

static VkWriteDescriptorSet write_of(uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo *ii) {
    return (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = binding,
                                  .descriptorCount = 1, .descriptorType = type, .pImageInfo = ii};
}

// One pass: sampled images in[] at bindings 0, 1, 2, 5, 6 (VK_NULL_HANDLE: unused), storage images at 3 and 4.
static void run(VkCommandBuffer cb, int p, const VkImageView in[5], const VkSampler smp[5], VkImageView out3,
                VkImageView out4, const float *pc, uint32_t pc_bytes) {
    static const uint32_t binding[5] = {0, 1, 2, 5, 6};
    VkDescriptorImageInfo ii[7];
    VkWriteDescriptorSet w[7];
    uint32_t n = 0;
    for (int i = 0; i < 5; i++)
        if (in[i]) ii[n] = (VkDescriptorImageInfo){smp[i], in[i], VK_IMAGE_LAYOUT_GENERAL},
                   w[n] = write_of(binding[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii[n]), n++;
    ii[n] = (VkDescriptorImageInfo){VK_NULL_HANDLE, out3, VK_IMAGE_LAYOUT_GENERAL};
    w[n] = write_of(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii[n]), n++;
    if (out4) {
        ii[n] = (VkDescriptorImageInfo){VK_NULL_HANDLE, out4, VK_IMAGE_LAYOUT_GENERAL};
        w[n] = write_of(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii[n]), n++;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe[p]);
    rec_push(G.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.layout, 0, n, w);
    vkCmdPushConstants(cb, G.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    vkCmdDispatch(cb, (G.w + 7) / 8, (G.h + 7) / 8, 1);
    mem_barrier(cb);
}

// The game finished its AO mask: GTAO's half-resolution mask, outside any pass, and the lighting reads it from now on.
void vk_gtao_flush(VkExec *ex) {
    if (!G.pending) return;
    G.pending = 0;
    T.why = ex->in_pass ? "in a pass" : !G.ok ? "not set up" : !G.depth ? "no depth" : !G.material ? "no material" :
            !G.albedo ? "no albedo" : !G.c4 ? "no camera" : NULL;
    if (T.why) return;
    const float *u = (const float *)(uintptr_t)G.c4;
    float px = u[7 * 4 + 0], py = u[8 * 4 + 1], znear = u[14 * 4 + 0], zfar = u[14 * 4 + 1];
    if (!(px > 0) || !(py > 0) || !(zfar > 1) || !(znear >= 0 && znear < zfar)) {
        T.why = "camera block not a projection";
        return;
    }
    VkCtx *c = vk_ctx();
    if (!ensure_images(c, (G.dw + 1) / 2, (G.dh + 1) / 2)) {
        T.why = "images";
        return;
    }
    VkImageView depth = vk_exec_target_view(G.depth, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
                                            VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView mask = vk_exec_target_view(G.mask, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D,
                                           VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView material = vk_exec_target_view(G.material, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D,
                                               VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView albedo = vk_exec_target_view(G.albedo, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_VIEW_TYPE_2D,
                                             VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView gnormal = G.gnormal ? vk_exec_target_view(G.gnormal, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D,
                                                          VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0)
                                    : VK_NULL_HANDLE;
    if (!depth || !mask || !material || !albedo) {
        T.why = "views";
        return;
    }
    T.ran++;
    vk_exec_barrier(ex);   // the mask's writes; images in the feedback layout go back to GENERAL
    VkCommandBuffer cb = ex->cb;
    int n = 1;
    if (G.fresh) {
        for (int i = 0; i < NIMG; i++) to_general(cb, G.img[i].img);
        G.fresh = 0, n += NIMG;
    }
    VkSampler pt[5] = {G.point, G.point, G.point, G.point, G.point};
    float mpc[24] = {px, py, zfar - znear, G.radius,
                     (float)G.w, (float)G.h, 1.0f / (float)G.w, 1.0f / (float)G.h,
                     (float)G.dw, (float)G.dh, 1.0f / (float)G.dw, 1.0f / (float)G.dh,
                     G.slices, G.steps, 0.615f, 2.2f,   // slices, steps (TOTK_AO_SLICES, TOTK_AO_STEPS), falloff range, power
                     G.thin, G.fade, gnormal ? 1.0f : 0.0f, znear,   // TOTK_AO_FOLIAGE; the G-buffer's normals
                     G.char_occ, G.grass, G.tilt, G.micro};
    run(cb, P_MAIN, (VkImageView[5]){depth, VK_NULL_HANDLE, material, albedo, gnormal}, pt, G.img[I_AO].view,
        G.img[I_EDGES].view, mpc, sizeof mpc);
    // denoise passes back and forth between the two visibility images, the compose's own the last: XeGTAO leaves the
    // rest to a temporal filter, which this has not, so more passes than its two
    float dpc[4] = {1.2f / 5.0f, 0, 0, 0};        // XeGTAO's denoise blur beta, / 5 for a pass that is not the last
    int src = I_AO, dst = I_AO2;
    for (int k = 1; k < G.denoise; k++) {
        run(cb, P_DENOISE, (VkImageView[5]){G.img[src].view, G.img[I_EDGES].view, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                            VK_NULL_HANDLE}, pt,
            G.img[dst].view, VK_NULL_HANDLE, dpc, sizeof dpc);
        int t = src;
        src = dst, dst = t;
    }
    float cpc[8] = {1.2f, G.strength, G.kept_ao, G.crease, 0, 0, 0, 0};   // blur beta, strength as a power, characters' AO,
                                                                          // crease emphasis; the shadow's depth allowance
    // the shadow channel follows depth where vk_shadowblur.c has a depth at the mask's size; both views are bound either
    // way, because the shader declares them
    const VkTexImage *md = vk_shadowblur_depth(G.mask);
    VkImageView mdv = md ? vk_exec_target_view(md, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                               0, 0, 1, 0)
                         : VK_NULL_HANDLE;
    cpc[4] = mdv ? vk_shadowblur_slope() : 0.0f;
    run(cb, P_COMPOSE, (VkImageView[5]){G.img[src].view, G.img[I_EDGES].view, mask, mdv ? mdv : depth, depth},
        (VkSampler[5]){G.point, G.point, G.linear, G.point, G.point}, G.img[I_OUT].view, VK_NULL_HANDLE, cpc, sizeof cpc);
    ex->recorded += n + (2 + G.denoise) * 5;
    if (vk_dump_armed) {   // TOTK_DUMP: the game's finished mask and ours, AO in blue, to compare the two AO passes
        vk_dump_image(ex, G.mask->image, VK_FORMAT_R8G8B8A8_UNORM, G.mask->width, G.mask->height, "ao_game_mask");
        vk_dump_image(ex, G.img[I_OUT].img, VK_FORMAT_R8G8B8A8_UNORM, G.w, G.h, "ao_gtao_mask");
    }
    // the game's bound compute pipeline and descriptors were replaced: bind them again; and from here the lighting
    // samples the half-resolution mask
    vk_exec_draw_new_cb();
    vk_gtao_swap_img = G.mask, vk_gtao_swap_view = G.img[I_OUT].view;
}

void vk_gtao_present(uint64_t frame) {
    __atomic_store_n(&G.frame, frame, __ATOMIC_RELAXED);
    if (!T.left || G.mode != AO_GTAO || (!G.ok && !T.raw_pass)) return;
    T.left--;
    rt_log("vk_gtao trace: frame %llu: G-buffer passes %u (resets %u), raw AO pass %u, finished mask pass %u, GTAO %s%s, "
           "swapped reads %u", (unsigned long long)frame, T.gbuf, T.resets, T.raw_pass, T.fin_pass,
           T.ran ? "ran" : "did not run", T.why ? (const char *)"" : "", vk_gtao_swap_hits);
    if (T.why) rt_log("vk_gtao trace: frame %llu: not run: %s", (unsigned long long)frame, T.why);
    uint32_t left = T.left;
    memset(&T, 0, sizeof T);
    T.left = left;
    vk_gtao_swap_hits = 0;
}

// A G-buffer pass: a D32F depth and, all of the pass's size (at least 640 wide), the material classes (target 0, RGBA8),
// the albedo (target 1, RGBA8 sRGB), the shading normals (target 3, RGBA8) and an R32F linear depth; returns that depth
// and sets the others, else NULL. All of them: a frame can begin with passes into a D32F depth and another R32F image
// alone (1.4.3, outdoors: 2 draws into an R32F that is not the G-buffer's linear depth), which as GTAO's depth gave
// blank AO with black blocks.
static const VkTexImage *gbuffer_depth(const VkExec *ex, const VkTexImage **material, const VkTexImage **albedo,
                                       const VkTexImage **gnormal) {
    *material = *albedo = *gnormal = NULL;
    if (ex->pass_noatt || ex->pass_w < 640 || ex->depth.format != VK_FORMAT_D32_SFLOAT || ex->depth.level) return NULL;
    const VkExecTarget *m = &ex->color[0];
    if (m->image && !m->level && m->width == ex->pass_w && m->height == ex->pass_h &&
        m->format == VK_FORMAT_R8G8B8A8_UNORM)
        *material = m->img;
    const VkExecTarget *a = &ex->color[1];
    if (ex->pass_ncolor > 1 && a->image && !a->level && a->width == ex->pass_w && a->height == ex->pass_h &&
        a->format == VK_FORMAT_R8G8B8A8_SRGB)
        *albedo = a->img;
    const VkExecTarget *nt = &ex->color[3];
    if (ex->pass_ncolor > 3 && nt->image && !nt->level && nt->width == ex->pass_w && nt->height == ex->pass_h &&
        nt->format == VK_FORMAT_R8G8B8A8_UNORM)
        *gnormal = nt->img;
    if (!*material || !*albedo || !*gnormal) return *material = *albedo = *gnormal = NULL, NULL;
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++) {
        const VkExecTarget *t = &ex->color[i];
        if (t->image && !t->level && t->width == ex->pass_w && t->height == ex->pass_h &&
            t->format == VK_FORMAT_R32_SFLOAT)
            return t->img;
    }
    *material = *albedo = *gnormal = NULL;
    return NULL;
}

void vk_gtao_pass_begin(VkExec *ex) {
    update_mode();
    if (ex->pass_w > G.max_w) G.max_w = ex->pass_w;
    if (G.mode == AO_GTAO) {
        // the camera block: c4 of the fragment stage, 0x940 bytes, as vk_contact.c finds it
        const p_bind_buf *b = &ex->ubo[1][1];
        if (b->addr && b->size == 0x940) G.c4 = b->addr;
        // A new frame's G-buffer: the game's AO passes read its own mask again. Only the first G-buffer pass after a
        // present: a frame can draw into its G-buffer again between the lighting passes (1.4.3, the Depths), and the
        // lighting after that must keep reading GTAO's mask.
        const VkTexImage *mat, *alb, *gn, *d = gbuffer_depth(ex, &mat, &alb, &gn);
        uint64_t f = __atomic_load_n(&G.frame, __ATOMIC_RELAXED);
        if (d) T.gbuf++;
        if (d && (f != G.gb_frame || !G.depth)) {
            T.resets++;
            G.gb_frame = f;
            G.depth = d, G.material = mat, G.albedo = alb, G.gnormal = gn, G.dw = ex->pass_w, G.dh = ex->pass_h;
            G.raw_seen = G.done = 0;
            swap_off();
        }
    }
    vk_gtao_flush(ex);
}

static void log_draw(VkExec *ex) {
    if (__atomic_load_n(&G.frame, __ATOMIC_RELAXED) + 1 != G.log_frame || G.logged >= 400) return;
    const VkExecTarget *t = &ex->color[0];
    if (ex->pass_ncolor < 5 || !t->image || t->format != VK_FORMAT_R8G8B8A8_UNORM || t->width * 3 > G.max_w) return;
    uint32_t len = 0;
    uint64_t h = vk_exec_fs_hash(ex, &len);
    G.logged++;
    rt_log("vk_gtao: frame %llu pass %u draw into %ux%u (%u targets): fs %016llx (%u bytes)",
           (unsigned long long)G.log_frame, ex->pass_serial, t->width, t->height, ex->pass_ncolor,
           (unsigned long long)h, len);
}

// A draw about to be recorded, its pass open.
void vk_gtao_draw(VkExec *ex) {
    update_mode();
    if (G.log_frame) log_draw(ex);
    if (G.mode != AO_GTAO || G.done || !G.depth || ex->pass_ncolor < 5) return;
    const VkExecTarget *t = &ex->color[0];
    if (!t->img || t->level || t->format != VK_FORMAT_R8G8B8A8_UNORM || t->width < 64) return;
    if (t->width * 4 + 4 < G.dw || t->width * 4 > G.dw + 4 || t->height * 4 + 4 < G.dh || t->height * 4 > G.dh + 4) return;
    if (!G.raw_seen) {   // the raw AO draw
        G.raw_seen = 1, G.mask = t->img, G.raw_pass = ex->pass_serial, T.raw_pass = ex->pass_serial;
        return;
    }
    if (t->img != G.mask || ex->pass_serial == G.raw_pass) return;   // a blur into the other image, or the same pass
    G.done = 1, T.fin_pass = ex->pass_serial;   // the finished mask
    if (!G.init_tried) init(vk_ctx());
    if (G.ok) G.pending = 1;
}

void vk_gtao_release(void) {
    VkCtx *c = vk_ctx();
    vk_gtao_swap_img = NULL, vk_gtao_swap_view = VK_NULL_HANDLE;
    if (!c || !G.tried) return;
    for (int i = 0; i < G.nold; i++)
        for (int k = 0; k < NIMG; k++) free_image(c, &G.old[i][k]);
    for (int k = 0; k < NIMG; k++) free_image(c, &G.img[k]);
    for (int i = 0; i < NPIPE; i++)
        if (G.pipe[i]) vkDestroyPipeline(c->device, G.pipe[i], NULL);
    if (G.point) vkDestroySampler(c->device, G.point, NULL);
    if (G.linear) vkDestroySampler(c->device, G.linear, NULL);
    if (G.layout) vkDestroyPipelineLayout(c->device, G.layout, NULL);
    if (G.dsl) vkDestroyDescriptorSetLayout(c->device, G.dsl, NULL);
    memset(&G, 0, sizeof G);
}
