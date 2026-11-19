// Colour adjustment on the present (vk_color.h): brightness, contrast, saturation and gamma of the game's picture, after
// the game and before the FSR 1 pass or the blit scale it to the window. Settings TOTK_COLOR_BRIGHTNESS (-30..30, %
// added), TOTK_COLOR_CONTRAST and TOTK_COLOR_SATURATION (%, 100 = unchanged) and TOTK_COLOR_GAMMA (exponent, 1 =
// unchanged), read when the settings change; the menu's colour profiles only write these four. At the default values
// nothing is recorded and the present is what it was. Design and preset values: STS78721's colour profiles (issue #136).
//
// One compute pass (shaders/color_grade.comp) reads the crop rectangle of the game's image and writes an RGBA16F image
// of its size, which present_image then hands to FSR or the blit in place of the game's image. The HUD is part of that
// picture and is adjusted with it; the in-game menu is drawn later, straight into the swapchain image, and is not.
// With TOTK_COLOR_HUD = 0 the adjustment runs earlier instead, on the scene colour right after the game's tonemap draw
// and before the HUD (vk_color_scene_record, called from vk_aa.c), and the present leaves that frame alone.
// The source view of slot s is destroyed when slot s records again (present_image has waited for that slot's fence);
// the output image is shared by all slots, and each frame's first barrier orders it after the previous frame's reads.
#include "vk.h"

#include <string.h>

#include "cfg.h"
#include "color_grade_scene_spv.h"
#include "color_grade_spv.h"
#include "host.h"

#define SLOTS 8

static struct {
    int tried, ok;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler sampler;
    VkImage img;                   // the adjusted picture, RGBA16F at the crop size
    VmaAllocation alloc;
    VkImageView view;
    uint32_t w, h;
    VkImageView src_view[SLOTS];   // the source view each slot's last frame read
    unsigned seen;                 // cfg_version the values below were read at
    float adj[4];                  // brightness (added), contrast, saturation, gamma
    int hud;                       // TOTK_COLOR_HUD: 1 grades the HUD with the picture, 0 grades the scene before it
    // the scene pass (vk_color_scene_record): the adjustment before the HUD
    int scene_tried, scene_ok;
    VkDescriptorSetLayout scene_dsl;
    VkPipelineLayout scene_layout;
    VkPipeline scene_pipe;
    int scene_graded;              // the scene of the frame on its way to the present was adjusted already (atomic)
} C = {.seen = ~0u, .hud = 1};

static int init(VkCtx *c) {
    C.tried = 1;
    C.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 2, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &C.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof color_grade_spv,
                                   .pCode = color_grade_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!C.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &C.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &C.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &C.sampler) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) {
        rt_log("vk: colour adjustment off: its pipeline could not be created");
        return 0;
    }
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = C.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &C.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) {
        rt_log("vk: colour adjustment off: its pipeline could not be created");
        return 0;
    }
    return C.ok = 1;
}

static void free_image(VkCtx *c) {
    if (C.view) vkDestroyImageView(c->device, C.view, NULL);
    if (C.img) vmaDestroyImage(c->vma, C.img, C.alloc);
    C.view = VK_NULL_HANDLE, C.img = VK_NULL_HANDLE, C.w = C.h = 0;
}

static int ensure_image(VkCtx *c, uint32_t w, uint32_t h) {
    if (C.img && C.w == w && C.h == h) return 1;
    // another crop size (a resolution change): earlier presents may still read the old image
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    free_image(c);
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R16G16B16A16_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &C.img, &C.alloc, NULL) != VK_SUCCESS) {
        C.img = VK_NULL_HANDLE;
        return 0;
    }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = C.img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &C.view) != VK_SUCCESS) {
        C.view = VK_NULL_HANDLE;
        free_image(c);
        return 0;
    }
    C.w = w, C.h = h;
    return 1;
}

// The four values from the settings, re-read when they change. 1 when any differs from "unchanged".
static int adjusting(void) {
    unsigned v = cfg_version();
    if (v != C.seen) {
        C.seen = v;
        C.adj[0] = (float)cfg_double("TOTK_COLOR_BRIGHTNESS") / 100.0f;
        C.adj[1] = (float)cfg_double("TOTK_COLOR_CONTRAST") / 100.0f;
        C.adj[2] = (float)cfg_double("TOTK_COLOR_SATURATION") / 100.0f;
        C.adj[3] = (float)cfg_double("TOTK_COLOR_GAMMA");
        C.hud = cfg_bool("TOTK_COLOR_HUD");
    }
    return C.adj[0] != 0.0f || C.adj[1] != 1.0f || C.adj[2] != 1.0f || C.adj[3] != 1.0f;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkPipelineStageFlags2 ss, VkAccessFlags2 sa,
                    VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                               .dstStageMask = ds, .dstAccessMask = da, .oldLayout = from,
                               .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

// The scene pass below already adjusted this frame's picture before the HUD was drawn: the present leaves it alone. A
// frame without that pass takes the whole picture's adjustment here as before, except with TOTK_COLOR_HUD = 0 on 1.4.3,
// where such a frame is a menu, the map or the title screen and stays as the game drew it.
VkImage vk_color_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4]) {
    if (__atomic_exchange_n(&C.scene_graded, 0, __ATOMIC_ACQ_REL)) return VK_NULL_HANDLE;
    if (!adjusting()) return VK_NULL_HANDLE;
    // TOTK_COLOR_HUD = 0 where the scene pass knows the game's tonemap (1.4.3): a frame without that draw is a screen of
    // the game's menus, the map or the title, which have no scene to adjust, so they keep their colours as the HUD does
    if (!C.hud && rt_game_version == 143) return VK_NULL_HANDLE;
    VkCtx *c = vk_ctx();
    if (!C.tried) init(c);
    if (!C.ok || !ensure_image(c, (uint32_t)crop[2], (uint32_t)crop[3])) return VK_NULL_HANDLE;
    slot %= SLOTS;
    if (C.src_view[slot]) vkDestroyImageView(c->device, C.src_view[slot], NULL);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = src,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &C.src_view[slot]) != VK_SUCCESS) {
        C.src_view[slot] = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    const VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ALL = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    // the game's writes to src are visible; last frame's reads of the output are done (UNDEFINED: its old content goes)
    barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, ALL, VK_ACCESS_2_MEMORY_WRITE_BIT, CS, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    barrier(cb, C.img, VK_IMAGE_LAYOUT_UNDEFINED, ALL, 0, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    VkDescriptorImageInfo ii = {C.sampler, C.src_view[slot], VK_IMAGE_LAYOUT_GENERAL},
                          oi = {VK_NULL_HANDLE, C.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, C.pipe);
    rec_push(C.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, C.layout, 0, 2, w);
    struct { int32_t offset[2], size[2]; float adj[4]; } pc = {{crop[0], crop[1]}, {crop[2], crop[3]},
                                                              {C.adj[0], C.adj[1], C.adj[2], C.adj[3]}};
    vkCmdPushConstants(cb, C.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, ((uint32_t)crop[2] + 7) / 8, ((uint32_t)crop[3] + 7) / 8, 1);
    // later submissions that write src wait for this read; FSR and the blit read the output
    barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, CS, 0, ALL, 0);
    barrier(cb, C.img, VK_IMAGE_LAYOUT_GENERAL, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, ALL,
            VK_ACCESS_2_MEMORY_READ_BIT);
    return C.img;
}

// TOTK_COLOR_HUD = 0 on an SDR output with an adjustment set: the scene is adjusted where the game's tonemap wrote it.
int vk_color_scene_wanted(void) {
    return adjusting() && !C.hud && !vk_hdr_wanted();
}

static int init_scene(VkCtx *c) {
    C.scene_tried = 1;
    if (!C.tried) init(c);
    if (!C.ok) return 0;
    VkDescriptorSetLayoutBinding b = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
                                      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 1, .pBindings = &b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &C.scene_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof color_grade_scene_spv, .pCode = color_grade_scene_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &C.scene_dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &C.scene_layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) {
        rt_log("vk: colour adjustment before the HUD off: its pipeline could not be created");
        return 0;
    }
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = C.scene_layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &C.scene_pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) {
        rt_log("vk: colour adjustment before the HUD off: its pipeline could not be created");
        return 0;
    }
    rt_log("vk: colour adjustment before the HUD (TOTK_COLOR_HUD = 0): the HUD keeps its own colours");
    return C.scene_ok = 1;
}

// Adjusts the tonemapped scene colour in place, outside any pass; the caller has made the tonemap's writes visible and
// orders the HUD's draws after this one. Returns 1 when the dispatch was recorded.
int vk_color_scene_record(VkCommandBuffer cb, VkImageView scene, uint32_t w, uint32_t h) {
    VkCtx *c = vk_ctx();
    if (!C.scene_tried) init_scene(c);
    if (!C.scene_ok) return 0;
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, scene, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, C.scene_pipe);
    rec_push(C.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, C.scene_layout, 0, 1, &wr);
    struct { int32_t size[2], pad[2]; float adj[4]; } pc = {{(int32_t)w, (int32_t)h}, {0, 0}, {C.adj[0], C.adj[1], C.adj[2], C.adj[3]}};   // adj sits at 16 as a vec4
    vkCmdPushConstants(cb, C.scene_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    __atomic_store_n(&C.scene_graded, 1, __ATOMIC_RELEASE);
    return 1;
}

void vk_color_release(void) {
    VkCtx *c = vk_ctx();
    if (C.scene_pipe) vkDestroyPipeline(c->device, C.scene_pipe, NULL);
    if (C.scene_layout) vkDestroyPipelineLayout(c->device, C.scene_layout, NULL);
    if (C.scene_dsl) vkDestroyDescriptorSetLayout(c->device, C.scene_dsl, NULL);
    C.scene_pipe = VK_NULL_HANDLE, C.scene_layout = VK_NULL_HANDLE, C.scene_dsl = VK_NULL_HANDLE;
    C.scene_tried = C.scene_ok = 0;
    for (int i = 0; i < SLOTS; i++)
        if (C.src_view[i]) vkDestroyImageView(c->device, C.src_view[i], NULL), C.src_view[i] = VK_NULL_HANDLE;
    free_image(c);
    if (C.pipe) vkDestroyPipeline(c->device, C.pipe, NULL);
    if (C.sampler) vkDestroySampler(c->device, C.sampler, NULL);
    if (C.layout) vkDestroyPipelineLayout(c->device, C.layout, NULL);
    if (C.dsl) vkDestroyDescriptorSetLayout(c->device, C.dsl, NULL);
    C.pipe = VK_NULL_HANDLE, C.sampler = VK_NULL_HANDLE, C.layout = VK_NULL_HANDLE, C.dsl = VK_NULL_HANDLE;
    C.tried = C.ok = 0;
}
