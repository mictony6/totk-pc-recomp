// Depth-aware upsampling of the game's half-resolution effects (TOTK_EFFECTS_UPSAMPLE: game, smooth; read at start).
// TotK draws particles and other effects (the ice rod's glow, sparkles, auras) into a half-size RGBA16F target with a
// half-size depth test, then one full-screen draw blends that target onto the scene. The draw's fragment shader
// (1.4.3: SPIR-V e200d30e726bfe27 in a TOTK_DUMP) picks, per full-resolution pixel, a bilinear read pushed towards the
// half-size texels whose depth matches the pixel's; at an edge the read lands on one texel, so effects end in
// two-pixel steps. With smooth, that shader becomes shaders/effects_upsample.frag, a joint bilateral upsample of the
// same inputs (its header has the details). The translated shader is recognised by the hash of its SPIR-V, the one
// TOTK_DUMP prints, and replaced in memory only, as vk_hdr.c does for the tonemap.
#include "vk.h"

#include <string.h>

#include "cfg.h"
#include "effects_upsample_spv.h"
#include "gfx.h"
#include "host.h"
#include "platform/platform.h"

#define COMPOSITE_FS 0xe200d30e726bfe27ull   // the composite's SPIR-V, FNV-1a 64 of its bytes (vk_dump.c shader_hash)

// The game's variables, in the order of effects_upsample.frag's bindings 0..2.
static const char *const names[3] = {"fp_t_tcb_A", "fp_t_tcb_C", "fp_t_tcb_8"};

static uint64_t fnv1a(const uint32_t *w, uint64_t words) {
    const uint8_t *b = (const uint8_t *)w;
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < words * 4; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

void vk_upsample_shader_swap(uint32_t **spirv, uint64_t *words) {
    static int wanted = -1;
    if (wanted < 0) wanted = !strcmp(cfg_value("TOTK_EFFECTS_UPSAMPLE"), "smooth");
    if (!wanted || !*spirv || fnv1a(*spirv, *words) != COMPOSITE_FS) return;
    if (!vk_shader_swap_named(spirv, words, effects_upsample_spv, sizeof effects_upsample_spv / 4, names, 3)) return;
    static volatile int32_t swaps;
    rt_log("upsample: the game's effects composite runs as shaders/effects_upsample.frag (%d so far)", plat_inc32(&swaps));
}
