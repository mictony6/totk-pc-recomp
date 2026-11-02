#version 450
#extension GL_GOOGLE_include_directive : require
// The game's composite of its half-resolution effects (particles, the ice rod's glow, sparkles, the electric dome) onto
// the full-resolution scene. runtime/vk_upsample.c vk_upsample_shader_swap puts it in place of the translated shader
// with TOTK_EFFECTS_UPSAMPLE=smooth.
// The game's shader reads the pixel's full-resolution linear depth (fp_t_tcb_A), gathers the four half-resolution
// depths around it (fp_t_tcb_C) and moves one bilinear read of the effects (fp_t_tcb_8) towards the texels whose depth
// is closest, clamped to the 2x2 cell. Where the depths differ a lot (an edge) the read lands on a corner texel: each
// full-resolution pixel takes one half-resolution texel whole, and effects cut by geometry end in two-pixel steps.
// The effects themselves are drawn at half resolution without anti-aliasing, so their own outlines (the dome's rim)
// are stair-stepped in the half-resolution target already. upsample_common.glsl handles both.
// Bindings 0..2 stand for the game's fp_t_tcb_A, _C and _8: vk_upsample.c gives each the original's binding and set.
layout(location = 0) in vec4 uv;   // xy: the pixel's position in the half-resolution textures (0..1)
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D fp_t_tcb_A;   // full-resolution linear depth
layout(set = 0, binding = 1) uniform sampler2D fp_t_tcb_C;   // half-resolution linear depth
layout(set = 0, binding = 2) uniform sampler2D fp_t_tcb_8;   // half-resolution effects

#define UPSAMPLE_SRC fp_t_tcb_8
#define UPSAMPLE_DEPTH fp_t_tcb_C
#define UPSAMPLE_ALPHA 1   // the composite's blend uses alpha
#include "upsample_common.glsl"

void main() { color = upsample_at(uv.xy, texture(fp_t_tcb_A, uv.xy).x); }
