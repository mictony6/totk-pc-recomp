// Depth-aware upsampling of one of the game's half-resolution images to full resolution, shared by
// effects_upsample.frag (the effects composite) and scene_upsample.comp (the half-resolution scene the water shows).
// The includer defines UPSAMPLE_SRC (the half-resolution image) and UPSAMPLE_DEPTH (its linear depth, R32F at the same
// size), and UPSAMPLE_ALPHA 1 to filter alpha as well as RGB, before including this file.
// Two upsamples are mixed by how much the four half-resolution depths around the pixel disagree with the pixel's:
// - on one surface, AMD FidelityFX Super Resolution 1's edge-adaptive upsample (EASU), which follows the direction of
//   edges inside the image, so outlines drawn at half resolution without anti-aliasing come out as lines instead of
//   steps;
// - across a depth edge, a joint bilateral upsample: each of the four texels weighted by its bilinear weight times how
//   close its depth is to the pixel's, so texels on another surface drop out smoothly. The depths are linear
//   (proportional to distance), so differences are taken relative to the pixel's depth, and counted from the closest of
//   the four: on a surface thinner than a half-resolution texel no depth matches, and the closest texel still wins.
// ffx/ffx_a.h and ffx/ffx_fsr1.h are AMD's headers, unmodified (MIT, ffx/LICENSE.txt). EASU filters RGB; with
// UPSAMPLE_ALPHA it runs a second time on alpha.

const float UPSAMPLE_TOLERANCE = 0.02;   // relative depth difference at which a texel's bilateral weight halves

#define A_GPU 1
#define A_GLSL 1
#include "ffx/ffx_a.h"
#define FSR_EASU_F 1
int easu_alpha;   // 0: EASU's three gathers read red, green, blue; 1: all three read alpha
// (textureGather's component has to be a constant)
AF4 FsrEasuRF(AF2 p) { return easu_alpha != 0 ? textureGather(UPSAMPLE_SRC, p, 3) : textureGather(UPSAMPLE_SRC, p, 0); }
AF4 FsrEasuGF(AF2 p) { return easu_alpha != 0 ? textureGather(UPSAMPLE_SRC, p, 3) : textureGather(UPSAMPLE_SRC, p, 1); }
AF4 FsrEasuBF(AF2 p) { return easu_alpha != 0 ? textureGather(UPSAMPLE_SRC, p, 3) : textureGather(UPSAMPLE_SRC, p, 2); }
#include "ffx/ffx_fsr1.h"

// uv: the pixel's position in the half-resolution image (0..1); d: the pixel's full-resolution linear depth
vec4 upsample_at(vec2 uv, float d) {
    ivec2 size = textureSize(UPSAMPLE_SRC, 0);
    vec2 p = uv * vec2(size) - 0.5;   // the half-resolution position, texel centres at whole numbers

    // joint bilateral upsample, and how much the depths disagree
    ivec2 b = ivec2(floor(p));
    vec2 f = p - floor(p);
    // textureGather order: x (0,1), y (1,1), z (1,0), w (0,0) from the cell's lower-left texel b
    vec4 g = textureGather(UPSAMPLE_DEPTH, uv, 0);
    vec4 wb = vec4((1.0 - f.x) * f.y, f.x * f.y, f.x * (1.0 - f.y), (1.0 - f.x) * (1.0 - f.y));
    vec4 diff = abs(g - d);
    float best = min(min(diff.x, diff.y), min(diff.z, diff.w));
    vec4 r = (diff - best) / max(d, 1e-12) / UPSAMPLE_TOLERANCE;
    vec4 w = wb / (1.0 + r * r) + 1e-6;   // the small floor keeps the sum above zero at a corner of the cell
    ivec2 hi = size - 1;
    vec4 bilateral = (texelFetch(UPSAMPLE_SRC, clamp(b + ivec2(0, 1), ivec2(0), hi), 0) * w.x +
                      texelFetch(UPSAMPLE_SRC, clamp(b + ivec2(1, 1), ivec2(0), hi), 0) * w.y +
                      texelFetch(UPSAMPLE_SRC, clamp(b + ivec2(1, 0), ivec2(0), hi), 0) * w.z +
                      texelFetch(UPSAMPLE_SRC, clamp(b, ivec2(0), hi), 0) * w.w) / (w.x + w.y + w.z + w.w);
    // the texels the bilinear weights favour that lie on another surface: 0 on one surface, 1 across an edge
    float edge = clamp(dot(wb, min(r * r, vec4(1.0))) * 2.0, 0.0, 1.0);
    if (edge >= 1.0) return bilateral;

    // EASU at p: con0 normally maps an output pixel to p; here it is p itself (output pixel 0, offset p)
    AU4 c0, c1, c2, c3;
    FsrEasuCon(c0, c1, c2, c3, float(size.x), float(size.y), float(size.x), float(size.y), float(size.x),
               float(size.y));
    c0 = AU4(floatBitsToUint(0.0), floatBitsToUint(0.0), floatBitsToUint(p.x), floatBitsToUint(p.y));
    AF3 rgb;
    easu_alpha = 0;
    FsrEasuF(rgb, AU2(0u), c0, c1, c2, c3);
    float a = bilateral.a;
#if UPSAMPLE_ALPHA
    AF3 aaa;
    easu_alpha = 1;
    FsrEasuF(aaa, AU2(0u), c0, c1, c2, c3);
    a = aaa.x;
#endif
    return mix(vec4(rgb, a), bilateral, edge);
}
