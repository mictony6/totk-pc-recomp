// GTAO denoise (runtime/vk_gtao.c; included by gtao_denoise.comp and gtao_compose.comp): an edge-aware 3x3 blur of the
// visibility, adapted from XeGTAO's XeGTAO_Denoise and XeGTAO_UnpackEdges (XeGTAO.hlsli, GameTechDev/XeGTAO a5b1686):
//
//   Copyright (C) 2016-2021, Intel Corporation
//   SPDX-License-Identifier: MIT
//
// Changes: GLSL, one pixel per invocation with texel fetches instead of two per thread with gathers; pixels whose game
// value is kept (characters, gtao_main.comp) keep it and stay out of their neighbours' blur.
// The including shader declares: sampler2D ao (x: visibility / OCCLUSION_TERM_SCALE, y: kept), sampler2D edges.

#define OCCLUSION_TERM_SCALE 1.5

vec4 unpack_edges(float v) {
    uint p = uint(v * 255.5);
    return clamp(vec4((p >> 6) & 3u, (p >> 4) & 3u, (p >> 2) & 3u, p & 3u) / 3.0, 0.0, 1.0);
}

vec2 vis(ivec2 p) {
    return texelFetch(ao, clamp(p, ivec2(0), textureSize(ao, 0) - 1), 0).xy;
}

// a neighbour's weighted visibility (x) and its weight (y); kept pixels weigh nothing
vec2 tap(ivec2 p, float w) {
    vec2 v = vis(p);
    w *= 1.0 - v.y;
    return vec2(v.x * w, w);
}

vec4 edge(ivec2 p) {
    return unpack_edges(texelFetch(edges, clamp(p, ivec2(0), textureSize(edges, 0) - 1), 0).x);
}

// The denoised visibility of p (same scale as ao.x) and its kept flag; beta: the centre's weight (XeGTAO's blur beta)
vec2 denoise(ivec2 p, float beta) {
    vec2 c = vis(p);
    if (c.y > 0.5) return c;
    vec4 el = edge(p + ivec2(-1, 0)), er = edge(p + ivec2(1, 0)), et = edge(p + ivec2(0, -1)), eb = edge(p + ivec2(0, 1));
    vec4 ec = edge(p) * vec4(el.y, er.x, et.w, eb.z);   // both sides of an edge agree
    // a little leaking where there are three or four edges, against isolated noisy pixels
    const float leak_threshold = 2.5, leak_strength = 0.5;
    float edginess = (clamp(4.0 - leak_threshold - dot(ec, vec4(1.0)), 0.0, 1.0) / (4.0 - leak_threshold)) * leak_strength;
    ec = clamp(ec + edginess, 0.0, 1.0);
    const float diag = 0.85 * 0.5;
    float wtl = diag * (ec.x * el.z + ec.z * et.x), wtr = diag * (ec.z * et.y + ec.y * er.z);
    float wbl = diag * (ec.w * eb.x + ec.x * el.w), wbr = diag * (ec.y * er.w + ec.w * eb.y);
    vec2 s = vec2(c.x * beta, beta);
    s += tap(p + ivec2(-1, 0), ec.x) + tap(p + ivec2(1, 0), ec.y) + tap(p + ivec2(0, -1), ec.z) + tap(p + ivec2(0, 1), ec.w);
    s += tap(p + ivec2(-1, -1), wtl) + tap(p + ivec2(1, -1), wtr) + tap(p + ivec2(-1, 1), wbl) + tap(p + ivec2(1, 1), wbr);
    return vec2(s.x / s.y, 0.0);
}
