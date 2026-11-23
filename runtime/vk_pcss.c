// Soft sun shadows (TOTK_PCSS): the eight-tap disc of translator patch 0010 turned by a different angle at each pixel.
//
// Patch 0010 (tools/xlate/patches/0010-pcss-for-cascade-compares.patch, GeneratePcssSample) replaces the lighting
// shaders' single cascade compare with eight compares at uv + off_i * radius, off_i a fixed Poisson disc. Every pixel
// samples the same eight points, so once the radius spans a few shadow map texels an edge shows as eight shifted
// copies stacked on each other. Turned by an angle that changes from pixel to pixel (interleaved gradient noise of the
// fragment position, Jimenez, "Next generation post processing in Call of Duty: Advanced Warfare", 2014), neighbours
// sample different points and the game's blur of its quarter-size lighting image averages them into a gradient.
//
// The translator that is installed may predate any change to the patch, so the turn is added here, to the translated
// SPIR-V of each fragment shader as it is loaded (gfx_shader.c, beside vk_hdr_shader_swap): the disc's eight
// OpConstantComposite vec2 values are recognised by their components, and each OpFMul that scales one of them gets the
// turned offset instead. The angle is worked out once per block, before its first such multiply. Shaders without a
// FragCoord input or the GLSL.std.450 import are left alone. The edit runs only inside patch 0010's soft shadow
// branch, so with TOTK_PCSS at 0 the game's single compare is unchanged.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "host.h"
#include "platform/platform.h"

enum {
    OP_NAME = 5, OP_EXT_INST_IMPORT = 11, OP_EXT_INST = 12, OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23, OP_TYPE_POINTER = 32,
    OP_CONSTANT = 43, OP_CONSTANT_COMPOSITE = 44, OP_FUNCTION = 54, OP_VARIABLE = 59, OP_LOAD = 61, OP_DECORATE = 71,
    OP_COMPOSITE_CONSTRUCT = 80, OP_COMPOSITE_EXTRACT = 81, OP_FADD = 129, OP_FSUB = 131, OP_FMUL = 133, OP_LABEL = 248,
    DEC_BUILTIN = 11, BUILTIN_FRAG_COORD = 15, GLSL_FRACT = 10, GLSL_SIN = 13, GLSL_COS = 14,
    HEAD_WORDS = 63,   // the angle: load, two extracts, the dot product, two fracts, the scales, cos and sin
    TURN_WORDS = 35,   // one offset: four multiplies, a subtract, an add, the vec2
};

// PcssDisk of patch 0010, as pairs
static const float disc[16] = {
    -0.9420f, -0.3991f, 0.9456f, -0.7689f, -0.0942f, -0.9294f, 0.3450f, 0.2939f,
    -0.9159f, 0.4577f, -0.8154f, -0.8791f, -0.3828f, 0.2768f, 0.9748f, 0.7565f,
};

static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static int is_glsl(const uint32_t *a, uint32_t len) {   // OpExtInstImport operands after the result id
    static const char s[] = "GLSL.std.450";
    return (len - 2) * 4 >= sizeof s && !memcmp(a + 1, s, sizeof s);
}

void vk_pcss_shader_turn(uint32_t **spirv, uint64_t *words) {
    static int off = -1;   // TOTK_PCSS_TURN=0 (debug, environment only): leave the shaders as translated, for timing
    if (off < 0) {
        const char *e = getenv("TOTK_PCSS_TURN");
        off = e && !strcmp(e, "0");
    }
    if (off || !*spirv || *words < 5) return;
    const uint32_t *w = *spirv;
    uint64_t n = *words;
    uint32_t bound = w[3];
    if (!bound) return;
    uint32_t glsl = 0, f32 = 0, v2 = 0, frag = 0, disc_id[16] = {0};
    uint64_t first_fn = 0;
    // the module's declarations: the import, float and vec2, the FragCoord variable's id, the disc's scalars
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return;
        const uint32_t *a = w + i + 1;
        if (op == OP_FUNCTION) {
            first_fn = i;
            break;
        }
        if (op == OP_EXT_INST_IMPORT && len >= 3 && is_glsl(a, len)) glsl = a[0];
        else if (op == OP_TYPE_FLOAT && len == 3 && a[1] == 32) f32 = a[0];
        else if (op == OP_TYPE_VECTOR && len == 4 && a[1] == f32 && f32 && a[2] == 2) v2 = a[0];
        else if (op == OP_DECORATE && len == 4 && a[1] == DEC_BUILTIN && a[2] == BUILTIN_FRAG_COORD) frag = a[0];
        else if (op == OP_CONSTANT && len == 4 && a[0] == f32 && f32)
            for (int k = 0; k < 16; k++)
                if (a[2] == fbits(disc[k]) && !disc_id[k]) disc_id[k] = a[1];
        i += len;
    }
    if (!first_fn || !glsl || !f32 || !v2 || !frag) return;
    for (int k = 0; k < 16; k++)
        if (!disc_id[k]) return;
    // the disc's vec2 constants (id -> pair index + 1), the FragCoord variable's vec4 type
    uint8_t *pair = calloc(bound, 1);
    uint32_t *ptr_pointee = calloc(bound, sizeof *ptr_pointee);
    if (!pair || !ptr_pointee) {
        free(pair), free(ptr_pointee);
        return;
    }
    uint32_t v4 = 0;
    int pairs = 0;
    for (uint64_t i = 5; i < first_fn; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == OP_CONSTANT_COMPOSITE && len == 5 && a[0] == v2 && a[1] < bound) {
            for (int k = 0; k < 8; k++)
                if (a[2] == disc_id[2 * k] && a[3] == disc_id[2 * k + 1] && !pair[a[1]]) pair[a[1]] = (uint8_t)(k + 1), pairs++;
        } else if (op == OP_TYPE_POINTER && len == 4 && a[0] < bound) {
            ptr_pointee[a[0]] = a[2];
        } else if (op == OP_VARIABLE && len >= 4 && a[1] == frag && a[0] < bound) {
            v4 = ptr_pointee[a[0]];
        }
    }
    free(ptr_pointee);
    if (pairs < 8 || !v4) {
        free(pair);
        return;
    }
    // the multiplies to turn, and the blocks they are in
    uint64_t turns = 0, heads = 0;
    int in_block = 0;
    for (uint64_t i = first_fn; i < n; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == OP_LABEL) in_block = 0;
        else if (op == OP_FMUL && len == 5 && a[0] == v2 &&
                 ((a[2] < bound && pair[a[2]]) || (a[3] < bound && pair[a[3]]))) {
            turns++;
            if (!in_block) heads++, in_block = 1;
        }
    }
    if (!turns) {
        free(pair);
        return;
    }
    uint64_t cap = n + 16 + heads * HEAD_WORDS + turns * TURN_WORDS;
    uint32_t *out = malloc(cap * sizeof *out);
    if (!out) {
        free(pair);
        return;
    }
    uint32_t next = bound;
    uint32_t k_x = next++, k_y = next++, k_ign = next++, k_tau = next++;   // the noise's constants
    uint64_t o = 0;
    memcpy(out, w, first_fn * sizeof *w);
    o = first_fn;
    const float kv[4] = {0.06711056f, 0.00583715f, 52.9829189f, 6.28318531f};
    const uint32_t kid[4] = {k_x, k_y, k_ign, k_tau};
    for (int k = 0; k < 4; k++) {
        out[o++] = 4u << 16 | OP_CONSTANT, out[o++] = f32, out[o++] = kid[k], out[o++] = fbits(kv[k]);
    }
    uint32_t cos_a = 0, sin_a = 0;
    in_block = 0;
#define I4(op, x, y, z) (out[o++] = 4u << 16 | (op), out[o++] = (x), out[o++] = (y), out[o++] = (z))
#define I5(op, x, y, z, u) (out[o++] = 5u << 16 | (op), out[o++] = (x), out[o++] = (y), out[o++] = (z), out[o++] = (u))
#define EXT(r, fn, x) (out[o++] = 6u << 16 | OP_EXT_INST, out[o++] = f32, out[o++] = (r), out[o++] = glsl, \
                       out[o++] = (fn), out[o++] = (x))
    for (uint64_t i = first_fn; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == OP_LABEL) in_block = 0;
        int which = op == OP_FMUL && len == 5 && a[0] == v2 ? (a[2] < bound && pair[a[2]] ? 2 : a[3] < bound && pair[a[3]] ? 3 : 0) : 0;
        if (which) {
            if (!in_block) {   // angle = 2 pi fract(52.98 fract(dot(FragCoord.xy, (0.0671, 0.00584))))
                uint32_t fc = next++, fx = next++, fy = next++, t1 = next++, t2 = next++, d = next++, f1 = next++,
                         m = next++, f2 = next++, ang = next++;
                cos_a = next++, sin_a = next++;
                I4(OP_LOAD, v4, fc, frag);
                I5(OP_COMPOSITE_EXTRACT, f32, fx, fc, 0);
                I5(OP_COMPOSITE_EXTRACT, f32, fy, fc, 1);
                I5(OP_FMUL, f32, t1, fx, k_x);
                I5(OP_FMUL, f32, t2, fy, k_y);
                I5(OP_FADD, f32, d, t1, t2);
                EXT(f1, GLSL_FRACT, d);
                I5(OP_FMUL, f32, m, f1, k_ign);
                EXT(f2, GLSL_FRACT, m);
                I5(OP_FMUL, f32, ang, f2, k_tau);
                EXT(cos_a, GLSL_COS, ang);
                EXT(sin_a, GLSL_SIN, ang);
                in_block = 1;
            }
            int k = pair[a[which]] - 1;   // the offset (ox, oy) turned: (ox c - oy s, ox s + oy c)
            uint32_t ox = disc_id[2 * k], oy = disc_id[2 * k + 1];
            uint32_t p1 = next++, p2 = next++, rx = next++, p3 = next++, p4 = next++, ry = next++, v = next++;
            I5(OP_FMUL, f32, p1, ox, cos_a);
            I5(OP_FMUL, f32, p2, oy, sin_a);
            I5(OP_FSUB, f32, rx, p1, p2);
            I5(OP_FMUL, f32, p3, ox, sin_a);
            I5(OP_FMUL, f32, p4, oy, cos_a);
            I5(OP_FADD, f32, ry, p3, p4);
            I5(OP_COMPOSITE_CONSTRUCT, v2, v, rx, ry);
            uint32_t ins[5] = {a[-1], a[0], a[1], a[2], a[3]};
            ins[which + 1] = v;   // a[k] is ins[k + 1]
            memcpy(out + o, ins, sizeof ins);
            o += 5;
        } else {
            memcpy(out + o, w + i, len * sizeof *w);
            o += len;
        }
        i += len;
    }
#undef I4
#undef I5
#undef EXT
    free(pair);
    out[3] = next;
    free(*spirv);
    *spirv = out;
    *words = o;
    static volatile int32_t count;
    int32_t c = plat_inc32(&count);
    if (c <= 3 || !(c & (c - 1)))
        rt_log("vk: soft shadow disc turned per pixel in a lighting shader (%llu offsets; %d shaders so far)",
               (unsigned long long)turns, c);
}
