#version 450
// The game's tonemap pass for SDR output with a white point (runtime/vk_hdr.c vk_hdr_shader_swap puts it in place of the
// translated shader when HDR is off). Same inputs and bindings as the game's shader and as hdr_tonemap.frag: fp_t_tcb_8
// (bloom, clamped to 10000 as the game does) plus fp_t_tcb_A (the scene), weights in c1[0] and the saturation grade in
// c3[0]. The game maps luminance and each channel with 1 - e^-x, blends the two by the luminance curve squared, clamps to
// 0..1 and grades saturation; this does the same on a compressed input.
// White point (support_buffer.white.x, TOTK_WHITE_POINT, 1 = the game's own curve): 1 - e^-x flattens fast, so scene
// values of 3 and 6 end up at 0.95 and 0.9975 and the highlight detail between them is lost. The input is first passed
// through f(x) = x / (1 + (w - 1) * x / (x + 1)) on each channel: f(x) is x at w = 1 and x / w for large x, so a higher
// white point spreads bright values over more of the curve (they come out somewhat darker, with more gradation). The
// compression starts at low values too, so mid-tones are darker and slightly less saturated than the game's.
// Bindings are those of the game's translated shader: vk_hdr.c rewrites them from the original's names.
layout(location = 0) in vec4 uv;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0, std140) uniform SupportBuffer {
    vec4 pad[90];
    vec4 hdr;     // byte 1440, read by hdr_tonemap.frag only
    vec4 white;   // byte 1456 (vk_exec_draw.c support_fill): x white point
} support_buffer;
layout(set = 0, binding = 1) uniform sampler2D fp_t_tcb_8;
layout(set = 0, binding = 2) uniform sampler2D fp_t_tcb_A;
layout(set = 0, binding = 3, std140) uniform C1 { vec4 v[4096]; } fp_c1;
layout(set = 0, binding = 4, std140) uniform C3 { vec4 v[4096]; } fp_c3;

float maxc(vec3 c) { return max(c.r, max(c.g, c.b)); }

void main() {
    vec3 x = min(texture(fp_t_tcb_8, uv.xy).rgb, vec3(10000.0)) + texture(fp_t_tcb_A, uv.xy).rgb;
    vec4 w = fp_c1.v[0], g = fp_c3.v[0];
    float white = max(support_buffer.white.x, 1.0);
    x = x / (vec3(1.0) + (white - 1.0) * x / (x + vec3(1.0)));

    float lum = max(x.r * 0.2989 + x.g * w.x + x.b * w.y, 0.0);
    float sdr = 1.0 - exp(-lum);
    vec3 c = clamp(mix(lum > 0.0 ? x * (sdr / lum) : x, vec3(1.0) - exp(-x), sdr * sdr), 0.0, 1.0);
    float m = maxc(c), d = c.r * 0.6666667 + (c.g + c.b) * w.z - 1.0;
    c = (c - m) * (g.y * (1.0 - d * d) + g.x) + m;
    color = vec4(max(c, vec3(0.0)), 0.0);
}
