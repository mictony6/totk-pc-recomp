// SMAA 1x (smaa/SMAA.hlsl, MIT, Jorge Jimenez et al.) as compute shaders, for vk_aa.c. Each pass runs one invocation
// per pixel at texcoord = (pixel + 0.5) / size and does what SMAA's vertex and pixel shader pair does there. SMAA's
// porting layer maps onto explicit-LOD sampling (compute shaders have no derivatives); its "discard" in edge
// detection becomes "no edge", as every pixel is written. Push constant: SMAA_RT_METRICS = (1/w, 1/h, w, h).
layout(local_size_x = 8, local_size_y = 8) in;
layout(push_constant) uniform Pc { vec4 rt; } pc;

#define SMAA_RT_METRICS pc.rt
#define SMAA_PRESET_HIGH
#define SMAA_CUSTOM_SL 1
#define SMAATexture2D(tex) sampler2D tex
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) textureLod(tex, coord, 0.0)
#define SMAASampleLevelZeroPoint(tex, coord) textureLod(tex, coord, 0.0)
#define SMAASampleLevelZeroOffset(tex, coord, offset) textureLodOffset(tex, coord, 0.0, offset)
#define SMAASample(tex, coord) textureLod(tex, coord, 0.0)
#define SMAASamplePoint(tex, coord) textureLod(tex, coord, 0.0)
#define SMAASampleOffset(tex, coord, offset) textureLodOffset(tex, coord, 0.0, offset)
#define SMAAGather(tex, coord) textureGather(tex, coord)
#define SMAA_FLATTEN
#define SMAA_BRANCH
#define lerp(a, b, t) mix(a, b, t)
#define saturate(a) clamp(a, 0.0, 1.0)
#define mad(a, b, c) fma(a, b, c)
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4
#define bool2 bvec2
#define bool3 bvec3
#define bool4 bvec4
#define discard return float2(0.0, 0.0)
#include "smaa/SMAA.hlsl"
#undef discard

// This invocation's pixel, or -1 outside the image.
ivec2 smaa_pixel(out vec2 texcoord) {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    texcoord = (vec2(p) + 0.5) * pc.rt.xy;
    return all(lessThan(p, ivec2(pc.rt.zw))) ? p : ivec2(-1);
}
