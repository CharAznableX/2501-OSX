#include <metal_stdlib>
using namespace metal;

static float orbHash(float2 p) {
    float3 p3 = fract(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

static float orbNoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(orbHash(i), orbHash(i + float2(1, 0)), u.x),
        mix(orbHash(i + float2(0, 1)), orbHash(i + float2(1, 1)), u.x),
        u.y);
}

static float orbFbm(float2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 3; i++) {
        v += a * orbNoise(p);
        float2 r = float2(p.x * 0.8 + p.y * 0.6, -p.x * 0.6 + p.y * 0.8);
        p = r * 2.0 + 100.0;
        a *= 0.5;
    }
    return v;
}

[[ stitchable ]]
half4 orbEffect(float2 position, half4 currentColor, float time, float seed, float4 bounds) {
    if (bounds.z <= 0.0 || bounds.w <= 0.0) {
        return half4(0.0);
    }
    float2 uv = position / bounds.zw;
    // Guard against potential NaN/Inf by clamping uv to a reasonable range
    uv = clamp(uv, -1.0, 2.0);
    float3 base = float3(currentColor.rgb);

    float n1 = orbFbm(uv * 3.0 + float2(time * 0.18 + seed * 10.0, time * 0.14));
    float n2 = orbFbm(uv * 3.5 + float2(-time * 0.12 + seed * 7.0, time * 0.16 + seed * 3.0));
    float2 d = uv + float2(n1, n2) * 0.15;

    float grad = smoothstep(0.0, 1.0, d.y * 0.7 + 0.2);
    grad += orbFbm(d * 2.5 + float2(time * 0.08, 0.0)) * 0.25;

    float h1 = smoothstep(0.48, 0.60, orbFbm(d * 5.0 + float2(time * 0.22 + seed * 5.0, seed)));
    float h2 = smoothstep(0.52, 0.65, orbFbm(d * 4.0 + float2(seed * 8.0, -time * 0.16)));
    float hl = max(h1 * 0.9, h2 * 0.65);

    float dist = length(uv - 0.5) * 2.0;
    float core = smoothstep(1.0, 0.0, dist) * (0.45 + 0.15 * sin(time * 0.8 + seed * 6.28));
    float rim = smoothstep(0.55, 1.0, dist) * 0.35;
    float inner = orbFbm(d * 6.0 + float2(time * 0.05, 0.0)) * 0.1;

    float alpha = smoothstep(1.0, 0.88, dist);

    float gf = grad * 0.85 + 0.15;
    float3 col = base * gf;
    col = mix(col, float3(1.0), hl);
    col += base * core * 0.55;
    col += float3(1.0) * rim * 0.25;
    col += base * inner;
    col = saturate(col);

    return half4(half3(col * alpha), half(alpha));
}
