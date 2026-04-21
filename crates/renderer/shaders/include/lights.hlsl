// lights.hlsl — shader-side types for the light-dispatch lighting kernel
// (decision-lighting-dispatch-kernel-over-clustered).
//
// Every light — sun, point, spot, area — is handled by one shading pattern:
// some subset of pixels cast one DDA ray toward a world-space target. This
// header declares the data types; the shading kernel branches on `kind`.
//
// No bindings are declared here. Consumer shaders declare their own
// `[[vk::binding(N, S)]] ConstantBuffer<LightList> g_lights` matching the
// project's pure-include convention (`material.hlsl`, `camera.hlsl`).
//
// # Layout
//
// `LightDesc` is 48 bytes, aligned to 16 bytes (HLSL cbuffer array stride).
// Fields pack into three 16-byte rows:
//
//   row 0: position.xyz | kind
//   row 1: direction.xyz | radius
//   row 2: color.xyz | _pad
//
// `LightList` is 16 bytes header + 32 × 48 bytes = 1552 bytes total.
// Mirrored byte-for-byte by `renderer::LightList` on the Rust side.

#ifndef RENDERER_LIGHTS_HLSL
#define RENDERER_LIGHTS_HLSL

// ---- Capacity constant (keep in lockstep with Rust `MAX_LIGHTS`) -------
#define MAX_LIGHTS 32u

// ---- Light kind enum (keep in lockstep with Rust `LightKind`) ----------
#define LIGHT_KIND_DIRECTIONAL 0u
#define LIGHT_KIND_POINT       1u

struct LightDesc {
    // World-space position. Meaningful for point/spot; unused for directional.
    float3 position;
    // Discriminant — one of LIGHT_KIND_*.
    uint   kind;

    // Unit vector *toward* the light. Meaningful for directional; unused
    // for point (the shader derives it from `position - hit`).
    float3 direction;
    // Maximum effective range in world units. Meaningful for point/spot
    // (contribution clamps to 0 at d >= radius). Unused for directional.
    float  radius;

    // Linear-space radiance. For directional, this is the sun radiance at
    // the surface. For point, this is the per-light reference radiance —
    // it combines with the distance attenuation term inside the kernel.
    float3 color;
    float  _pad;
};

struct LightList {
    // Number of valid entries in `lights[0..count]`. Entries past `count`
    // are untouched by the shader and may hold stale data.
    uint      count;
    uint3     _pad;

    LightDesc lights[MAX_LIGHTS];
};

#endif // RENDERER_LIGHTS_HLSL
