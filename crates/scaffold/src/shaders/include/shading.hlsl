// Shading utilities: Lambert diffuse, edge outlines.

#ifndef SHADING_HLSL
#define SHADING_HLSL

/// Apply Lambert diffuse shading with ambient floor.
float3 apply_lambert(float3 base_color, float3 normal,
                     float3 sun_dir, float ambient) {
    float n_dot_l = max(dot(normal, sun_dir), 0.0);
    float light   = ambient + (1.0 - ambient) * n_dot_l;
    return base_color * light;
}

/// Apply quad edge outline darkening.
///
/// quad_uv: interpolated UV within the quad (0-1).
/// quad_size: quad dimensions in voxel units.
/// Returns the darkened color.
float3 apply_outline(float3 base_color, float2 quad_uv, float2 quad_size) {
    // Distance to nearest quad edge in voxel units.
    float2 d = quad_uv * quad_size;
    float edge_dist = min(
        min(d.x, quad_size.x - d.x),
        min(d.y, quad_size.y - d.y)
    );

    // Fixed-width edge line regardless of quad size.
    float edge = 1.0 - smoothstep(0.02, 0.06, edge_dist);
    return lerp(base_color, float3(0, 0, 0), edge * 0.8);
}

#endif // SHADING_HLSL
