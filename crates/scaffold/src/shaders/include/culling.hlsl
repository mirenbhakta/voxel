// Frustum culling utilities.

#ifndef CULLING_HLSL
#define CULLING_HLSL

static const float3 CHUNK_HALF_EXTENT = float3(16.0, 16.0, 16.0);

/// Test a chunk AABB against six frustum planes (p-vertex method).
///
/// Each plane is (nx, ny, nz, d) with inward-pointing normal.
/// Returns true if the chunk is inside or intersecting the frustum.
bool frustum_test(float4 planes[6], float3 chunk_center) {
    for (uint i = 0; i < 6; i++) {
        float3 normal = planes[i].xyz;
        float  d      = planes[i].w;

        // P-vertex: AABB corner most in the direction of the plane normal.
        float3 p_vertex = chunk_center + sign(normal) * CHUNK_HALF_EXTENT;

        if (dot(normal, p_vertex) + d < 0.0) {
            return false;
        }
    }

    return true;
}

#endif // CULLING_HLSL
