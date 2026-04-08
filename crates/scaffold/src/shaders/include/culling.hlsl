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

/// Compute the frustum-visible layer range for a chunk direction.
///
/// For direction `dir`, the normal axis is dir/2 (0=X, 1=Y, 2=Z).
/// Layer l occupies [origin.axis + l, origin.axis + l + 1] along the
/// normal axis, with the full [0, 32) extent on the other two axes.
///
/// Each frustum plane gives a linear constraint on l via the p-vertex
/// test. Six constraints intersect to give [l_min, l_max].
///
/// Returns uint2(l_min, l_max) inclusive, clamped to [0, 31].
/// If l_min > l_max the direction is fully culled.
uint2 compute_layer_range(float4 frustum[6], float3 origin, uint dir) {
    uint axis = dir / 2;

    float l_min_f = 0.0;
    float l_max_f = 31.0;

    for (uint i = 0; i < 6; i++) {
        float3 n = frustum[i].xyz;
        float  d = frustum[i].w;

        // Normal-axis component of the plane normal.
        float n_ax = (axis == 0) ? n.x : (axis == 1) ? n.y : n.z;

        // Build p-vertex at l=0.
        //   Normal axis: [origin, origin + 1] -> pick +1 if n_ax > 0.
        //   Lateral axes: [origin, origin + 32] -> pick +32 if n > 0.
        float3 p0;
        p0.x = origin.x + ((axis != 0) ? (n.x > 0.0 ? 32.0 : 0.0)
                                        : (n.x > 0.0 ?  1.0 : 0.0));
        p0.y = origin.y + ((axis != 1) ? (n.y > 0.0 ? 32.0 : 0.0)
                                        : (n.y > 0.0 ?  1.0 : 0.0));
        p0.z = origin.z + ((axis != 2) ? (n.z > 0.0 ? 32.0 : 0.0)
                                        : (n.z > 0.0 ?  1.0 : 0.0));

        float C = dot(n, p0) + d;

        // Constraint: C + n_ax * l >= 0.
        if (n_ax > 0.0001) {
            l_min_f = max(l_min_f, -C / n_ax);
        }
        else if (n_ax < -0.0001) {
            l_max_f = min(l_max_f, -C / n_ax);
        }
    }

    // Clamp to valid layer range.
    int lo = (int)floor(l_min_f);
    int hi = (int)ceil(l_max_f);
    lo = clamp(lo, 0, 31);
    hi = clamp(hi, 0, 31);

    return uint2((uint)lo, (uint)hi);
}

#endif // CULLING_HLSL
