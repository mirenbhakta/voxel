// Frustum culling compute shader.
//
// Iterates over all chunk slots, tests visible chunks against the
// frustum, and emits compacted MDI entries + per-draw metadata.
// Also accumulates per-frame statistics (visible quads, back-face
// culled quads) into the draw_count buffer for CPU readback.
//
// Dispatch: ceil(total_slots / 64) workgroups of 64 threads.
//
// Phase 3: per-(chunk, direction) entries with frustum layer-range culling.

#include "include/bindings.hlsl"
#include "include/culling.hlsl"

// Bindings (cull pass).
cbuffer CullUniforms : register(b0, space0) {
    float4 planes[6];
    float4 camera_pos;    // xyz: world-space camera position
};

ByteAddressBuffer   chunk_offsets  : register(t1, space0);
ByteAddressBuffer   chunk_meta_buf : register(t2, space0);
ByteAddressBuffer   quad_range_buf : register(t3, space0);
RWByteAddressBuffer dst_draws      : register(u4, space0);
RWByteAddressBuffer draw_data_buf  : register(u5, space0);
RWByteAddressBuffer draw_count     : register(u6, space0);

// draw_count layout:
//   [ 0]: visible draw count (MDI entries emitted)
//   [ 4]: total visible quads (sum of quad_count for frustum-visible chunks)
//   [ 8]: back-face culled quads (quads on entirely back-facing directions)
//   [12]: visible chunk count (chunks that passed frustum test)
//   [16]: layer-culled quads (quads outside the frustum-visible layer range)

// Push constants.
struct CullPush {
    uint total_slots;
};

[[vk::push_constant]] CullPush push;

/// Sum all 32 layer counts for a single direction.
uint sum_direction_quads(uint slot, uint dir) {
    uint base  = slot * 776 + 8 + dir * 128;
    uint total = 0;

    for (uint l = 0; l < 32; l++) {
        total += quad_range_buf.Load(base + l * 4);
    }

    return total;
}

/// Sum layer counts in [l_min, l_max] and return the prefix to l_min.
///
/// Reads all 32 layer entries (sequential, cache-friendly) and
/// partitions them into the prefix before l_min and the visible range.
uint sum_layer_range(uint slot, uint dir, uint l_min, uint l_max,
                     out uint prefix_to_lmin) {
    uint base   = slot * 776 + 8 + dir * 128;
    uint total  = 0;
    uint prefix = 0;

    for (uint l = 0; l < 32; l++) {
        uint count = quad_range_buf.Load(base + l * 4);

        if (l < l_min) {
            prefix += count;
        }
        else if (l <= l_max) {
            total += count;
        }
    }

    prefix_to_lmin = prefix;
    return total;
}

/// Test whether all faces in a direction are back-facing relative to
/// the camera. For axis-aligned faces, this depends only on camera
/// position vs chunk bounds -- viewing angle is irrelevant.
///
/// +X faces span x in [origin.x+1, origin.x+32]: cull if cam.x < origin.x+1
/// -X faces span x in [origin.x,   origin.x+31]: cull if cam.x > origin.x+31
/// (analogous for Y and Z)
bool is_backfacing(uint dir, float3 cam, float3 origin) {
    if (dir == 0) return cam.x < origin.x +  1.0;  // +X
    if (dir == 1) return cam.x > origin.x + 31.0;  // -X
    if (dir == 2) return cam.y < origin.y +  1.0;  // +Y
    if (dir == 3) return cam.y > origin.y + 31.0;  // -Y
    if (dir == 4) return cam.z < origin.z +  1.0;  // +Z
                  return cam.z > origin.z + 31.0;  // -Z
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint slot = gid.x;
    if (slot >= push.total_slots) {
        return;
    }

    // Read quad count from chunk meta. Skip empty slots.
    // ChunkMeta: [quad_count, flags, _reserved, _reserved] = 16 bytes.
    uint quad_count = chunk_meta_buf.Load(slot * 16);
    if (quad_count == 0) {
        return;
    }

    // Read chunk world offset. Layout: int4 per slot = 16 bytes.
    int3 offset_raw;
    offset_raw.x = asint(chunk_offsets.Load(slot * 16 + 0));
    offset_raw.y = asint(chunk_offsets.Load(slot * 16 + 4));
    offset_raw.z = asint(chunk_offsets.Load(slot * 16 + 8));

    // AABB center: chunk offset is the min corner in voxel units.
    float3 center = float3(
        float(offset_raw.x) + 16.0,
        float(offset_raw.y) + 16.0,
        float(offset_raw.z) + 16.0
    );

    // Frustum test.
    if (!frustum_test(planes, center)) {
        return;
    }

    // Read quad range base offset.
    // QuadRange: [buffer_index(4), base_offset(4), dir_layer_counts...]
    uint base_offset = quad_range_buf.Load(slot * 776 + 4);

    // Compute per-direction quad counts and prefix sums. Quads are
    // stored direction-ordered, so prefix[d] gives the offset from
    // base_offset to the start of direction d's quads.
    float3 origin = float3(offset_raw);
    uint dir_count[6];
    uint dir_prefix[6];
    uint running         = 0;
    uint backface_total  = 0;
    uint layercull_total = 0;

    for (uint d = 0; d < 6; d++) {
        dir_prefix[d] = running;
        dir_count[d]  = sum_direction_quads(slot, d);
        running += dir_count[d];
    }

    // Accumulate chunk and quad visibility counters.
    draw_count.InterlockedAdd(4, quad_count);
    draw_count.InterlockedAdd(12, 1);

    // Emit one MDI entry per front-facing direction that has quads,
    // narrowed to the frustum-visible layer range.
    for (uint d = 0; d < 6; d++) {
        if (dir_count[d] == 0) {
            continue;
        }

        if (is_backfacing(d, camera_pos.xyz, origin)) {
            backface_total += dir_count[d];
            continue;
        }

        // Compute the visible layer range for this direction.
        uint2 lr    = compute_layer_range(planes, origin, d);
        uint  l_min = lr.x;
        uint  l_max = lr.y;

        // If the range is empty, the entire direction is layer-culled.
        if (l_min > l_max) {
            layercull_total += dir_count[d];
            continue;
        }

        // Sum only the visible layers and get the prefix offset.
        uint prefix_to_lmin;
        uint visible_count = sum_layer_range(
            slot, d, l_min, l_max, prefix_to_lmin
        );

        if (visible_count == 0) {
            continue;
        }

        // Track layer-culled quads (outside the visible range).
        layercull_total += dir_count[d] - visible_count;

        // Atomically append a visible draw.
        uint draw_index;
        draw_count.InterlockedAdd(0, 1, draw_index);

        // Write the MDI entry with the narrowed range.
        uint mdi_offset = draw_index * 16;
        dst_draws.Store(mdi_offset +  0, 4);              // vertex_count
        dst_draws.Store(mdi_offset +  4, visible_count);   // instance_count
        dst_draws.Store(mdi_offset +  8, 0);               // first_vertex
        dst_draws.Store(mdi_offset + 12,                    // first_instance
            base_offset + dir_prefix[d] + prefix_to_lmin);

        // Write per-draw metadata with the actual direction.
        uint dd_offset = draw_index * 8;
        draw_data_buf.Store(dd_offset + 0, slot);
        draw_data_buf.Store(dd_offset + 4, d);
    }

    // Accumulate culled quad counts.
    if (backface_total > 0) {
        draw_count.InterlockedAdd(8, backface_total);
    }

    if (layercull_total > 0) {
        draw_count.InterlockedAdd(16, layercull_total);
    }
}
