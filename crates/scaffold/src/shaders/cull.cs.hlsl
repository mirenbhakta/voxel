// Frustum culling compute shader.
//
// Iterates over all chunk slots, tests visible chunks against the
// frustum, and emits compacted MDI entries + per-draw metadata.
//
// Dispatch: ceil(total_slots / 64) workgroups of 64 threads.
//
// Phase 1: one MDI entry per visible chunk (all directions combined).
// Phase 3 extends to per-(chunk, direction) entries with layer culling.

#include "include/bindings.hlsl"
#include "include/culling.hlsl"

// Bindings (cull pass).
cbuffer FrustumPlanes : register(b0, space0) {
    float4 planes[6];
};

ByteAddressBuffer   chunk_offsets  : register(t1, space0);
ByteAddressBuffer   chunk_meta_buf : register(t2, space0);
ByteAddressBuffer   quad_range_buf : register(t3, space0);
RWByteAddressBuffer dst_draws      : register(u4, space0);
RWByteAddressBuffer draw_data_buf  : register(u5, space0);
RWByteAddressBuffer draw_count     : register(u6, space0);

// Push constants.
struct CullPush {
    uint total_slots;
};

[[vk::push_constant]] CullPush push;

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
    int4 offset_raw;
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

    // Atomically append a visible draw.
    uint draw_index;
    draw_count.InterlockedAdd(0, 1, draw_index);

    // Write the MDI entry.
    // DrawIndirectArgs: [vertex_count, instance_count, first_vertex, first_instance]
    uint mdi_offset = draw_index * 16;
    dst_draws.Store(mdi_offset +  0, 4);             // vertex_count (triangle strip)
    dst_draws.Store(mdi_offset +  4, quad_count);    // instance_count
    dst_draws.Store(mdi_offset +  8, 0);             // first_vertex
    dst_draws.Store(mdi_offset + 12, base_offset);   // first_instance = quad base

    // Write per-draw metadata for the vertex shader (via DrawIndex).
    // DrawData: [slot, direction]. Direction = 0xFF for "all directions".
    uint dd_offset = draw_index * 8;
    draw_data_buf.Store(dd_offset + 0, slot);
    draw_data_buf.Store(dd_offset + 4, 0xFFFFFFFF);  // all directions (Phase 1)
}
