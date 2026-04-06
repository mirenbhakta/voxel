// Vertex shader: unpack quad descriptor and reconstruct world position.
//
// Each instance is one quad. SV_InstanceID = first_instance + local,
// where first_instance = quad base offset. Direct indexing into the
// shared quad buffer -- no page table.
//
// DrawIndex provides the draw index for looking up per-draw metadata
// (slot, direction) from draw_data_buf.

#include "include/bindings.hlsl"
#include "include/quad.hlsl"

// Bindings (render pass, vertex shader).
cbuffer CameraCB : register(b0, space0) {
    Camera camera;
};

ByteAddressBuffer quad_buf      : register(t1, space0);
ByteAddressBuffer chunk_offsets : register(t2, space0);
ByteAddressBuffer draw_data_buf : register(t3, space0);

// Vertex output to the pixel shader.
struct VS_Output {
    float4 clip_pos    : SV_Position;
    float3 world_pos   : TEXCOORD0;
    nointerpolation uint direction : TEXCOORD1;
    nointerpolation uint slot      : TEXCOORD2;
    float2 quad_uv     : TEXCOORD3;
    nointerpolation float2 quad_size : TEXCOORD4;
};

VS_Output main(uint vertex_id   : SV_VertexID,
               uint instance_id : SV_InstanceID,
               [[vk::builtin("DrawIndex")]] uint draw_index : TEXCOORD10) {
    // Read per-draw metadata: slot and direction.
    // DrawData: [slot(4), direction(4)] = 8 bytes per draw.
    uint slot      = draw_data_buf.Load(draw_index * 8 + 0);
    uint direction = draw_data_buf.Load(draw_index * 8 + 4);

    // Read the packed quad descriptor. SV_InstanceID includes
    // first_instance (= quad base offset in Vulkan semantics).
    uint packed = quad_buf.Load(instance_id * 4);

    // Unpack quad fields.
    QuadFields q = unpack_quad(packed);

    // If direction is 0xFFFFFFFF (Phase 1: all-directions mode),
    // read it from the packed descriptor's high bits. In Phase 1 the
    // build shader still writes direction in bits 25-27 for backward
    // compatibility. In Phase 2+ this field is implicit.
    if (direction == 0xFFFFFFFF) {
        direction = (packed >> 25) & 0x7;
    }

    // Pick the quad corner for this vertex.
    float2 corner = quad_corner(vertex_id, direction);

    // Reconstruct 3D local position.
    float3 local_pos = quad_position(q, corner, direction);

    // Apply chunk world offset.
    // chunk_offsets: ivec4 per slot = 16 bytes.
    int3 offset;
    offset.x = asint(chunk_offsets.Load(slot * 16 + 0));
    offset.y = asint(chunk_offsets.Load(slot * 16 + 4));
    offset.z = asint(chunk_offsets.Load(slot * 16 + 8));

    float3 world_pos = local_pos + float3(offset);

    VS_Output o;
    o.clip_pos  = mul(camera.view_proj, float4(world_pos, 1.0));
    o.world_pos = world_pos;
    o.direction = direction;
    o.slot      = slot;
    o.quad_uv   = corner;
    o.quad_size  = float2(q.width, q.height);
    return o;
}
