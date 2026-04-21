// instance.hlsl — shared Instance struct + slot_mask field accessors.
//
// `slot_mask` packs two fields + a padding sentinel (see SubchunkInstance
// docs in Rust):
//   bits  0-21: occupancy slot index (22 bits)  — selects entry in g_occ_array
//   bits 22-25: LOD level (4 bits)              — voxel_size = 1 << level
//   bits 26-30: reserved (5 bits), must be zero
//   bit  31   : padding sentinel (consumed by the cull shader only; the
//               cull pass drops padding instances before they can reach
//               the vertex stage, so the VS never sees a padding entry)

#ifndef RENDERER_INSTANCE_HLSL
#define RENDERER_INSTANCE_HLSL

struct Instance {
    int3 origin;
    uint slot_mask;
};

static const uint INSTANCE_PADDING_BIT = 0x80000000u;

uint instance_slot(Instance i)       { return i.slot_mask & 0x3FFFFFu; }
uint instance_level(Instance i)      { return (i.slot_mask >> 22u) & 0xFu; }
bool instance_is_padding(Instance i) { return (i.slot_mask & INSTANCE_PADDING_BIT) != 0u; }

#endif // RENDERER_INSTANCE_HLSL
