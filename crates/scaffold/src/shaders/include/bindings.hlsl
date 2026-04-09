// Shared buffer and resource declarations for all shader stages.
//
// Every shader includes this file to get consistent bind group layouts.
// The Rust-side bind group construction must match these declarations
// exactly.
//
// Binding layout per pipeline:
//
//   Build count:
//     set 0, binding 0 : occupancy_buf      (read-only storage)
//     set 0, binding 1 : boundary_cache_buf  (read-only storage)
//     set 0, binding 2 : chunk_meta_buf      (read-write storage)
//     set 0, binding 3 : quad_range_buf      (read-write storage)
//     push constants    : BuildPush { slot_index, base_offset }
//
//   Build alloc:
//     set 0, binding 0 : bump_state_buf          (read-write storage)
//     set 0, binding 1 : build_batch_buf          (read-only storage)
//     set 0, binding 2 : chunk_meta_buf           (read-write storage)
//     set 0, binding 3 : quad_range_buf           (read-write storage)
//     set 0, binding 4 : quad_free_list_buf       (read-write storage)
//     set 0, binding 5 : material_range_buf       (read-write storage)
//     set 0, binding 6 : material_bump_state_buf  (read-write storage)
//     set 0, binding 7 : material_free_list_buf   (read-write storage)
//     set 0, binding 8 : material_dispatch_buf    (read-write storage)
//     push constants    : AllocPush { batch_size, quad_capacity,
//                           material_capacity, material_segment_units }
//
//   Build write:
//     set 0, binding 0 : occupancy_buf      (read-only storage)
//     set 0, binding 1 : boundary_cache_buf  (read-only storage)
//     set 0, binding 2 : quad_range_buf      (read-only storage)
//     set 0, binding 3 : quad_buf            (read-write storage)
//     push constants    : BuildPush { slot_index, base_offset }
//
//   Cull:
//     set 0, binding 0 : frustum_planes      (uniform)
//     set 0, binding 1 : chunk_offsets        (read-only storage)
//     set 0, binding 2 : chunk_meta_buf       (read-only storage)
//     set 0, binding 3 : quad_range_buf       (read-only storage)
//     set 0, binding 4 : dst_draws            (read-write storage)
//     set 0, binding 5 : draw_data_buf        (read-write storage)
//     set 0, binding 6 : draw_count           (read-write storage)
//     push constants    : CullPush { total_slots }
//
//   Render (vertex + pixel):
//     set 0, binding 0 : camera               (uniform)
//     set 0, binding 1 : quad_buf             (read-only storage)
//     set 0, binding 2 : chunk_offsets         (read-only storage)
//     set 0, binding 3 : draw_data_buf         (read-only storage)
//     set 0, binding 4 : material_range_buf    (read-only storage)
//     set 0, binding 5 : (reserved)
//     set 0, binding 6 : material_table        (read-only storage)
//     set 0, binding 7 : face_textures         (read-only storage)
//     set 0, binding 8 : block_textures        (Texture2DArray)
//     set 0, binding 9 : tex_sampler           (SamplerState)
//     set 1, binding 0 : material_bufs[]       (read-only storage array)
//
//   Material pack:
//     set 0, binding 0 : material_staging      (read-only storage)
//     set 0, binding 1 : material_range_buf    (read-only storage)
//     set 1, binding 0 : material_bufs[]       (read-write storage array)

#ifndef BINDINGS_HLSL
#define BINDINGS_HLSL

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

static const uint MAX_CHUNKS       = 4096;
static const uint CHUNK_SIZE       = 32;
static const uint CHUNK_VOLUME     = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
static const uint OCC_WORDS        = 1024;  // 32 * 32
static const uint CHUNK_ALLOC_BYTES = 16;   // 4 x u32 per slot

// -----------------------------------------------------------------------
// Structures
// -----------------------------------------------------------------------

/// Per-draw metadata written by the cull shader, read by the vertex
/// shader via DrawIndex.
struct DrawData {
    uint slot;
    uint direction;
};

/// Per-chunk metadata written by the build shader.
struct ChunkMeta {
    uint quad_count;
    uint flags;
    uint sub_mask_lo;   // bits 0-31 of 64-bit sub-block visibility mask
    uint sub_mask_hi;   // bits 32-63
};

/// Per-chunk quad range metadata with precomputed prefix sums.
///
/// The count pass writes raw dir_layer_counts. The alloc pass
/// overwrites them with prefix sums:
///   dir_prefix[d]       = total quads for direction d.
///   dir_layer_pfx[d][l] = sum of layers [0, l) for direction d.
///   dir_layer_pfx[d][32] = total for direction d (same as dir_prefix[d]).
///
/// base_offset = starting quad index in the quad buffer.
/// buffer_index = which segment (Phase 4, always 0 for now).
///
/// Layout (824 bytes):
///   buffer_index   : u32           //   4 B, offset 0
///   base_offset    : u32           //   4 B, offset 4
///   dir_prefix     : [u32; 6]      //  24 B, offset 8
///   dir_layer_pfx  : [[u32; 33]; 6]// 792 B, offset 32
struct QuadRange {
    uint buffer_index;
    uint base_offset;
    uint dir_prefix[6];
    uint dir_layer_pfx[6][33];
};

/// Size in bytes of the QuadRange struct (before prefix sums are
/// computed, the count pass writes raw counts into a smaller region
/// starting at offset 32).
static const uint QUAD_RANGE_BYTES = 824;

/// Byte offset into QuadRange where the count pass writes raw
/// dir_layer_counts (6 directions x 32 layers x 4 bytes = 768 B).
/// These are overwritten by the alloc pass with prefix sums.
static const uint QUAD_RANGE_COUNTS_OFFSET = 32;

/// Camera uniform.
struct Camera {
    float4x4 view_proj;
    float4   sun_dir;     // xyz: direction to sun, w: ambient factor
    uint     flags;       // bit 0: shading, bit 1: outlines
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

/// Per-chunk material range metadata for sparse sub-block packing.
/// sub_mask is a 64-bit bitmask: bit i set means sub-block i (4x4x4
/// grid of 8x8x8 blocks, index = bz*16 + by*4 + bx) is populated.
struct MaterialRange {
    uint buffer_index;   // segment index into material_bufs[]
    uint base_offset;    // byte offset within the segment
    uint sub_mask_lo;    // bits 0-31 of sub-block visibility mask
    uint sub_mask_hi;    // bits 32-63
};

/// Size of one material sub-block in bytes (8x8x8 voxels x 2 bytes).
static const uint SUB_BLOCK_SIZE   = 8;
static const uint SUB_BLOCK_BYTES  = 1024;  // 8*8*8 * 2 (u16 per voxel)

/// GPU material entry (16 bytes, matches GpuMaterial in Rust).
struct MaterialEntry {
    uint color_rgba;     // packed RGBA (LE byte order)
    uint texture_idx;    // default texture array layer
    uint face_offset;    // 0 = uniform, nonzero = base into face_textures
    uint _pad;
};

/// Indirect draw arguments (matches wgpu DrawIndirectArgs).
struct DrawIndirectArgs {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

#endif // BINDINGS_HLSL
