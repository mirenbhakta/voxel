// Shared buffer and resource declarations for all shader stages.
//
// Every shader includes this file to get consistent bind group layouts.
// The Rust-side bind group construction must match these declarations
// exactly.
//
// Binding layout per pipeline:
//
//   Build (count + write):
//     set 0, binding 0 : occupancy_buf      (read-only storage)
//     set 0, binding 1 : boundary_cache_buf  (read-only storage)
//     set 0, binding 2 : chunk_meta_buf      (read-write storage)
//     set 0, binding 3 : quad_range_buf      (read-write storage)
//     set 0, binding 4 : quad_buf            (read-write storage, write pass only)
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
//     set 0, binding 4 : material_volume       (read-only storage)
//     set 0, binding 5 : material_table        (read-only storage)
//     set 0, binding 6 : face_textures         (read-only storage)
//     set 0, binding 7 : block_textures        (Texture2DArray)
//     set 0, binding 8 : tex_sampler           (SamplerState)

#ifndef BINDINGS_HLSL
#define BINDINGS_HLSL

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

static const uint MAX_CHUNKS       = 4096;
static const uint CHUNK_SIZE       = 32;
static const uint CHUNK_VOLUME     = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
static const uint OCC_WORDS        = 1024;  // 32 * 32

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
    uint _reserved0;
    uint _reserved1;
};

/// Per-chunk quad range metadata.
/// dir_layer_counts[d][l] = number of quads for direction d, layer l.
/// base_offset = starting quad index in the quad buffer.
/// buffer_index = which segment (Phase 4, always 0 for now).
struct QuadRange {
    uint buffer_index;
    uint base_offset;
    uint dir_layer_counts[6][32];
};

/// Camera uniform.
struct Camera {
    float4x4 view_proj;
    float4   sun_dir;     // xyz: direction to sun, w: ambient factor
    uint     flags;       // bit 0: shading, bit 1: outlines
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

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
