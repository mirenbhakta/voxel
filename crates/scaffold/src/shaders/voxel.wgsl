// Camera uniform with the combined view-projection matrix and shading params.
struct Camera {
    view_proj : mat4x4<f32>,
    // xyz: normalized direction toward the sun, w: ambient light factor.
    sun_dir   : vec4<f32>,
    // Bit 0: shading enabled, bit 1: outline enabled.
    flags     : u32,
}

@group(0) @binding(0) var<uniform> camera : Camera;

// Packed quad descriptors.
// Bits  0- 4: col       (5 bits)
// Bits  5- 9: row       (5 bits)
// Bits 10-14: layer     (5 bits)
// Bits 15-19: width-1   (5 bits)
// Bits 20-24: height-1  (5 bits)
// Bits 25-27: direction (3 bits, 0-5)
@group(0) @binding(1) var<storage, read> quads : array<u32>;

// Page table mapping logical block indices to physical block IDs.
@group(0) @binding(2) var<storage, read> page_table : array<u32>;

// Per-slot chunk world offsets in voxel units (vec4<i32>, w unused).
// Indexed by slot = instance_index / SLOT_INSTANCE_STRIDE (98304).
@group(0) @binding(3) var<storage, read> chunk_offsets : array<vec4<i32>>;

// Volumetric material buffer. Each chunk slot stores 32768 bytes (8192 u32s).
// Each byte is a resolved BlockId. Shader reads as array<u32> and unpacks.
@group(0) @binding(4) var<storage, read> material_volume : array<u32>;

// Material property table. One vec4<u32> per block type.
// .x = packed RGBA color (LE byte order, use unpack4x8unorm).
// .y = default texture array layer index (used when .z == 0).
// .z = face texture offset (0 = uniform, nonzero = base into face_textures).
// .w = reserved.
@group(0) @binding(5) var<storage, read> material_table : array<vec4<u32>>;

// Per-face texture overrides. Only populated for blocks with non-uniform
// face textures. Six consecutive u32 entries per block, indexed by
// material_table[block_id].z + direction.
@group(0) @binding(6) var<storage, read> face_textures : array<u32>;

// Block texture array for per-face sampling.
@group(0) @binding(7) var block_textures : texture_2d_array<f32>;

// Nearest-neighbor sampler with repeat addressing.
@group(0) @binding(8) var tex_sampler : sampler;


// Vertex shader output.
struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0)       world_pos     : vec3<f32>,
    @location(1) @interpolate(flat)  direction  : u32,
    @location(2) @interpolate(flat)  slot       : u32,
    @location(3)                     quad_uv    : vec2<f32>,
    @location(4) @interpolate(flat)  quad_size  : vec2<f32>,
}

// Outward face normal per direction.
const NORMAL_VEC = array<vec3<f32>, 6>(
    vec3<f32>( 1.0,  0.0,  0.0), // +X
    vec3<f32>(-1.0,  0.0,  0.0), // -X
    vec3<f32>( 0.0,  1.0,  0.0), // +Y
    vec3<f32>( 0.0, -1.0,  0.0), // -Y
    vec3<f32>( 0.0,  0.0,  1.0), // +Z
    vec3<f32>( 0.0,  0.0, -1.0), // -Z
);

// Column axis unit vector per direction (col_axis = (normal+1)%3).
const COL_VEC = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 1.0, 0.0), // +X -> Y
    vec3<f32>(0.0, 1.0, 0.0), // -X -> Y
    vec3<f32>(0.0, 0.0, 1.0), // +Y -> Z
    vec3<f32>(0.0, 0.0, 1.0), // -Y -> Z
    vec3<f32>(1.0, 0.0, 0.0), // +Z -> X
    vec3<f32>(1.0, 0.0, 0.0), // -Z -> X
);

// Row axis unit vector per direction (row_axis = (normal+2)%3).
const ROW_VEC = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, 1.0), // +X -> Z
    vec3<f32>(0.0, 0.0, 1.0), // -X -> Z
    vec3<f32>(1.0, 0.0, 0.0), // +Y -> X
    vec3<f32>(1.0, 0.0, 0.0), // -Y -> X
    vec3<f32>(0.0, 1.0, 0.0), // +Z -> Y
    vec3<f32>(0.0, 1.0, 0.0), // -Z -> Y
);

// Layer (normal) axis unit vector per direction.
const LAYER_VEC = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0), // +X
    vec3<f32>(1.0, 0.0, 0.0), // -X
    vec3<f32>(0.0, 1.0, 0.0), // +Y
    vec3<f32>(0.0, 1.0, 0.0), // -Y
    vec3<f32>(0.0, 0.0, 1.0), // +Z
    vec3<f32>(0.0, 0.0, 1.0), // -Z
);

// Face offset along normal: 1.0 for positive dirs, 0.0 for negative.
const IS_POSITIVE = array<f32, 6>(1.0, 0.0, 1.0, 0.0, 1.0, 0.0);

// Quad corner offsets for triangle strip (CCW for positive dirs).
const CORNERS_POS = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

// Reversed winding for negative directions (outward normal).
const CORNERS_NEG = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index)   vertex_index   : u32,
    @builtin(instance_index) instance_index : u32,
) -> VertexOutput
{
    // Map absolute instance index through the page table to the physical
    // quad location in the shared pool. first_instance encodes the
    // page-table base, so instance_index / 256 is already the global
    // block index.
    let block_idx = instance_index / 256u;
    let block_off = instance_index % 256u;
    let block_id  = page_table[block_idx];
    let packed    = quads[block_id * 256u + block_off];

    // Unpack quad descriptor fields.
    let col    = f32(packed & 0x1Fu);
    let row    = f32((packed >> 5u) & 0x1Fu);
    let layer  = f32((packed >> 10u) & 0x1Fu);
    let width  = f32(((packed >> 15u) & 0x1Fu) + 1u);
    let height = f32(((packed >> 20u) & 0x1Fu) + 1u);
    let dir    = (packed >> 25u) & 0x7u;

    // Pick quad corner based on vertex index and face winding.
    let vi = vertex_index % 4u;
    var corner : vec2<f32>;

    if IS_POSITIVE[dir] > 0.5 {
        corner = CORNERS_POS[vi];
    }
    else {
        corner = CORNERS_NEG[vi];
    }

    // Reconstruct 3D position from face coordinates.
    let face_col   = col + corner.x * width;
    let face_row   = row + corner.y * height;
    let face_layer = layer + IS_POSITIVE[dir];

    let local_pos = LAYER_VEC[dir] * face_layer
                  + COL_VEC[dir]   * face_col
                  + ROW_VEC[dir]   * face_row;

    // Derive slot from instance index and apply chunk world offset.
    // 98304 = MAX_CHUNK_BLOCKS (384) * BLOCK_SIZE (256).
    let slot   = instance_index / 98304u;
    let offset = chunk_offsets[slot];
    let pos    = local_pos + vec3<f32>(f32(offset.x), f32(offset.y), f32(offset.z));

    var out : VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos     = pos;
    out.direction     = dir;
    out.slot          = slot;
    out.quad_uv       = corner;
    out.quad_size     = vec2<f32>(width, height);
    return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    // Nudge half a voxel toward the interior to identify the owning voxel.
    // The face surface sits at the boundary between voxels; without the
    // nudge, positive-direction faces would resolve to the neighbor.
    let nudge    = NORMAL_VEC[in.direction] * 0.5;
    let voxel_f  = floor(in.world_pos - nudge);

    // Wrap to chunk-local coordinates (0..31).
    let vx = u32(i32(voxel_f.x) & 31);
    let vy = u32(i32(voxel_f.y) & 31);
    let vz = u32(i32(voxel_f.z) & 31);

    // Read the block ID from the volumetric material array.
    let voxel_idx = vz * 1024u + vy * 32u + vx;
    let word_idx  = in.slot * 8192u + voxel_idx / 4u;
    let byte_off  = (voxel_idx % 4u) * 8u;
    let block_id  = (material_volume[word_idx] >> byte_off) & 0xFFu;

    // Look up material properties.
    let mat_entry   = material_table[block_id];
    let color       = unpack4x8unorm(mat_entry.x);
    let face_offset = mat_entry.z;

    // Resolve texture index. Uniform blocks use the default index
    // directly. Per-face blocks branch into the face texture table.
    var tex_idx = mat_entry.y;

    if face_offset != 0u {
        tex_idx = face_textures[face_offset + in.direction];
    }

    // Compute UV from world position based on face direction.
    var uv : vec2<f32>;

    if in.direction < 2u {
        // X faces: texture from Y and Z axes.
        uv = fract(in.world_pos.yz);
    }
    else if in.direction < 4u {
        // Y faces: texture from X and Z axes.
        uv = fract(in.world_pos.xz);
    }
    else {
        // Z faces: texture from X and Y axes.
        uv = fract(in.world_pos.xy);
    }

    // Sample the texture array.
    let tex_color = textureSample(block_textures, tex_sampler, uv, i32(tex_idx));
    var base = tex_color.rgb * color.rgb;

    // Directional shading: Lambert diffuse with ambient floor.
    let shading_on = (camera.flags & 1u) != 0u;

    if shading_on {
        let normal  = NORMAL_VEC[in.direction];
        let n_dot_l = max(dot(normal, camera.sun_dir.xyz), 0.0);
        let ambient = camera.sun_dir.w;
        let light   = ambient + (1.0 - ambient) * n_dot_l;
        base = base * light;
    }

    // Outline: darken fragments at quad edges to visualize greedy merges.
    let outline_on = (camera.flags & 2u) != 0u;

    if outline_on {
        // Distance to nearest quad edge in voxel units.
        let d = in.quad_uv * in.quad_size;
        let edge_dist = min(
            min(d.x, in.quad_size.x - d.x),
            min(d.y, in.quad_size.y - d.y),
        );

        // Fixed-width edge line regardless of quad size.
        let edge = 1.0 - smoothstep(0.02, 0.06, edge_dist);
        base = mix(base, vec3<f32>(0.0, 0.0, 0.0), edge * 0.8);
    }

    return vec4<f32>(base, 1.0);
}
