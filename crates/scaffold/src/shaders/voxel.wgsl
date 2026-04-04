// Camera uniform with the combined view-projection matrix.
struct Camera {
    view_proj : mat4x4<f32>,
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


// Vertex shader output.
struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0)       color         : vec3<f32>,
}

// Per-direction face colors for visual identification.
const DIR_COLORS = array<vec3<f32>, 6>(
    vec3<f32>(0.90, 0.35, 0.30), // +X  red
    vec3<f32>(0.35, 0.85, 0.35), // -X  green
    vec3<f32>(0.35, 0.40, 0.95), // +Y  blue
    vec3<f32>(0.95, 0.85, 0.30), // -Y  yellow
    vec3<f32>(0.85, 0.35, 0.85), // +Z  magenta
    vec3<f32>(0.30, 0.85, 0.85), // -Z  cyan
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

// Quad corner offsets for two-triangle quad (CCW for positive dirs).
const CORNERS_POS = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
);

// Reversed winding for negative directions (outward normal).
const CORNERS_NEG = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
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
    let vi = vertex_index % 6u;
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

    let pos = LAYER_VEC[dir] * face_layer
            + COL_VEC[dir]   * face_col
            + ROW_VEC[dir]   * face_row;

    var out : VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color         = DIR_COLORS[dir];
    return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
