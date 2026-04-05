// Frustum culling compute shader.
//
// Reads source draw commands and chunk world offsets, tests each
// chunk's AABB against six frustum planes, and appends visible
// commands to a compacted output buffer. An atomic counter tracks
// the number of visible draws for multi_draw_indirect_count.
//
// Dispatch: ceil(total_draws / 64) workgroups of 64 threads each.

// ---------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------

/// Frustum planes (inward-pointing normals).
///
/// Each vec4 is (nx, ny, nz, d) where dot(n, point) + d >= 0 for
/// points inside the frustum. Order: left, right, bottom, top,
/// near, far.
@group(0) @binding(0)
var<uniform> planes : array<vec4<f32>, 6>;

/// Source indirect draw commands (CPU-written, all chunks with quads).
struct DrawIndirectArgs {
    vertex_count   : u32,
    instance_count : u32,
    first_vertex   : u32,
    first_instance : u32,
}

@group(0) @binding(1)
var<storage, read> src_draws : array<DrawIndirectArgs>;

/// Per-slot chunk world offsets in voxel units (w unused).
@group(0) @binding(2)
var<storage, read> chunk_offsets : array<vec4<i32>>;

/// Output compacted draw commands (visible chunks only).
@group(0) @binding(3)
var<storage, read_write> dst_draws : array<DrawIndirectArgs>;

/// Atomic visible draw count.
@group(0) @binding(4)
var<storage, read_write> draw_count : atomic<u32>;

/// Total source draw commands (push constant).
struct Immediates {
    total_draws : u32,
}

var<immediate> imm : Immediates;

// ---------------------------------------------------------------
// Constants
// ---------------------------------------------------------------

/// Instance stride per chunk slot (MAX_CHUNK_BLOCKS * BLOCK_SIZE).
const SLOT_STRIDE : u32 = 98304u;

/// Half-extent of a 32x32x32 chunk.
const HALF_EXTENT = vec3<f32>(16.0, 16.0, 16.0);

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------

@compute @workgroup_size(64, 1, 1)
fn cull_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if idx >= imm.total_draws {
        return;
    }

    let draw = src_draws[idx];

    // Derive the chunk slot from first_instance encoding.
    let slot   = draw.first_instance / SLOT_STRIDE;
    let offset = chunk_offsets[slot];

    // AABB center: chunk offset is the min corner in voxel units.
    let center = vec3<f32>(
        f32(offset.x) + 16.0,
        f32(offset.y) + 16.0,
        f32(offset.z) + 16.0,
    );

    // Test against all six frustum planes (p-vertex method).
    //
    // For each plane, the p-vertex is the AABB corner most in the
    // direction of the plane normal. If the p-vertex is behind the
    // plane, the entire AABB is outside the frustum.
    var visible = true;
    for (var i = 0u; i < 6u; i++) {
        let plane  = planes[i];
        let normal = plane.xyz;
        let d      = plane.w;

        let p_vertex = center + sign(normal) * HALF_EXTENT;

        if dot(normal, p_vertex) + d < 0.0 {
            visible = false;
            break;
        }
    }

    // Stream compaction: atomically append visible draws.
    if visible {
        let out_idx = atomicAdd(&draw_count, 1u);
        dst_draws[out_idx] = draw;
    }
}
