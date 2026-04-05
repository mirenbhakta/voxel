# Voxel

A format-agnostic voxel engine in Rust. The core philosophy, drawn from John
Lin's "The Perfect Voxel Engine," is that no single voxel representation is
optimal for every system. The engine uses whatever format best suits each job
and converts between them at runtime.

## Architecture

The library (`src/`) has zero GPU dependencies. A separate binary crate
(`crates/scaffold/`) provides a wgpu + winit render loop for visual testing and
development.

### Storage

Chunks are 32x32x32 volumes. Each chunk stores:

- **Occupancy bitmask** (4 KB) -- one bit per voxel, source of truth for
  geometry.
- **Volumetric material** (32 KB) -- one palette index per voxel, mapping
  through a per-chunk local palette to global block IDs.

Multiple storage formats (dense, RLE, palette, bitmask) are available, with
stream-based conversion between them.

### Rendering pipeline

A three-stage GPU-driven pipeline:

1. **Build** (compute, on edit) -- derives face bitmasks from chunk occupancy
   via AND-NOT, runs a material-agnostic greedy merge, and emits packed quad
   descriptors into a shared block pool.
2. **Filter** (compute, per frame) -- frustum culls chunk AABBs via stream
   compaction into a draw-indirect buffer.
3. **Render** (per frame) -- a single `multi_draw_indirect` call. The vertex
   shader unpacks quad descriptors (no vertex buffer). The fragment shader
   resolves material per-pixel from the volumetric array.

The central design insight is decoupling material from geometry. Because
materials are resolved per-pixel (not per-quad), the greedy merge is
material-agnostic and produces maximally large quads regardless of material
distribution.

### GPU memory architecture (planned)

The current pipeline splits memory management between the CPU and GPU, causing
sync stalls and allocation races. The planned redesign unifies this around
three principles:

- **Producer owns allocation.** The CPU owns source data (occupancy, material
  uploads). The GPU owns derived data (quads, page table entries, draw args).
  Neither side reaches into the other's allocation.
- **Segmented growth.** GPU pools are divided into large segments (64-256 MB)
  that can be added incrementally without reallocating or moving existing data.
  An async feedback loop (free stack level, overflow flags) tells the CPU when
  to grow.
- **Sparse volumetric material.** Instead of a flat 32 KB array per chunk,
  material is stored in 8x8x8 sub-blocks allocated from a shared pool.
  Only sub-blocks that contain visible faces are allocated. Empty chunks,
  solid interiors, and homogeneous regions cost zero pool storage. This
  reduces typical per-chunk material from 32 KB to ~4 KB and eliminates the
  128 MB pre-allocation.

All variable-size GPU data (quads and material) uses the same pattern: a block
pool with a GPU-side atomic free stack, an indirection table, and segmented
growth. The full design is in
[docs/gpu_memory_architecture.md](docs/gpu_memory_architecture.md).

### World management

`ChunkManager` bridges the CPU world and GPU representation with
distance-based streaming. Chunks are generated in parallel via rayon with a
per-frame time budget, loaded nearest-first, and unloaded with hysteresis to
prevent thrashing.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/gpu_memory_architecture.md](docs/gpu_memory_architecture.md) | Design for unified GPU memory: ownership model, segmented pools, sparse volumetric material, async feedback |
| [docs/render_pipeline_analysis.md](docs/render_pipeline_analysis.md) | Analysis of the current pipeline: patched holes, structural issues, priority ordering |
| [docs/references.md](docs/references.md) | Research references and reading list |
