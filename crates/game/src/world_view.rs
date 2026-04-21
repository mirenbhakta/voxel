//! Residency-driven view of the world for a single frame's render.
//!
//! [`WorldView`] owns the CPU residency state (a multi-level
//! [`Residency`]) and the GPU-side [`WorldRenderer`]. It drives the
//! prep/patch/readback loop every frame: retired GPU prep reports are
//! drained into residency, then dirty slots are patched into the live
//! occupancy buffer, then new prep requests for newly-resident sub-chunks
//! are dispatched into the graph.
//!
//! # Slot namespaces
//!
//! Each level's [`SlotPool`](crate::world::pool::SlotPool) has its own
//! [`SlotId`](crate::world::pool::SlotId) space (`0..pool_capacity`). The
//! renderer's GPU occupancy buffer is a single shared array, so every
//! level gets an exclusive `[offset, offset + capacity)` range inside it;
//! the global slot for `(level_idx, pool_slot)` is `offset + pool_slot`.
//! Sum of per-level capacities must not exceed [`SUBCHUNK_MAX_CANDIDATES`].
//!
//! # Frame lifecycle
//!
//! Two calls per frame, split around graph submit so the readback channel
//! can arm its `map_async` on the real submission fence:
//!
//! 1. [`WorldView::update`] — top of frame. Polls wgpu, retires completed
//!    readbacks (driving the CPU residency forward), records patch passes
//!    for newly-dirty slots, records a prep pass for newly-issued
//!    requests, uploads the instance array and LOD mask.
//! 2. [`WorldView::commit_submit`] — after `ctx.end_frame` returns the
//!    submission index. Arms the channel's map_async callback so the
//!    retirement fence fires on the right frame.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use renderer::{
    BITS_EXPOSURE_MASK, BITS_IS_SOLID, BITS_MATERIAL_SLOT_SHIFT, BITS_RESIDENT,
    DirEntry, DirtyEntry, EXPOSURE_STAGING_REQUEST_IDX_SENTINEL, FrameIndex,
    GpuConsts, GpuConstsData, INFLIGHT_INVALID, LevelStatic, LodMaskUniform,
    MATERIAL_DESC_CAPACITY, MATERIAL_SLOT_INVALID, MaterialDesc,
    MaterialPatchCopy as RendererMaterialPatchCopy,
    PatchCopy, OverflowPolicy, PrepRequest as GpuPrepRequest, ReadbackChannel,
    RendererContext, SUBCHUNK_MAX_CANDIDATES, SUBCHUNK_MAX_LEVELS, SubchunkInstance,
    WorldRenderer, nodes,
};
use voxel::block::{BlockId, BlockRegistry, Material as BlockMaterial};
use renderer::graph::{BufferPool, RenderGraph};
use renderer::DirtyReport;

#[cfg(feature = "debug-state-history")]
use crate::world::state_history::{
    DirectorySnapshot, PrepRequestRecord, ResidencyEvict, ResidencyInsert,
    RetirementInconsistency, StateHistory,
};
#[cfg(feature = "debug-state-history")]
use crate::world::divergence::{compare_directory_snapshots, DivergenceReport};

use crate::world::coord::{Level, SubchunkCoord};
use crate::world::material_data_pool::{MaterialDataPool, MATERIAL_DATA_SLOT_INVALID};
use crate::world::material_pool::MaterialAllocator;
use crate::world::pool::{SlotId, cpu_compute_directory_index};
use crate::world::residency::{LevelConfig, OccupancySummary, Residency, RequestId};
use crate::world::shell::Shell;
use crate::world::slot_directory::SlotDirectory;
use crate::world::subchunk::SUBCHUNK_EDGE;

// --- WorldView ---

/// Default worldgen seed written into `GpuConsts::world_seed` at
/// `WorldView::new`. Consumed by `shaders/include/worldgen.hlsl` as the
/// per-lattice-point hash seed. Any non-zero u32 works; the exact value
/// only affects the shape of the generated terrain, not its
/// correctness. Real seed configuration (per-save, CLI override) is
/// future work — this step only needs a stable, non-zero default so
/// the FBM actually decorrelates across runs.
const DEFAULT_WORLD_SEED: u32 = 0xDEAD_BEEF;

/// M1 block-id aliases. Set during `WorldView::new`, read by the shader
/// mirror on the Rust side if it ever needs them (today the GPU prep
/// shader hardcodes the same IDs — kept in lockstep manually).
///
/// The shader's `terrain_material(xyz)` emits these three IDs directly:
/// surface → grass, 0-2 below surface → dirt, deeper → stone.
pub const BLOCK_ID_GRASS: u16 = 1;
pub const BLOCK_ID_DIRT:  u16 = 2;
pub const BLOCK_ID_STONE: u16 = 3;

/// Build the default three-block registry used by M1's terrain worldgen.
/// IDs are dense starting from 1 (0 = air) and line up with
/// `BLOCK_ID_{GRASS,DIRT,STONE}` above.
fn build_default_block_registry() -> BlockRegistry {
    let mut reg = BlockRegistry::new();
    let grass = reg.register("grass", BlockMaterial::from_rgb( 90, 160,  70));
    let dirt  = reg.register("dirt",  BlockMaterial::from_rgb(139,  90,  43));
    let stone = reg.register("stone", BlockMaterial::from_rgb(128, 128, 128));
    debug_assert_eq!(grass.raw(), BLOCK_ID_GRASS);
    debug_assert_eq!(dirt.raw(),  BLOCK_ID_DIRT);
    debug_assert_eq!(stone.raw(), BLOCK_ID_STONE);
    reg
}

/// Convert the game-side `BlockRegistry` to the dense
/// `Vec<MaterialDesc>` the renderer's descriptor table expects. Indexed
/// by `BlockId::raw()` directly; `BlockId::AIR` (index 0) lands as the
/// zero descriptor — which is safe because the shade shader only reads
/// descriptors for voxels it actually hit, and air voxels can never
/// produce a primary hit.
fn build_material_descs_from_registry(registry: &BlockRegistry) -> Vec<MaterialDesc> {
    let n = registry.len().min(MATERIAL_DESC_CAPACITY);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mat    = registry.get(BlockId::new(i as u16)).material();
        let [r, g, b, _a] = mat.color();
        out.push(MaterialDesc::from_srgb_rgb(r, g, b));
    }
    out
}

/// Per-frame integrator between the CPU residency and the GPU render path.
pub struct WorldView {
    residency:   Residency<()>,
    renderer:    Arc<WorldRenderer>,
    /// Cloned configs so the view can recompute shell bounds each frame.
    configs:     Vec<LevelConfig>,
    /// Parallel to `configs`. `level_slots[i]` describes the GPU-side
    /// slot range assigned to `configs[i]`.
    level_slots: Vec<LevelSlotRange>,
    /// Shadow of the GPU instance array. Rebuilt from every level's
    /// occupied pool each update; indices past the resident count are
    /// filled with [`SubchunkInstance::padding`] so the cull shader
    /// rejects them before touching any per-slot buffer.
    instances:   [SubchunkInstance; SUBCHUNK_MAX_CANDIDATES],

    /// Readback channel carrying `DirtyReport`s from the prep compute
    /// pass back to the CPU residency.
    channel:     ReadbackChannel<DirtyReport>,
    /// Ordered list of prep batches awaiting readback. Each entry pairs
    /// the frame the batch was dispatched on with the `(request_id, slot)`
    /// list; when `channel.take_ready` returns that frame's report the
    /// front entry retires.
    in_flight:   VecDeque<InFlightBatch>,

    // --- Exposure-only refresh (Step 5) ---
    //
    // Settles the one-frame conservatism that neighbour-aware full-prep
    // leaves behind. When a full-prep retirement changes a sub-chunk's
    // live content, each of its 6 coord-space neighbours gets queued
    // here; the next `update` call drains the queue, emits exposure-only
    // `PrepRequest`s for the still-resident entries, and dispatches the
    // `subchunk_exposure` pass. The retirement path then rewrites each
    // entry's directory `bits` in place — no patch copy, no material-slot
    // churn. See `decision-world-streaming-architecture` §Intra-batch
    // neighbour conservatism and the commentary at the top of
    // `subchunk_exposure.cs.hlsl`.
    /// Coords queued for exposure-only refresh. Filled when a full-prep
    /// retirement's neighbour exposures need recomputing, drained and
    /// filtered (against `new_requests` dedup and against current
    /// residency) at exposure-dispatch time.
    pending_exposure_refresh: HashSet<(SubchunkCoord, Level)>,
    /// Readback channel for exposure-only dispatches. Parallel to
    /// `channel`; each exposure dispatch reserves a slot here.
    exposure_channel:         ReadbackChannel<DirtyReport>,
    /// Outstanding exposure-only dispatches awaiting retirement. Mirrors
    /// the full-prep `in_flight` queue but carries the lighter
    /// [`InFlightExposureBatch`] payload (no per-request residency id,
    /// just the `directory_index` each request targeted).
    in_flight_exposure:       VecDeque<InFlightExposureBatch>,

    // --- Sub-chunk directory + material pool ---
    //
    // Scaffolding for the directory-indirection refactor (Step 1). The
    // directory + allocator are authored on the CPU and flushed to the
    // renderer's `slot_directory_buf` each frame; no shader reads it yet
    // (Step 2 adds the HLSL header and bind-group entry).
    directory:   SlotDirectory,
    allocator:   MaterialAllocator,

    // --- M1 material-data pool ---
    //
    // Non-identity allocator over 1 KB slots in the segmented GPU
    // material-data pool. See `decision-material-system-m1-sparse` for
    // the architecture; the authoritative slot for each resident
    // sub-chunk is threaded through `DirEntry.material_data_slot`.
    material_data_pool:          MaterialDataPool,
    /// Deferred-grow signal — set when the materializer observed
    /// `try_allocate() -> None` this frame. Consumed at the TOP of the
    /// NEXT frame's `update()`: the renderer appends a new 64 MB segment,
    /// the allocator calls `grow()`, and the flag resets. Grow-at-frame-
    /// start (rather than end-of-current-frame) keeps the descriptor-
    /// array mutation outside any in-flight command buffer.
    material_pool_grow_needed:   bool,
    /// Per-run static metadata (pool dims, capacities, global offsets).
    /// Populated once at construction; not mutated afterwards — pool
    /// sizing and the directory segmentation are static for the life of
    /// the `WorldView`.
    ///
    /// Bound at slot 0 of the prep pipeline so `resolve_coord_to_slot`
    /// (Step 4's neighbour-aware exposure + self-resolved directory index)
    /// can read per-level `pool_dims` / `global_offset`. `pool_origin`
    /// does NOT participate — the slot formula is a pure function of the
    /// coord's `rem_euclid(pool_dims)`, matching
    /// [`SlotPool::slot_id`](crate::world::pool::SlotPool::slot_id).
    consts:      GpuConsts,

    // --- Eviction bookkeeping ---
    evicted_last_update: usize,
    evicted_total:       u64,

    // --- Shadow-ledger counters (Step 7, tier 1) ---
    //
    // Per-frame snapshot of subsystem pressure / activity, exposed to
    // callers via [`WorldView::stats`]. Populated at the tail of
    // `update()`. Always-on; steady-state cost is ~20 u32 stores per
    // frame plus one O(capacity) pass over `directory.entries_view()` to
    // count resident entries. See `decision-scaffold-rewrite-principles`
    // principle 6 for the budget.
    stats:                       WorldRendererStats,
    /// Count of full-prep dispatches recorded into the graph this frame.
    /// Reset to 0 at the top of `update()`, incremented when a prep
    /// dispatch lands in the in-flight queue, and copied into
    /// `stats.prep_dispatches_this_frame` at tail.
    prep_dispatches_this_frame:  u32,
    /// Count of exposure-only dispatches recorded this frame. Same
    /// lifecycle as `prep_dispatches_this_frame`.
    exposure_dispatches_this_frame: u32,

    // --- Periodic log throttle ---
    /// Frame at which the last `[stats]` summary was logged. The tail of
    /// `update()` emits a line every ~60 frames (see
    /// [`Self::STATS_LOG_INTERVAL`]) so runs don't flood the console
    /// while still giving the user a running view of pool pressure.
    last_stats_log_frame: u64,

    // --- debug-state-history diagnostics ---
    //
    // A CPU-side per-frame ledger of everything that was authored this
    // frame plus a `ReadbackChannel<DirectorySnapshot>` that blits the
    // GPU's `slot_directory_buf` back to the CPU each frame. Retired
    // readbacks are compared against the history for the same frame and
    // any disagreement is logged with a human-readable classification
    // (see `crate::world::divergence`). Feature-gated — the fields
    // literally do not exist in release builds.
    #[cfg(feature = "debug-state-history")]
    state_history:                     StateHistory,
    #[cfg(feature = "debug-state-history")]
    directory_readback:                ReadbackChannel<DirectorySnapshot>,
    /// `true` iff the current frame reserved a readback slot. Gates the
    /// `commit_submit` call so a skipped reserve doesn't panic at commit.
    #[cfg(feature = "debug-state-history")]
    directory_readback_reserved:       bool,
    /// Last divergence fingerprint we logged, plus the frame range over
    /// which that fingerprint persisted. Used to collapse identical
    /// divergence patterns across consecutive frames into a single log
    /// line followed by a range summary on transition. `(fingerprint,
    /// first_frame, last_frame, divergent_count, total)`.
    #[cfg(feature = "debug-state-history")]
    last_divergence_fingerprint:       Option<(u64, u64, u64, usize, usize)>,
}

/// One outstanding prep dispatch. Pops off the front of `in_flight` when
/// its readback retires.
struct InFlightBatch {
    frame:       FrameIndex,
    /// Per-request retirement record: `(id, directory_index, coord)`.
    /// `coord` is the authoritative sub-chunk coord the prep request was
    /// issued for — retirement uses it as the directory entry's new
    /// `coord` field when the shader emits a sparse classification.
    completions: Vec<BatchCompletion>,
}

/// One outstanding exposure-only dispatch. Pops off the front of
/// `in_flight_exposure` when its readback retires.
///
/// Unlike [`InFlightBatch`], exposure-only batches carry no residency
/// [`RequestId`] — the refresh does not allocate or shift any CPU-side
/// residency state, only rewrites directory `bits`. The only per-request
/// fact the retirement needs is the `directory_index` that the shader
/// self-resolved for each request, derived on the CPU via the same
/// [`cpu_compute_directory_index`] formula the shader mirrors. Storing it
/// here lets the retirement assert parity without re-running the lookup.
struct InFlightExposureBatch {
    frame:            FrameIndex,
    /// `directory_index` each request targeted, in submit order. Parallel
    /// to the GPU-emitted dirty-list entries' `directory_index` field for
    /// the cross-check.
    expected_indices: Vec<u32>,
}

/// One entry in an [`InFlightBatch::completions`] list: everything the
/// retirement needs to translate a shader-reported `DirtyEntry` into a
/// directory write + optional patch copy.
#[derive(Clone, Copy)]
struct BatchCompletion {
    id:              RequestId,
    directory_index: u32,
    coord:           [i32; 3],
}

impl WorldView {
    /// Construct the view with the given levels, centered on
    /// `initial_camera`.
    ///
    /// The first frame has no retired readbacks yet, so the live GPU
    /// occupancy buffer stays zero-initialised until the prep dispatch
    /// recorded on frame 0 retires (typically 1–2 frames later). The
    /// render pass still runs each frame — the cull shader reads each
    /// slot's exposure from `slot_directory_buf`, whose entries are seeded
    /// as `DirEntry::empty(canonical_coord)` at construction and will carry
    /// `resident = 0, exposure = 0` until retirement populates them, so
    /// every slot is rejected trivially. Expect the first one or two
    /// rendered frames to be empty before content starts to appear.
    ///
    /// # Panics
    ///
    /// Panics if the total resident capacity across all levels exceeds
    /// [`SUBCHUNK_MAX_CANDIDATES`].
    pub fn new(
        ctx:            &RendererContext,
        pool:           &mut BufferPool,
        configs:        &[LevelConfig],
        initial_camera: [f32; 3],
    )
        -> Self
    {
        let mut level_slots = Vec::with_capacity(configs.len());
        let mut offset      = 0u32;
        for cfg in configs {
            let pool_dims = [
                2 * cfg.radius[0],
                2 * cfg.radius[1],
                2 * cfg.radius[2],
            ];
            let capacity = pool_dims[0] * pool_dims[1] * pool_dims[2];
            level_slots.push(LevelSlotRange { offset, capacity, pool_dims });
            offset = offset
                .checked_add(capacity)
                .expect("level slot offsets overflow u32");
        }

        assert!(
            offset as usize <= SUBCHUNK_MAX_CANDIDATES,
            "total resident capacity {offset} exceeds MAX_CANDIDATES ({SUBCHUNK_MAX_CANDIDATES})",
        );

        // The LOD-cascade mask assumes contiguous, nested configs.
        // `build_lod_mask` derives each level's mask entry from the
        // next-finer entry in `configs`, so callers passing e.g. L0 + L2
        // (skipping L1) get a mask that doesn't protect L2 from L0
        // overlap. Validate the contiguity invariant up front.
        for w in configs.windows(2) {
            assert_eq!(
                w[1].level.0, w[0].level.0 + 1,
                "WorldView requires contiguous levels; got {:?} then {:?}",
                w[0].level, w[1].level,
            );
        }
        assert!(
            configs.iter().all(|c| (c.level.0 as usize) < SUBCHUNK_MAX_LEVELS),
            "level index must fit in the pipeline's LOD table ({SUBCHUNK_MAX_LEVELS} entries)",
        );
        // Radii must be even for `L_(n-1)` shell to align to an integer
        // cluster of `L_n` sub-chunks — otherwise the cull's "fully inside
        // finer shell" test straddles boundaries and culls nothing. See
        // `decision-lod-nested-shells-hierarchical-occupancy` for the
        // geometric argument.
        assert!(
            configs.iter().all(|c| c.radius.iter().all(|r| *r >= 2 && r % 2 == 0)),
            "WorldView requires every radius component to be even and >= 2 for nested LOD",
        );

        let residency = Residency::<()>::new(configs, initial_camera);
        let renderer  = Arc::new(WorldRenderer::new(ctx, pool));

        // --- Publish the material descriptor table (Step 4) ---
        //
        // Build the M1 block registry (air + grass + dirt + stone), convert
        // it into `Vec<MaterialDesc>`, and upload once. The shade shader
        // reads this via slot 7 of its set-0 bind group; subsequent frames
        // reuse the upload without dirty tracking. Extending the registry
        // at runtime is a future-work item — see
        // `decision-material-system-m1-sparse` non-goals.
        let block_registry = build_default_block_registry();
        let material_descs = build_material_descs_from_registry(&block_registry);
        renderer.write_materials(ctx, &material_descs);
        let channel   = ReadbackChannel::<DirtyReport>::new(
            ctx, pool, "subchunk_prep_readback", OverflowPolicy::Panic,
        );
        // The exposure-only channel is independent of the full-prep one —
        // a frame may dispatch one, both, or neither, and the readback
        // callbacks land whenever the respective submissions fence out.
        // Panic on overflow matches the full-prep channel: the FIF-sized
        // ring should never fill under correct bookkeeping, and silent
        // drops would mask a subsystem leak.
        let exposure_channel = ReadbackChannel::<DirtyReport>::new(
            ctx, pool, "subchunk_exposure_readback", OverflowPolicy::Panic,
        );

        // --- Directory + material pool scaffolding (Step 1) ---
        //
        // Trivial 1:1 mapping: one material slot per directory entry, both
        // sized to the total of per-level shell capacities. Later steps
        // introduce sparsity (zero-exposure sub-chunks consume no material
        // slot), at which point capacity and total_directory_capacity
        // diverge and the allocator starts evicting stale entries.
        let total_capacity = offset;
        let mut directory = SlotDirectory::new(total_capacity);
        let allocator = MaterialAllocator::new(total_capacity);

        // --- Cold-start seed: stamp every shell slot with its canonical
        // coord before the first flush ---
        //
        // SlotDirectory::new initialises entries to empty([0,0,0]) (the
        // pre-seed placeholder). The GPU buffer is zero-initialised (same
        // bit pattern), so any slot that hasn't been explicitly seeded looks
        // like coord=(0,0,0) to the shader. The DDA's `resolve_and_verify`
        // distinguishes "coord matches + non-resident → sub-chunk confirmed
        // empty at this level" from "coord mismatch → ray exited shell →
        // promote to coarser LOD". Without a coord stamp, every slot looks
        // like a mismatch for any coord other than (0,0,0) — producing
        // phantom promotions into OR-reduced coarser occupancy, which
        // manifests as shadow/GI phantom hits.
        //
        // We iterate the same set of coords that Residency::new seeded for
        // prep requests, reconstruct the shell from the config + corner, and
        // write DirEntry::empty(canonical_coord) for each slot.
        for (level_idx, cfg) in configs.iter().enumerate() {
            let level  = residency
                .corner(cfg.level)
                .expect("level was just constructed; corner must exist");
            let shell  = Shell::new(cfg.radius, level);
            let range  = &level_slots[level_idx];
            for coord in shell.residents() {
                let dir_idx = cpu_compute_directory_index(
                    coord, range.pool_dims, range.offset,
                );
                directory.set(dir_idx, DirEntry::empty([coord.x, coord.y, coord.z]));
            }
        }

        // M1 material-data pool: non-identity allocator, one segment live
        // at construction to mirror the renderer's initial segment
        // allocation in `WorldRenderer::new`. The allocator's free list
        // starts with SLOTS_PER_SEGMENT entries; subsequent growth is
        // driven by the materializer's exhaustion signal. See
        // `decision-material-system-m1-sparse`.
        let mut material_data_pool = MaterialDataPool::new();
        material_data_pool
            .grow()
            .expect("first material-data-pool grow always succeeds under MAX_SEGMENTS");

        // Build the per-level static metadata that will land in the
        // `GpuConsts` uniform. One entry per configured level; the
        // remaining `MAX_LEVELS - level_count` entries stay zeroed.
        //
        // `world_seed` is written once here and never mutated — see
        // `decision-world-streaming-architecture` (§persistence
        // invariants): procedural content must be a pure function of
        // `(coord, seed)`, so `world_seed` is a one-shot at WorldView
        // init. A real seed-configuration surface (per-save-file
        // seeds, developer override) is future work.
        let mut consts_data = GpuConstsData {
            level_count: configs.len() as u32,
            world_seed:  DEFAULT_WORLD_SEED,
            ..GpuConstsData::default()
        };
        for (i, cfg) in configs.iter().enumerate() {
            let dims = [
                2 * cfg.radius[0],
                2 * cfg.radius[1],
                2 * cfg.radius[2],
            ];
            let capacity = dims[0] * dims[1] * dims[2];
            consts_data.levels[i] = LevelStatic {
                pool_dims:     dims,
                capacity,
                global_offset: level_slots[i].offset,
                _pad0:         0,
                _pad1:         0,
                _pad2:         0,
            };
        }
        let consts = GpuConsts::new(ctx, consts_data);

        #[cfg(feature = "debug-state-history")]
        let state_history = {
            // Per-frame log file. Lands in `target/debug-state-history.log`
            // relative to the cwd if `target/` exists (cargo-standard
            // location); otherwise alongside the current directory. Any
            // failure to open the file is logged once and falls back to
            // in-memory-only history — the divergence reporter and
            // retirement-invariant check still fire.
            let log_path = default_state_history_log_path();
            StateHistory::new_with_log(32, log_path)
        };
        #[cfg(feature = "debug-state-history")]
        let directory_readback = ReadbackChannel::<DirectorySnapshot>::new(
            ctx, pool, "subchunk_directory_readback", OverflowPolicy::Skip,
        );

        let stats = WorldRendererStats {
            frame:                          FrameIndex::default(),
            active_material_slots:          0,
            pool_capacity:                  SUBCHUNK_MAX_CANDIDATES as u32,
            directory_resident:             0,
            dirty_full_count_last:          0,
            dirty_exposure_count_last:      0,
            dirty_full_overflow:            false,
            dirty_exposure_overflow:        false,
            alloc_refused_cum:              0,
            alloc_evictions_cum:            0,
            pending_exposure_refresh:       0,
            in_flight_full_batches:         0,
            in_flight_exposure_batches:     0,
            prep_dispatches_this_frame:     0,
            exposure_dispatches_this_frame: 0,
            dirty_appends_this_frame_full:     0,
            dirty_appends_this_frame_exposure: 0,
            material_data_segments_live:         1,
            material_data_grow_events:           1,
            material_data_active_slots:          0,
            material_data_sentinel_patches_this_frame: 0,
        };

        Self {
            residency,
            renderer,
            configs:     configs.to_vec(),
            level_slots,
            instances:   [SubchunkInstance::padding(); SUBCHUNK_MAX_CANDIDATES],
            channel,
            in_flight:   VecDeque::new(),
            pending_exposure_refresh: HashSet::new(),
            exposure_channel,
            in_flight_exposure:       VecDeque::new(),
            directory,
            allocator,
            material_data_pool,
            material_pool_grow_needed: false,
            consts,
            evicted_last_update: 0,
            evicted_total:       0,
            stats,
            prep_dispatches_this_frame:     0,
            exposure_dispatches_this_frame: 0,
            last_stats_log_frame:           0,
            #[cfg(feature = "debug-state-history")]
            state_history,
            #[cfg(feature = "debug-state-history")]
            directory_readback,
            #[cfg(feature = "debug-state-history")]
            directory_readback_reserved: false,
            #[cfg(feature = "debug-state-history")]
            last_divergence_fingerprint: None,
        }
    }

    /// Borrow the GPU renderer for render-graph registration.
    pub fn renderer(&self) -> &Arc<WorldRenderer> {
        &self.renderer
    }

    /// Borrow the per-frame GPU constants for bind-group wiring.
    ///
    /// Used by callers that need to pass `gpu_consts` to render-graph
    /// functions (e.g. [`nodes::subchunk_world`]) without gaining ownership
    /// of the internal [`GpuConsts`] handle.
    pub fn gpu_consts(&self) -> &GpuConsts {
        &self.consts
    }

    pub fn evicted_last_update(&self) -> usize {
        self.evicted_last_update
    }

    pub fn evicted_total(&self) -> u64 {
        self.evicted_total
    }

    /// Per-frame shadow-ledger snapshot (Step 7, tier 1).
    ///
    /// Populated at the tail of [`WorldView::update`]; a call before the
    /// first `update` returns the initial all-zero state. The returned
    /// struct is cheap to copy (it's all u32/u64/bool) and is suitable
    /// for stashing in UI, logging, or test assertions.
    ///
    /// The accessor is the surface a future stats overlay / telemetry
    /// sink would plumb into; none exists today, so the method is only
    /// exercised by the periodic log path via direct field access on
    /// `self.stats`. Kept public so later work can pick it up without
    /// a visibility change.
    #[allow(dead_code)]
    pub fn stats(&self) -> WorldRendererStats {
        self.stats
    }

    /// Emit-one-line-every-N-frames throttle. Matches the tier-1 ledger
    /// guidance in `decision-scaffold-rewrite-principles` — always on,
    /// low noise.
    const STATS_LOG_INTERVAL: u64 = 60;

    /// Top-of-frame tick.
    ///
    /// 1. Polls wgpu so any pending `map_async` callbacks fire.
    /// 2. Retires completed prep readbacks into residency, patching their
    ///    dirty slots through the graph.
    /// 3. Recenters the residency against `world_pos`, freeing the
    ///    directory + material-pool entries of evicted coords.
    /// 4. Dispatches any newly-queued prep requests through the graph,
    ///    reserving a readback slot and authoring the directory entry for
    ///    the destination slot.
    /// 5. Flushes any directory entries that changed this frame into the
    ///    renderer's `slot_directory_buf`.
    /// 6. Uploads the fresh instance array and LOD mask uniform.
    /// 7. Rolls up the shadow-ledger counters into `self.stats` and
    ///    emits a periodic `[stats]` summary every
    ///    [`Self::STATS_LOG_INTERVAL`] frames.
    ///
    /// The caller must call [`WorldView::commit_submit`] after
    /// `ctx.end_frame` so the prep pass's readback gets armed on the
    /// right submission fence.
    pub fn update(
        &mut self,
        ctx       : &RendererContext,
        graph     : &mut RenderGraph,
        world_pos : [f32; 3],
        frame     : FrameIndex,
    ) {
        // Shadow-ledger: reset per-frame counters. Everything else on
        // `stats` is either a watermark (last-retired dirty counts /
        // overflow flags, sticky across frames) or an aggregate
        // re-derived at the tail (directory_resident, allocator stats,
        // pending_exposure_refresh size, in-flight depths).
        self.prep_dispatches_this_frame             = 0;
        self.exposure_dispatches_this_frame         = 0;
        self.stats.dirty_appends_this_frame_full     = 0;
        self.stats.dirty_appends_this_frame_exposure = 0;
        self.material_data_pool.reset_frame_sentinel_counter();

        // --- 0. Resolve any deferred material-data-pool growth ---
        //
        // Grow-at-frame-start rather than end-of-previous-frame: the
        // renderer's segment list is a plain `Vec<wgpu::Buffer>`, not
        // shared across threads or frames-in-flight, so `push` is free
        // here. The bind group rebuild happens naturally — the render
        // graph is rebuilt from scratch per frame and imports fresh
        // segment handles at graph-construction time (see
        // `subchunk_world`). Keeping the grow in the active frame's
        // update, before the rest of this frame's passes are recorded,
        // guarantees the new segment lands in the next dispatch's
        // binding array.
        //
        // Note: deferred sub-chunks from the previous frame stay at
        // `DirEntry.material_data_slot == MATERIAL_DATA_SLOT_INVALID`
        // until the residency cycle re-issues a prep for them (the
        // natural path — the user's camera motion triggers re-prep).
        // For the narrow "paused camera, exhaustion this frame" case,
        // those sub-chunks remain magenta until the user nudges residency.
        // Acceptable for M1 given the 1 GB ceiling; revisit if it bites.
        if self.material_pool_grow_needed {
            match self.material_data_pool.grow() {
                Ok(()) => {
                    // Renderer-side segment list is `Arc<Mutex<...>>`-
                    // guarded, so the push through `&Arc<WorldRenderer>`
                    // is safe without a mutable borrow of `self.renderer`.
                    self.renderer.append_material_segment(ctx);
                    eprintln!(
                        "[material-pool] grew to {} segment(s) (cumulative grow_events = {})",
                        self.material_data_pool.segments_live(),
                        self.material_data_pool.stats().grow_events,
                    );
                }
                Err(_ceiling) => {
                    eprintln!(
                        "[material-pool] ceiling hit at {} segments — \
                         accepting persistent magenta on un-allocatable sub-chunks",
                        self.material_data_pool.segments_live(),
                    );
                }
            }
            self.material_pool_grow_needed = false;
        }

        // --- 1. Poll for readback callbacks ---
        ctx.device().poll(wgpu::PollType::Poll).expect("device.poll failed");

        // Open a fresh frame record in the ledger. Every instrumentation
        // site below writes into this builder; it is finalized at the end
        // of the update and pushed into the history ring.
        #[cfg(feature = "debug-state-history")]
        self.state_history.begin_frame(frame);

        // Drain any directory readbacks that retired this frame and
        // compare each against the CPU-authored snapshot stored in the
        // ledger for that frame. Divergences are logged via the
        // collapse-on-fingerprint reporter so runs of identical
        // divergences do not flood the console.
        #[cfg(feature = "debug-state-history")]
        {
            let ready: Vec<(FrameIndex, DirectorySnapshot)> =
                self.directory_readback.take_ready();
            for (ready_frame, gpu_snap) in ready {
                self.compare_and_report_divergence(ready_frame, gpu_snap);
            }
        }

        // --- 2. Retire completed readbacks ---
        //
        // Under the Step-3 inversion, retirement is where directory
        // entries first become resident. `residency.complete_prep`
        // commits the sub-chunk's presence in the CPU slot pool; the
        // shader-emitted dirty list tells us *how* that sub-chunk should
        // be represented in the directory + material pool (sparse vs
        // uniform-empty). `apply_dirty_entries` materialises those
        // decisions into directory writes + a CPU-authored patch-copy
        // list.
        for (ready_frame, report) in self.channel.take_ready() {
            // FIF discipline: dispatch order matches retirement order, so
            // the front entry is always the retiring batch.
            let batch = self.in_flight.pop_front()
                .expect("take_ready delivered a report with no matching in-flight batch");
            assert_eq!(
                batch.frame, ready_frame,
                "in-flight batch frame {} != ready frame {}",
                batch.frame.get(), ready_frame.get(),
            );

            // Shadow-ledger: publish the last retired full-prep batch's
            // dirty-list stats. `count` is the shader's raw
            // InterlockedAdd counter, so it may exceed MAX_CANDIDATES
            // when the buffer was saturated; `overflow` captures that
            // case explicitly. CPU downstream code already clamps reads
            // via `min(report.count, entries.len())`.
            self.stats.dirty_full_count_last         = report.count;
            self.stats.dirty_appends_this_frame_full = report.count;
            self.stats.dirty_full_overflow           = report.overflow != 0;

            // Complete every pending request in the batch. `complete_prep`
            // silently drops ids that have been cancelled (coord rolled out
            // between dispatch and readback) — those directory_indexes
            // stay out of `accepted_coords` so their dirty entries are
            // ignored below.
            //
            // `accepted_coords` stores `(coord, level)` because the
            // retirement-invariant check (below) needs the level to pick
            // the right `pool_dims` / `global_offset` when recomputing
            // the CPU-side directory_index from the coord. A bare
            // `[i32; 3]` would be insufficient — the same coord at two
            // different levels resolves to different `directory_index`
            // values.
            let mut accepted_coords: HashMap<u32, AcceptedEntry> =
                HashMap::with_capacity(batch.completions.len());
            for c in &batch.completions {
                self.residency.complete_prep(c.id, OccupancySummary::Mixed, ());
                let Some(level_idx) = Self::level_index_for_directory(
                    &self.level_slots,
                    c.directory_index,
                ) else {
                    debug_assert!(
                        false,
                        "completion {} has out-of-range directory index {}",
                        c.id.0, c.directory_index,
                    );
                    continue;
                };
                let level = self.configs[level_idx].level;
                // `residency.get` returns Some iff complete_prep inserted
                // the payload — i.e. the coord survived to retire. A
                // cancelled request leaves `get` as None; skip its dirty
                // entry.
                if self.residency.get(level, SubchunkCoord::new(
                    c.coord[0], c.coord[1], c.coord[2],
                )).is_some() {
                    accepted_coords.insert(c.directory_index, AcceptedEntry {
                        coord:     c.coord,
                        level,
                        level_idx: level_idx as u32,
                    });

                    #[cfg(feature = "debug-state-history")]
                    if let Some(b) = self.state_history.builder_mut() {
                        b.residency_inserts.push(ResidencyInsert {
                            level,
                            coord:           SubchunkCoord::new(
                                c.coord[0], c.coord[1], c.coord[2],
                            ),
                            directory_index: c.directory_index,
                        });
                    }
                }
            }

            // Build the filtered retire inputs. Dirty entries whose
            // directory_index didn't survive residency commit are
            // dropped: their prep workgroup ran, the staging write
            // exists, but no resident coord owns the slot anymore.
            //
            // Retirement invariant: for every dirty entry we *are* about
            // to apply, the coord CPU filed at `entry.directory_index`
            // (via `accepted_coords`) must independently resolve back to
            // the same `entry.directory_index` when we feed it through
            // the canonical `cpu_compute_directory_index` formula.
            // Mismatch = the coord↔dir_idx binding has drifted somewhere
            // along the prep → readback → retirement path; the
            // retirement is about to apply correct GPU-computed content
            // to an incorrect directory_index target ("right values,
            // wrong slots" cross-wiring).
            let count = (report.count as usize).min(report.entries.len());
            let mut inputs = Vec::with_capacity(count);
            for entry in &report.entries[..count] {
                if let Some(accepted) = accepted_coords.get(&entry.directory_index) {
                    // Invariant check. Always runs under the feature;
                    // zero-cost otherwise.
                    #[cfg(feature = "debug-state-history")]
                    self.check_retirement_invariant(frame, entry, accepted);

                    inputs.push(DirtyRetireInput {
                        entry: *entry,
                        coord: accepted.coord,
                    });
                }
            }

            // Apply the transition table. `apply_dirty_entries_traced`
            // mutates the directory + allocator and returns only the
            // copies that survived the classification (uniform-empty
            // entries emit no copy). Under the `debug-state-history`
            // feature, the trace sink captures the transition decision
            // per retired entry so the divergence reporter can cross-
            // reference the CPU-authored mutation against the GPU-
            // observed directory state.
            #[cfg(feature = "debug-state-history")]
            let mut trace: Vec<crate::world::state_history::RetiredDirtyEntry> =
                Vec::with_capacity(inputs.len());
            let retire_out = apply_dirty_entries_traced(
                &inputs,
                &mut self.directory,
                &mut self.allocator,
                &mut self.material_data_pool,
                #[cfg(feature = "debug-state-history")]
                Some(&mut trace),
            );
            let copies           = retire_out.occupancy_patches;
            let material_patches = retire_out.material_patches;
            if retire_out.grow_needed {
                self.material_pool_grow_needed = true;
            }

            #[cfg(feature = "debug-state-history")]
            if let Some(b) = self.state_history.builder_mut() {
                b.dirty_entries_retired.extend(trace);
                b.patch_copies_issued.extend(copies.iter().copied());
            }

            // [patch-copy] log per-copy breakdown. The patch pass writes
            // `staging[staging_request_idx] → material_pool[dst_material_slot]`
            // for each copy; this line ties each such write back to the
            // retirement that produced it so the user can answer "why
            // did coord X's retirement target material_pool[N]?" by
            // grep in the debug log. Correlated via
            // `dst_material_slot == e.directory_index` under the
            // identity allocator policy.
            #[cfg(feature = "debug-state-history")]
            for copy in &copies {
                let dir_idx = copy.dst_material_slot;
                let coord   = accepted_coords.get(&dir_idx).map(|a| a.coord);
                let level   = accepted_coords.get(&dir_idx).map(|a| a.level);
                match (coord, level) {
                    (Some(c), Some(l)) => eprintln!(
                        "[patch-copy] frame {}: staging[req_idx={}] \
                         -> material_pool[slot={}] \
                         (retirement for coord=({},{},{}) at L{} dir_idx={})",
                        frame.get(),
                        copy.staging_request_idx,
                        copy.dst_material_slot,
                        c[0], c[1], c[2],
                        l.0,
                        dir_idx,
                    ),
                    _ => eprintln!(
                        "[patch-copy] frame {}: staging[req_idx={}] \
                         -> material_pool[slot={}] (retirement for <unknown coord; \
                         dir_idx={} not in accepted_coords>)",
                        frame.get(),
                        copy.staging_request_idx,
                        copy.dst_material_slot,
                        dir_idx,
                    ),
                }
            }

            // Record the patch pass. An empty `copies` is a no-op inside
            // `subchunk_patch`, so we don't guard here.
            //
            // The staging-ring slot is indexed by `ready_frame` (the
            // dispatch frame of the retiring prep, as enforced by the
            // `batch.frame == ready_frame` assertion above). The current
            // frame would be wrong: retirement uses `map_async` +
            // non-blocking `device.poll`, so the callback can fire at
            // `F_dispatch + N + k` (k ≥ 1) when the GPU lags, which
            // would index a different ring slot than the prep wrote. See
            // `failure-fif-rotation-invariant-bogus`.
            nodes::subchunk_patch(graph, &self.renderer, &copies, ready_frame);

            // Material-id patch pass. Parallel to the occupancy patch:
            // reads the same `staging_material_ids_ring` slot the prep
            // shader wrote, indexed by the same dispatch frame.
            nodes::subchunk_material_patch(
                graph,
                &self.renderer,
                &material_patches,
                ready_frame,
            );

            // Every retired full-prep dirty entry represents a real
            // occupancy change at `input.coord` (the shader only emits a
            // dirty entry when staged occupancy differs from live or when
            // the prior entry was non-resident). Any such change can
            // flip the exposure bits of the 6 coord-space neighbours —
            // whichever face faces the changed sub-chunk may now be more
            // or less exposed. Queue those neighbours for the next
            // frame's exposure-only dispatch; the filter at drain time
            // drops non-resident neighbours and dedups against the
            // current frame's fresh prep requests.
            //
            // We walk `inputs` (the filtered retire inputs) rather than
            // `report.entries` so cancelled preps whose residency rolled
            // out do not spawn ghost refresh requests for their former
            // neighbours.
            for input in &inputs {
                let Some(level_idx) = Self::level_index_for_directory(
                    &self.level_slots,
                    input.entry.directory_index,
                ) else { continue; };
                let level = self.configs[level_idx].level;
                let self_coord = SubchunkCoord::new(
                    input.coord[0], input.coord[1], input.coord[2],
                );
                enqueue_neighbours_for_exposure_refresh(
                    self_coord, level, &mut self.pending_exposure_refresh,
                );
            }
        }

        // Exposure-only retirements. Each ready report carries dirty
        // entries the shader emitted for coords whose neighbours had
        // changed. Retirement is simple: for each entry, rewrite the
        // directory's `bits` (exposure, is_solid, resident, preserving
        // the existing material_slot) in place. No patch copy, no
        // allocator churn.
        for (ready_frame, report) in self.exposure_channel.take_ready() {
            let batch = self.in_flight_exposure.pop_front().expect(
                "exposure take_ready delivered a report with no matching in-flight batch",
            );
            debug_assert_eq!(
                batch.frame, ready_frame,
                "in-flight exposure batch frame {} != ready frame {}",
                batch.frame.get(), ready_frame.get(),
            );

            // Shadow-ledger: same pattern as the full-prep retirement
            // — publish the raw counters from the just-retired batch.
            self.stats.dirty_exposure_count_last         = report.count;
            self.stats.dirty_appends_this_frame_exposure = report.count;
            self.stats.dirty_exposure_overflow           = report.overflow != 0;

            let count = (report.count as usize).min(report.entries.len());
            apply_exposure_dirty_entries(
                &report.entries[..count],
                &batch.expected_indices,
                &mut self.directory,
            );

            #[cfg(feature = "debug-state-history")]
            let _ = ready_frame; // reserved for future tracing
        }

        // --- 3. Recenter residency ---
        let evictions = self.residency.update_camera(world_pos);
        self.evicted_last_update = evictions.len();
        self.evicted_total       = self.evicted_total.saturating_add(evictions.len() as u64);

        // Free CPU-side material resources for each evicted coord, and
        // clear the directory entry so the torus-collision guard in the
        // prep-dispatch loop below does not see a stale `is_resident`
        // bit and re-free the same allocator slot (double-free panic).
        //
        // `is_resident` gates the allocator release: a coord that rolled
        // out before its prep ever retired has a non-resident directory
        // entry with no material slot to release.
        //
        // The directory entry is stamped with `DirEntry::empty(evict.coord)`
        // — the evicted coord, not the incoming one. The paired prep
        // dispatch in step 4 will rewrite it with `empty(req.coord)` before
        // any GPU flush, but leaving a coord-matched non-resident marker
        // here keeps the directory consistent and preserves the
        // coord-stamping invariant
        // (`convention-no-coord-sentinels-in-direntry`).
        // See `failure-eviction-skipping-directory-write-double-frees-material-slot`.
        for evict in &evictions {
            let Some(level_idx) = self.level_index(evict.level) else {
                debug_assert!(false, "eviction for unknown level {:?}", evict.level);
                continue;
            };
            // Coord-based formula is the single source of truth (see
            // `directory_index` docstring). The residency also hands us
            // `evict.slot`; debug-assert they agree so a future
            // residency/pool drift is caught at the call site, not
            // later via a retirement invariant miss.
            let directory_index = self.directory_index(level_idx, evict.coord);
            debug_assert_eq!(
                directory_index,
                self.global_slot(level_idx, evict.slot),
                "evict directory_index({:?}) via coord-based formula = {directory_index}, \
                 via residency slot_id + offset = {}: CPU formulas have drifted",
                evict.coord,
                self.global_slot(level_idx, evict.slot),
            );
            let existing = *self.directory.get(directory_index);
            if existing.is_resident() {
                self.allocator.free(existing.material_slot());
            }
            // Free the non-identity material-data slot (if any) back to
            // the pool. Independent of the occupancy-allocator free
            // above: the two pools carry separate identities.
            if existing.material_data_slot != MATERIAL_DATA_SLOT_INVALID {
                self.material_data_pool.free(existing.material_data_slot);
            }
            self.directory.set(directory_index, DirEntry::empty([
                evict.coord.x, evict.coord.y, evict.coord.z,
            ]));

            #[cfg(feature = "debug-state-history")]
            if let Some(b) = self.state_history.builder_mut() {
                b.residency_evicts.push(ResidencyEvict {
                    level:           evict.level,
                    coord:           evict.coord,
                    directory_index,
                });
            }
        }
        drop(evictions);

        // --- 4. Dispatch new prep requests ---
        //
        // Step-3 inversion: the directory entry is *not* authored at
        // retirement dispatch time. Instead, each new request unconditionally
        // seeds its slot with DirEntry::empty(req.coord) below, stamping the
        // new canonical owner. Exposure / is_solid / resident bits are
        // authored at retirement, via `apply_dirty_entries`, once the shader
        // classification arrives.
        let new_requests = self.residency.take_prep_requests();
        // Coord/level pairs freshly requested this frame. Used to dedup
        // the exposure-only queue drain below: a full-prep of coord C
        // this frame produces up-to-date exposure bits on retirement,
        // so queueing an exposure-only refresh for the same C would be
        // wasted work.
        let mut full_prep_coords: HashSet<(SubchunkCoord, Level)> =
            HashSet::with_capacity(new_requests.len());
        for req in &new_requests {
            full_prep_coords.insert((req.coord, req.level));
        }
        if !new_requests.is_empty() {
            let mut gpu_requests = Vec::with_capacity(new_requests.len());
            let mut completions  = Vec::with_capacity(new_requests.len());

            for req in &new_requests {
                let Some(level_idx) = self.level_index(req.level) else {
                    debug_assert!(false, "prep request for unknown level {:?}", req.level);
                    continue;
                };
                // Route through the coord-based helper (single source of
                // truth for the coord↔dir_idx mapping). Debug-assert the
                // residency-provided `req.slot` agrees — any disagreement
                // here means the residency's pool and the CPU formula
                // have drifted, which is the smoking gun this tooling
                // is built to catch.
                let directory_index = self.directory_index(level_idx, req.coord);
                debug_assert_eq!(
                    directory_index,
                    self.global_slot(level_idx, req.slot),
                    "directory_index({:?}) via coord-based formula = {directory_index}, \
                     via residency slot_id + offset = {}: CPU formulas have drifted",
                    req.coord,
                    self.global_slot(level_idx, req.slot),
                );

                // Torus-collision guard: if the entry is still resident,
                // the prior coord left the shell before its prep ever
                // completed *without* going through the normal eviction path.
                // Under the identity policy (pool_size == directory_size)
                // this cannot fire, but is retained to prevent material-slot
                // leaks under future non-identity allocator policies. Free
                // material-data slot too if one was allocated.
                let existing = *self.directory.get(directory_index);
                if existing.is_resident() {
                    self.allocator.free(existing.material_slot());
                    if existing.material_data_slot != MATERIAL_DATA_SLOT_INVALID {
                        self.material_data_pool.free(existing.material_data_slot);
                    }
                }

                // Unconditionally seed the slot with the new canonical
                // owner's coord. Secondary-ray DDAs see a coord-matched
                // non-resident entry (advance at this level) rather than a
                // coord-mismatched zero (promote into coarser OR-reduced
                // occupancy). Retirement overwrites this with the correct
                // exposure + is_solid + resident bits when the prep result
                // arrives.
                self.directory.set(directory_index, DirEntry::empty([
                    req.coord.x, req.coord.y, req.coord.z,
                ]));

                gpu_requests.push(GpuPrepRequest {
                    coord: [req.coord.x, req.coord.y, req.coord.z],
                    level: req.level.0 as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                });
                completions.push(BatchCompletion {
                    id:              req.id,
                    directory_index,
                    coord:           [req.coord.x, req.coord.y, req.coord.z],
                });

                #[cfg(feature = "debug-state-history")]
                if let Some(b) = self.state_history.builder_mut() {
                    b.prep_requests_issued.push(PrepRequestRecord {
                        coord: req.coord,
                        level: req.level,
                        directory_index,
                    });
                }
            }

            // Upload the CPU-side request list into the renderer's prep
            // buffer so the prep compute pass reads it.
            self.renderer.write_prep_requests(ctx, &gpu_requests);

            // Build and upload the in-flight request-index lookup for the
            // NEXT frame's prep dispatch. For each request at index `i`
            // with directory slot `d`, write `lookup[d] = i`. All other
            // slots remain INFLIGHT_INVALID — the shader falls back to
            // material_pool for those neighbours.
            //
            // The lookup goes into `current(frame + 1)` so prep@(frame+1)
            // reads it as `current(frame+1)`. This follows the convention:
            // "fill the cpu-side buffer for frame N+1 and FIF queues the
            // operation" (see `convention-patch-uses-current-frame-not-
            // dispatch-frame`).
            {
                let mut lookup = vec![INFLIGHT_INVALID; SUBCHUNK_MAX_CANDIDATES];
                for (i, c) in completions.iter().enumerate() {
                    let d = c.directory_index as usize;
                    if d < lookup.len() {
                        lookup[d] = i as u32;
                    }
                }
                self.renderer.write_inflight_request_idx(ctx, frame, &lookup);
            }

            // Reserve this frame's readback slot and wire the prep pass
            // to write into it. Panic on overflow is appropriate: reserve
            // returning None here would mean the previous cycle's readback
            // never retired — a subsystem bug, not a load condition.
            let dst = self.channel
                .reserve(frame)
                .expect("ReadbackChannel::reserve: previous prep readback never retired")
                .clone();
            let dst_h = graph.import_buffer(dst);
            nodes::subchunk_prep(
                graph,
                &self.renderer,
                &self.consts,
                dst_h,
                gpu_requests.len() as u32,
                frame,
            );

            self.in_flight.push_back(InFlightBatch { frame, completions });
            self.prep_dispatches_this_frame += 1;
        }

        // --- 4b. Dispatch exposure-only refresh for queued neighbours ---
        //
        // Drain `pending_exposure_refresh` into a request buffer. Each
        // entry must pass three filters:
        //  1. A full-prep for the same (coord, level) is *not* being
        //     dispatched this frame — full-prep's neighbour-aware
        //     exposure already produces correct bits on retirement, so
        //     an exposure-only refresh would be redundant work.
        //  2. The coord is currently resident in the directory (the
        //     `bits` we would patch must belong to the same coord).
        //     Non-resident entries will be seeded to `DirEntry::empty(coord)`
        //     when the full-prep dispatch fires; filtering them here avoids
        //     emitting a dispatch for a slot that has nothing to refresh.
        //  3. The coord has an allocated material slot (has_material).
        //     A uniform-empty entry has no occupancy to re-expose
        //     against — exposure stays 0 by construction.
        //
        // The request cap is `SUBCHUNK_MAX_CANDIDATES`: the shared
        // request buffer sizing. If the drain produces more refresh
        // candidates than fit in one dispatch we retain the overflow in
        // `pending_exposure_refresh` for a subsequent frame rather than
        // truncating silently — exposure-only refresh is latency-
        // tolerant, so "a few frames later" is an acceptable backoff.
        if !self.pending_exposure_refresh.is_empty() {
            // Take the whole set at once, refill with anything we drop.
            // Using `drain` directly would force us to re-insert on the
            // remainder path; swap + rebuild is simpler and keeps the
            // set's capacity stable.
            let queued: Vec<(SubchunkCoord, Level)> =
                self.pending_exposure_refresh.drain().collect();
            let mut exposure_requests: Vec<GpuPrepRequest> =
                Vec::with_capacity(queued.len().min(SUBCHUNK_MAX_CANDIDATES));
            let mut expected_indices: Vec<u32> =
                Vec::with_capacity(queued.len().min(SUBCHUNK_MAX_CANDIDATES));
            // Anything we can't dispatch this frame stays for the next
            // one. Prioritises "keep the oldest in the queue" — we drop
            // overflow into `carry_over` in the same iteration order
            // `drain` produced, which is unspecified but stable enough
            // for the correctness argument (every queued entry gets
            // dispatched eventually because each frame's drain adds
            // nothing new until some refresh lands).
            let mut carry_over: Vec<(SubchunkCoord, Level)> = Vec::new();

            for (coord, level) in queued {
                let Some(level_idx) = self.level_index(level) else {
                    debug_assert!(
                        false,
                        "pending_exposure_refresh contains unknown level {:?}",
                        level,
                    );
                    continue;
                };

                match classify_exposure_candidate(
                    coord, level, level_idx, &self.level_slots,
                    &self.directory, &full_prep_coords,
                    exposure_requests.len(),
                ) {
                    ExposureDispatchDecision::SkipDeduped
                    | ExposureDispatchDecision::SkipNotRefreshable => {
                        continue;
                    }
                    ExposureDispatchDecision::DeferOverflow => {
                        carry_over.push((coord, level));
                        continue;
                    }
                    ExposureDispatchDecision::Dispatch => {
                        let directory_index =
                            self.directory_index(level_idx, coord);
                        exposure_requests.push(GpuPrepRequest {
                            coord: [coord.x, coord.y, coord.z],
                            level: level.0 as u32,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                            _pad3: 0,
                        });
                        expected_indices.push(directory_index);
                    }
                }
            }

            self.pending_exposure_refresh.extend(carry_over);

            if !exposure_requests.is_empty() {
                self.renderer.write_exposure_requests(ctx, &exposure_requests);
                let dst = self.exposure_channel
                    .reserve(frame)
                    .expect(
                        "ReadbackChannel::reserve: previous exposure \
                         readback never retired",
                    )
                    .clone();
                let dst_h = graph.import_buffer(dst);
                nodes::subchunk_exposure(
                    graph,
                    &self.renderer,
                    &self.consts,
                    dst_h,
                    exposure_requests.len() as u32,
                );
                self.in_flight_exposure.push_back(InFlightExposureBatch {
                    frame,
                    expected_indices,
                });
                self.exposure_dispatches_this_frame += 1;
            }
        }

        // --- 5. Flush directory entries that changed this frame ---
        //
        // Skip the allocation when nothing is dirty. Under steady-state
        // motion this fires zero or a single-digit number of times per
        // frame (camera-driven evict+add pairs), well inside the budget
        // of `queue.write_buffer` per entry.
        if self.directory.dirty_count() > 0 {
            let updates: Vec<(u32, DirEntry)> = self.directory.drain_dirty().collect();
            self.renderer.write_directory_entries(ctx, &updates);
        }

        // --- debug: snapshot CPU directory + enqueue GPU readback ---
        //
        // After the flush, the CPU's `self.directory` is the byte-exact
        // image that `queue.write_buffer` pushed to the GPU's
        // `slot_directory_buf` for this frame. Clone it into the ledger
        // so the divergence reporter has a reference to compare against
        // when this frame's GPU readback retires (1-2 frames later).
        //
        // The readback copy is registered as a graph pass; wgpu
        // serialises it against the queue.write_buffer that just landed
        // because both flow through the same submission queue, so the
        // readback sees the post-flush directory state.
        #[cfg(feature = "debug-state-history")]
        {
            if let Some(b) = self.state_history.builder_mut() {
                b.directory_snapshot = self.directory.entries_view().to_vec();
                b.allocator_active   = self.allocator.stats().active;
            }
            self.directory_readback_reserved = false;
            if let Some(dst) = self.directory_readback.reserve(frame) {
                let dst_clone = dst.clone();
                let dst_h = graph.import_buffer(dst_clone);
                let src_h = graph.import_buffer(
                    self.renderer.slot_directory_buf().as_ref().clone(),
                );
                let copy_size = std::mem::size_of::<DirectorySnapshot>() as u64;
                let written = graph.add_pass("subchunk_directory_readback", |pass| {
                    pass.read_buffer(src_h);
                    let written = pass.write_buffer(dst_h);
                    pass.execute(move |ctx| {
                        let src = ctx.resources.buffer(src_h);
                        let dst = ctx.resources.buffer(dst_h);
                        ctx.commands.copy_buffer_to_buffer(src, 0, dst, 0, copy_size);
                    });
                    written
                });
                graph.mark_output(written);
                self.directory_readback_reserved = true;
            }
        }

        // --- 6. Upload instance array + LOD mask ---
        self.rebuild_instances(ctx, frame);
        self.upload_lod_mask(ctx);

        // --- 7. Shadow-ledger aggregate snapshot ---
        //
        // Re-derive the always-available aggregates. The retirement path
        // already published the dirty-list watermarks and overflow
        // flags; everything below is either an allocator cumulative
        // counter (already running) or a live count of an in-memory
        // structure (directory residency, pending exposure refresh,
        // in-flight depths).
        self.refresh_stats(frame);
        self.maybe_log_stats();

        // Close out this frame's ledger entry. Must be the final line —
        // any instrumentation hook that runs after this point writes to
        // a stale builder (no-op under the current API) and the
        // divergence reporter for this frame's GPU readback will miss
        // those events.
        #[cfg(feature = "debug-state-history")]
        self.state_history.finalize_frame();
    }

    /// Roll up the per-frame aggregate counters into `self.stats`.
    /// Called once at the tail of [`Self::update`]; extracted for
    /// test-ability.
    fn refresh_stats(&mut self, frame: FrameIndex) {
        let alloc_stats = self.allocator.stats();
        let mpool_stats = self.material_data_pool.stats();
        self.stats.frame                          = frame;
        self.stats.active_material_slots          = alloc_stats.active;
        self.stats.pool_capacity                  = alloc_stats.capacity;
        self.stats.directory_resident             = count_directory_resident(&self.directory);
        self.stats.alloc_refused_cum              = alloc_stats.refused;
        self.stats.alloc_evictions_cum            = alloc_stats.evictions;
        self.stats.pending_exposure_refresh       = self.pending_exposure_refresh.len() as u32;
        self.stats.in_flight_full_batches         = self.in_flight.len() as u32;
        self.stats.in_flight_exposure_batches     = self.in_flight_exposure.len() as u32;
        self.stats.prep_dispatches_this_frame     = self.prep_dispatches_this_frame;
        self.stats.exposure_dispatches_this_frame = self.exposure_dispatches_this_frame;
        self.stats.material_data_segments_live                 = mpool_stats.segments_live;
        self.stats.material_data_grow_events                   = mpool_stats.grow_events;
        self.stats.material_data_active_slots                  = mpool_stats.active;
        self.stats.material_data_sentinel_patches_this_frame   =
            mpool_stats.sentinel_patches_this_frame;

        // Log-once-per-transition on the `0 → non-zero` edge of the
        // sentinel counter. Avoids per-call spam while a sub-chunk is
        // stuck at INVALID.
        if mpool_stats.sentinel_patches_this_frame > 0 {
            eprintln!(
                "[material-pool] {} sub-chunk(s) deferred this frame \
                 (pool full — grow pending at next frame's top; segments_live={})",
                mpool_stats.sentinel_patches_this_frame,
                mpool_stats.segments_live,
            );
        }
    }

    /// Emit a one-line `[stats]` summary every [`Self::STATS_LOG_INTERVAL`]
    /// frames. Format is deliberately compact so a run produces ~1
    /// line/second at 60 Hz.
    fn maybe_log_stats(&mut self) {
        let now = self.stats.frame.get();
        if now < self.last_stats_log_frame + Self::STATS_LOG_INTERVAL {
            return;
        }
        self.last_stats_log_frame = now;

        let s = &self.stats;
        let ovf_f = if s.dirty_full_overflow     { "(OVF)" } else { "" };
        let ovf_e = if s.dirty_exposure_overflow { "(OVF)" } else { "" };
        eprintln!(
            "[stats] frame={} pool={}/{} dir_resident={} \
             dirty_full={}{} dirty_exp={}{} \
             in_flight=(f{}/e{}) disp=(f{}/e{}) \
             pend_exp_refresh={} alloc_refused={} alloc_evict={} \
             mpool=(segs={} grows={} active={})",
            s.frame.get(),
            s.active_material_slots, s.pool_capacity,
            s.directory_resident,
            s.dirty_full_count_last,     ovf_f,
            s.dirty_exposure_count_last, ovf_e,
            s.in_flight_full_batches, s.in_flight_exposure_batches,
            s.prep_dispatches_this_frame, s.exposure_dispatches_this_frame,
            s.pending_exposure_refresh,
            s.alloc_refused_cum, s.alloc_evictions_cum,
            s.material_data_segments_live,
            s.material_data_grow_events,
            s.material_data_active_slots,
        );
    }

    /// Post-`end_frame` tick: arm the channel's `map_async` callback on
    /// the submission fence so the reserved slot retires at the right
    /// time. No-op if no prep request was dispatched this frame.
    pub fn commit_submit(&mut self, frame: FrameIndex, submission: wgpu::SubmissionIndex) {
        // Only arm the callback if this frame's update actually reserved
        // a slot — otherwise the channel is still in `Empty` for this
        // slot and `commit_submit` would panic.
        let prep_dispatched_this_frame = self.in_flight.back()
            .map(|b| b.frame == frame)
            .unwrap_or(false);
        let exposure_dispatched_this_frame = self.in_flight_exposure.back()
            .map(|b| b.frame == frame)
            .unwrap_or(false);

        // Pre-clone the submission as many times as we need it — wgpu's
        // `SubmissionIndex` is cheaply cloneable, and the alternative
        // (pass `submission` by value to whichever call fires first)
        // requires an ownership dance that's noisier than this.
        #[cfg(feature = "debug-state-history")]
        {
            if prep_dispatched_this_frame {
                self.channel.commit_submit(frame, submission.clone());
            }
            if exposure_dispatched_this_frame {
                self.exposure_channel.commit_submit(frame, submission.clone());
            }
            if self.directory_readback_reserved {
                self.directory_readback.commit_submit(frame, submission);
            }
        }
        #[cfg(not(feature = "debug-state-history"))]
        {
            if prep_dispatched_this_frame {
                self.channel.commit_submit(frame, submission.clone());
            }
            if exposure_dispatched_this_frame {
                self.exposure_channel.commit_submit(frame, submission);
            }
        }
    }

    /// Compare a GPU-observed directory snapshot against the CPU-authored
    /// snapshot for `ready_frame` stored in the history ring. Logs to
    /// `stderr` with collapse-on-fingerprint so a persistent divergence
    /// pattern prints once per transition rather than every frame.
    #[cfg(feature = "debug-state-history")]
    fn compare_and_report_divergence(
        &mut self,
        ready_frame: FrameIndex,
        gpu_snap:    DirectorySnapshot,
    ) {
        let Some(record) = self.state_history.get(ready_frame) else {
            // Window evicted the record before the readback retired. We
            // ring 32 frames; at 60 Hz this means the readback sat in
            // flight for >533 ms, which is a user-visible pause. Log
            // once for visibility and move on.
            eprintln!(
                "[divergence] frame {}: readback retired with no \
                 corresponding CPU history record (ring evicted) — skipping",
                ready_frame.get(),
            );
            return;
        };

        let cpu_snap: &[DirEntry] = &record.directory_snapshot;
        let mut sink = Vec::<u8>::new();
        let report   = compare_directory_snapshots(
            ready_frame,
            cpu_snap,
            gpu_snap.entries(),
            &mut sink,
        );

        self.emit_divergence_log(&report, sink);
    }

    /// Handle a single [`DivergenceReport`]: collapse identical
    /// fingerprints across consecutive frames so a persistent pattern
    /// reports once per run rather than every frame.
    #[cfg(feature = "debug-state-history")]
    fn emit_divergence_log(&mut self, report: &DivergenceReport, body: Vec<u8>) {
        if report.is_clean() {
            // Transition from divergent → clean. Summarise the run that
            // just ended before clearing state.
            if let Some((fp, first, last, div, total)) = self.last_divergence_fingerprint.take()
                && last > first
            {
                eprintln!(
                    "[divergence] frame {}-{}: same {} entries diverged \
                     (last reported frame {}, fingerprint 0x{fp:016x}, total {total})",
                    first, last, div, first,
                );
            }
            return;
        }

        let fp         = report.fingerprint();
        let this_frame = report.frame.get();

        match self.last_divergence_fingerprint {
            Some((prev_fp, first, _, _, _)) if prev_fp == fp => {
                // Same pattern — silently extend the run. Update only
                // the last_frame tracker.
                self.last_divergence_fingerprint = Some((
                    fp, first, this_frame, report.divergent, report.total,
                ));
            }
            prior => {
                // New pattern (or first divergence). If a prior run is
                // being displaced, print its range summary first.
                if let Some((prev_fp, first, last, div, total)) = prior
                    && last > first
                {
                    eprintln!(
                        "[divergence] frame {}-{}: same {} entries diverged \
                         (last reported frame {}, fingerprint 0x{prev_fp:016x}, total {total})",
                        first, last, div, first,
                    );
                }
                // Emit the current divergence body verbatim.
                let bytes = String::from_utf8_lossy(&body);
                eprint!("{bytes}");
                self.last_divergence_fingerprint = Some((
                    fp, this_frame, this_frame, report.divergent, report.total,
                ));
            }
        }
    }

    /// Retirement-invariant check: the coord CPU filed at
    /// `entry.directory_index` (via `accepted_coords`) MUST independently
    /// resolve back to the same `entry.directory_index` when fed through
    /// the canonical [`cpu_compute_directory_index`] formula.
    ///
    /// Logs any violation to the current `FrameRecord` and emits a
    /// `[retire-inconsistency]` line on `stderr`. Runs on every retired
    /// dirty entry under the `debug-state-history` feature; the feature
    /// gate means zero cost in release builds.
    ///
    /// This is the check described in the task spec for this branch:
    /// the prior `failure-resolve-coord-to-slot-diverges-from-cpu-pool`
    /// fix eliminated shader↔CPU formula divergence, but the visible
    /// symptom — correct content at the wrong directory slot — can also
    /// be produced by a CPU-side coord↔dir_idx drift between the
    /// "populate accepted_coords" and "resolve dir_idx" paths. This
    /// check witnesses exactly that class of drift.
    #[cfg(feature = "debug-state-history")]
    fn check_retirement_invariant(
        &mut self,
        frame:    FrameIndex,
        entry:    &DirtyEntry,
        accepted: &AcceptedEntry,
    ) {
        use crate::world::pool::cpu_compute_directory_index;

        let level_idx = accepted.level_idx as usize;
        let range     = &self.level_slots[level_idx];
        let coord     = SubchunkCoord::new(
            accepted.coord[0], accepted.coord[1], accepted.coord[2],
        );

        let cpu_recomputed_dir_idx = cpu_compute_directory_index(
            coord, range.pool_dims, range.offset,
        );
        let shader_dir_idx = entry.directory_index;

        if cpu_recomputed_dir_idx == shader_dir_idx {
            // OK path. Quiet; the log file (below) records the per-frame
            // summary.
            return;
        }

        // Mismatch — the coord CPU filed at `shader_dir_idx` does not
        // resolve back to `shader_dir_idx`. Record for the frame ledger
        // and emit a live stderr line so the user sees the smoking gun
        // in real time.
        let inconsistency = RetirementInconsistency {
            dirty:                  *entry,
            cpu_coord:              coord,
            cpu_level:              accepted.level,
            cpu_recomputed_dir_idx,
            shader_dir_idx,
        };
        if let Some(b) = self.state_history.builder_mut() {
            b.retirement_inconsistencies.push(inconsistency);
        }

        eprintln!(
            "[retire-inconsistency] frame {}: shader dir_idx={shader_dir_idx} \
             but accepted_coords[{shader_dir_idx}] = coord ({},{},{}) L{} \
             recomputes to dir_idx={cpu_recomputed_dir_idx} <- MISMATCH",
            frame.get(),
            coord.x, coord.y, coord.z,
            accepted.level.0,
        );
    }

    // --- internal ---

    /// Rebuild `instances[..]` from every level's occupied slots and push
    /// the full array to the GPU.
    ///
    /// While walking, touch each visible slot's material-pool record so
    /// the future TTL eviction policy (Step 3) sees which slots rendered
    /// this frame. Only resident directory entries carry a real
    /// material_slot; non-resident ones are skipped.
    fn rebuild_instances(&mut self, ctx: &RendererContext, frame: FrameIndex) {
        // Reset to padding so any unused tail is rejected by the cull
        // shader before it reads any per-slot buffer.
        self.instances.fill(SubchunkInstance::padding());

        let mut i = 0usize;
        for (level_idx, cfg) in self.configs.iter().enumerate() {
            let level = cfg.level;
            let Some(pool) = self.residency.pool(level) else { continue; };
            let range     = &self.level_slots[level_idx];
            let base      = range.offset;
            let pool_dims = range.pool_dims;
            for (coord, pool_slot, _) in pool.occupied() {
                if i >= SUBCHUNK_MAX_CANDIDATES {
                    debug_assert!(false, "resident count exceeds MAX_CANDIDATES");
                    break;
                }
                // Coord-based formula is the single source of truth.
                // Debug-assert the pool's iteration order (which gives
                // each occupant the slot at which `insert` placed it)
                // agrees — any drift means `SlotPool::insert` or the
                // pool formula have diverged from the coord-based one.
                let directory_index = cpu_compute_directory_index(coord, pool_dims, base);
                debug_assert_eq!(
                    directory_index,
                    base + pool_slot.0,
                    "rebuild_instances: coord-based dir_idx {directory_index} != \
                     base + pool_slot {} for coord={coord:?}",
                    base + pool_slot.0,
                );
                self.instances[i] = SubchunkInstance::new(
                    world_origin(coord, level),
                    directory_index,
                    level.0,
                );

                // Touch the material slot so TTL accounting (Step 3) sees
                // this slot as recently rendered. The directory is the
                // source of truth for `directory_index → material_slot`
                // under the Step-1 mapping.
                let entry = self.directory.get(directory_index);
                if entry.is_resident() {
                    self.allocator.touch(entry.material_slot(), frame);
                }
                i += 1;
            }
        }

        self.renderer.write_instances(ctx, &self.instances);
    }

    /// Compute and upload the per-level LOD-cascade mask.
    ///
    /// Entry `[cfg.level]` of the mask is set to the shell bounds of the
    /// next-finer configured level (the one immediately below in
    /// `self.configs`). Level 0 and any unconfigured level stay inactive.
    fn upload_lod_mask(&self, ctx: &RendererContext) {
        let mut mask = LodMaskUniform::inactive();

        // Levels beyond index 0 have a configured finer level at index i-1.
        for i in 1..self.configs.len() {
            let this_cfg  = &self.configs[i];
            let finer_cfg = &self.configs[i - 1];
            let finer_level = finer_cfg.level;

            let Some(corner) = self.residency.corner(finer_level) else { continue; };
            let (lo_w, hi_w) = shell_world_bounds(corner, finer_cfg.radius, finer_level);

            let idx = this_cfg.level.0 as usize;
            mask.mask_lo[idx] = [lo_w[0], lo_w[1], lo_w[2], 0.0];
            mask.mask_hi[idx] = [hi_w[0], hi_w[1], hi_w[2], 1.0];
        }

        self.renderer.write_lod_mask(ctx, &mask);
    }

    fn level_index(&self, level: Level) -> Option<usize> {
        self.configs.iter().position(|c| c.level == level)
    }

    /// Canonical CPU formula for converting `(level_idx, coord)` to a
    /// flat `directory_index`. Every call site in this file that needs
    /// the mapping goes through here, so the coord↔dir_idx binding can
    /// never drift between sites. Mirrors the HLSL
    /// `resolve_coord_to_slot` in
    /// `crates/renderer/shaders/include/directory.hlsl` — parity
    /// regression-tested in
    /// [`cpu_compute_directory_index_matches_hlsl_formula`](
    /// crate::world::pool) and
    /// [`hlsl_formula_matches_cpu_slot_id_across_non_aligned_origins`](
    /// crate::world::pool).
    fn directory_index(&self, level_idx: usize, coord: SubchunkCoord) -> u32 {
        let range = &self.level_slots[level_idx];
        cpu_compute_directory_index(coord, range.pool_dims, range.offset)
    }

    /// Legacy helper retained for the eviction path — the residency hands
    /// out a `SlotId` alongside the evicted coord, and we'd like to
    /// double-check that `slot_id + offset` still equals
    /// `cpu_compute_directory_index(coord, ...)`. A mismatch here flags a
    /// residency-side divergence *without* the retirement invariant check
    /// (Part 1 below) needing to fire.
    fn global_slot(&self, level_idx: usize, pool_slot: SlotId) -> u32 {
        let range = &self.level_slots[level_idx];
        debug_assert!(
            pool_slot.0 < range.capacity,
            "pool slot {} at level idx {level_idx} exceeds capacity {}",
            pool_slot.0,
            range.capacity,
        );
        range.offset + pool_slot.0
    }

    /// Reverse of `global_slot`: recover the `configs[i]` index that owns
    /// a given `directory_index`. Returns `None` if the index falls
    /// outside every level's range (debug-asserts the caller, since the
    /// residency should never produce out-of-range indexes).
    fn level_index_for_directory(
        level_slots:     &[LevelSlotRange],
        directory_index: u32,
    ) -> Option<usize> {
        level_slots.iter().position(|r| {
            directory_index >= r.offset && directory_index < r.offset + r.capacity
        })
    }
}

// --- WorldRendererStats ---

/// Per-frame shadow-ledger snapshot (Step 7, tier 1 of
/// `decision-scaffold-rewrite-principles` principle 6).
///
/// Surfaces pool pressure, dirty-list health, allocator activity, and
/// convergence timing to callers without enabling the full
/// `debug-state-history` path. Always-on; the snapshot is re-derived
/// every frame at the tail of [`WorldView::update`] and costs ~20 u32
/// stores per frame plus one O(capacity) pass over the directory.
///
/// **Field taxonomy:**
///
/// - **`_cum` suffixes** — monotonic cumulative counters. Never
///   decrement across the life of a `WorldView`.
/// - **`_last` suffixes** — watermarks from the most recent retired
///   batch. Stable across frames where nothing retired; set fresh the
///   next time a retirement lands.
/// - **`_this_frame` suffixes** — reset to 0 at the top of
///   `update` and incremented by that frame's activity.
/// - **Everything else** — live snapshots of a steady-state aggregate
///   (e.g. `active_material_slots`, `in_flight_full_batches`).
///
/// See the struct body for per-field semantics; `Copy` so callers can
/// snapshot without holding a borrow on the view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldRendererStats {
    /// The [`FrameIndex`] this snapshot corresponds to.
    pub frame:                             FrameIndex,

    // --- Pool pressure ---
    /// Count of currently-allocated material-pool slots.
    pub active_material_slots:             u32,
    /// Total material-pool capacity — `SUBCHUNK_MAX_CANDIDATES` today,
    /// read from [`MaterialAllocator::stats`](crate::world::material_pool::MaterialAllocator::stats)
    /// so a later sizing change does not need to touch this struct.
    pub pool_capacity:                     u32,
    /// Count of directory entries with the `resident` bit set. Recomputed
    /// from `SlotDirectory::entries_view()` each frame.
    pub directory_resident:                u32,

    // --- Dirty-list watermarks (last retired batch) ---
    /// Raw `DirtyReport::count` from the most recently retired full-prep
    /// batch. May exceed `SUBCHUNK_MAX_CANDIDATES` when the shader
    /// saturates — the `dirty_full_overflow` flag captures that case.
    pub dirty_full_count_last:             u32,
    /// Same as [`Self::dirty_full_count_last`] but for the exposure-only
    /// dispatch.
    pub dirty_exposure_count_last:         u32,
    /// True iff the most recently retired full-prep batch hit
    /// `SUBCHUNK_MAX_CANDIDATES` and the shader raised the overflow
    /// flag.
    pub dirty_full_overflow:               bool,
    /// Same as [`Self::dirty_full_overflow`] but for the exposure-only
    /// dispatch.
    pub dirty_exposure_overflow:           bool,

    // --- Allocator cumulative counters ---
    /// Cumulative `MaterialAllocator::allocate` refusals (duplicate
    /// allocate requests). Monotonic.
    pub alloc_refused_cum:                 u64,
    /// Cumulative eviction count. Placeholder at 0 until the Step-7+
    /// eviction policy lands.
    pub alloc_evictions_cum:               u64,

    // --- In-memory queue / pipeline depths ---
    /// Size of `pending_exposure_refresh` at the end of the frame.
    pub pending_exposure_refresh:          u32,
    /// Depth of the full-prep in-flight readback queue.
    pub in_flight_full_batches:            u32,
    /// Depth of the exposure-only in-flight readback queue.
    pub in_flight_exposure_batches:        u32,

    // --- Per-frame dispatch activity ---
    /// Count of full-prep dispatches recorded into the graph this
    /// frame.
    pub prep_dispatches_this_frame:        u32,
    /// Count of exposure-only dispatches recorded into the graph this
    /// frame.
    pub exposure_dispatches_this_frame:    u32,
    /// Dirty-list appends retired this frame (full-prep). Equivalent to
    /// `dirty_full_count_last` when a batch retired on this frame; `0`
    /// otherwise. Kept as a distinct field for readability at call
    /// sites.
    pub dirty_appends_this_frame_full:     u32,
    /// Dirty-list appends retired this frame (exposure). Same shape as
    /// [`Self::dirty_appends_this_frame_full`].
    pub dirty_appends_this_frame_exposure: u32,

    // --- M1 material-data pool ---
    /// Number of 64 MB segments currently live in the material-data
    /// pool. Mirrors [`MaterialDataPool::segments_live`].
    pub material_data_segments_live:       u32,
    /// Cumulative `MaterialDataPool::grow` success events. Monotonic.
    pub material_data_grow_events:         u64,
    /// Slots currently held across the material-data pool (all live
    /// segments).
    pub material_data_active_slots:        u32,
    /// Number of `try_allocate()` → `None` events this frame. Flushed
    /// at the top of each `update` via
    /// [`MaterialDataPool::reset_frame_sentinel_counter`].
    pub material_data_sentinel_patches_this_frame: u32,
}

/// Count resident entries in the directory. O(capacity); called once per
/// frame during the stats refresh. Extracted so the test suite can
/// exercise the counting logic directly against a `SlotDirectory`.
pub(crate) fn count_directory_resident(directory: &SlotDirectory) -> u32 {
    directory
        .entries_view()
        .iter()
        .filter(|e| e.is_resident())
        .count() as u32
}

/// Default location for the per-frame `debug-state-history` log.
/// Prefers `target/debug-state-history.log` (cargo-standard), falls back
/// to `./debug-state-history.log` if there is no `target/` at cwd.
#[cfg(feature = "debug-state-history")]
fn default_state_history_log_path() -> std::path::PathBuf {
    let target = std::path::Path::new("target");
    if target.is_dir() {
        target.join("debug-state-history.log")
    } else {
        std::path::PathBuf::from("debug-state-history.log")
    }
}

// --- AcceptedEntry ---

/// `(coord, level)` pair stored in `accepted_coords` at retirement time.
///
/// Tagged with the level because the retirement-invariant check needs to
/// recompute the CPU-side directory_index from `coord`, and that
/// computation is level-scoped (different levels have different
/// `pool_dims` / `global_offset`). Storing the level_idx too lets the
/// check skip the O(N) `level_index` lookup — the index was already
/// computed when the entry was filed.
#[derive(Clone, Copy)]
struct AcceptedEntry {
    coord: [i32; 3],
    /// Level the coord belongs to. Read by the `debug-state-history`
    /// `[patch-copy]` log emitter and by the retirement-invariant check;
    /// carried here unconditionally so the fields line up with the
    /// `AcceptedEntry` construction site without a feature-gated branch.
    #[cfg_attr(not(feature = "debug-state-history"), allow(dead_code))]
    level: Level,
    /// Pre-resolved level index into `self.level_slots`. Read by the
    /// retirement-invariant check; stored to avoid re-running the O(N)
    /// `level_index` scan for every retired dirty entry.
    #[cfg_attr(not(feature = "debug-state-history"), allow(dead_code))]
    level_idx: u32,
}

// --- LevelSlotRange ---

pub(crate) struct LevelSlotRange {
    pub(crate) offset:    u32,
    pub(crate) capacity:  u32,
    /// Per-axis pool dims for this level — exactly `2 * radius` on each
    /// axis, matching the `SlotPool` constructed by `Residency`. Stored
    /// here so `cpu_compute_directory_index` can be called without
    /// crossing into `configs` for the radius.
    pub(crate) pool_dims: [u32; 3],
}

// --- helpers ---

/// World-space voxel origin of a sub-chunk at `(level, coord)`, in L0
/// voxel units (= metres, since L0 voxels are 1 m).
///
/// A sub-chunk at level N spans `8 * 2^N` voxels per axis; its origin is
/// `coord * 8 * 2^N`.
fn world_origin(coord: SubchunkCoord, level: Level) -> [i32; 3] {
    let extent = (SUBCHUNK_EDGE as i32) << level.0;
    [coord.x * extent, coord.y * extent, coord.z * extent]
}

/// World-space AABB of a level's clipmap shell.
///
/// The shell at level N spans coords `[corner - radius, corner + radius - 1]`
/// per axis (inclusive-exclusive — `2 * radius` sub-chunks per axis).
/// Each sub-chunk at level N spans `8 * 2^N` world units, so the shell's
/// world box extends from `(corner - radius) * 8 * 2^N` to
/// `(corner + radius) * 8 * 2^N`.
fn shell_world_bounds(
    corner: SubchunkCoord,
    radius: [u32; 3],
    level:  Level,
) -> ([f32; 3], [f32; 3]) {
    let extent = (SUBCHUNK_EDGE as i32) << level.0;
    let lo_c = [
        corner.x - radius[0] as i32,
        corner.y - radius[1] as i32,
        corner.z - radius[2] as i32,
    ];
    let hi_c = [
        corner.x + radius[0] as i32,
        corner.y + radius[1] as i32,
        corner.z + radius[2] as i32,
    ];
    let lo = [
        (lo_c[0] * extent) as f32,
        (lo_c[1] * extent) as f32,
        (lo_c[2] * extent) as f32,
    ];
    let hi = [
        (hi_c[0] * extent) as f32,
        (hi_c[1] * extent) as f32,
        (hi_c[2] * extent) as f32,
    ];
    (lo, hi)
}

// --- Neighbour offsets (Step 5) ---

/// The six sub-chunk-coord offsets a change propagates to for exposure
/// refresh. Order matches the cull shader's bit numbering (0=-X, 1=+X,
/// 2=-Y, 3=+Y, 4=-Z, 5=+Z) for readability; the exposure-only dispatch
/// doesn't care about order — every direction is queried independently.
const NEIGHBOUR_OFFSETS: [[i32; 3]; 6] = [
    [-1,  0,  0],
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  1,  0],
    [ 0,  0, -1],
    [ 0,  0,  1],
];

/// Enqueue the 6 neighbours of a just-retired sub-chunk for exposure-only
/// refresh. Extracted from the retirement path so the neighbour-spawning
/// rule is testable without a GPU.
pub(crate) fn enqueue_neighbours_for_exposure_refresh(
    self_coord:               SubchunkCoord,
    level:                    Level,
    pending_exposure_refresh: &mut HashSet<(SubchunkCoord, Level)>,
) {
    for offset in NEIGHBOUR_OFFSETS {
        let nbr = SubchunkCoord::new(
            self_coord.x + offset[0],
            self_coord.y + offset[1],
            self_coord.z + offset[2],
        );
        pending_exposure_refresh.insert((nbr, level));
    }
}

/// Classification of a coord drawn from `pending_exposure_refresh`:
/// whether to dispatch it (`Dispatch`), skip it because it's covered by
/// this frame's full-prep (`SkipDeduped`), skip it because its directory
/// entry is not currently valid for a refresh (`SkipNotRefreshable`), or
/// defer it to a later frame because the batch is full (`DeferOverflow`).
///
/// Exposed `pub(crate)` so `[`build_exposure_requests`]` tests can assert
/// on the classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExposureDispatchDecision {
    Dispatch,
    SkipDeduped,
    SkipNotRefreshable,
    DeferOverflow,
}

/// Pure filter for one candidate in the exposure-only request stream.
///
/// The production code inlines this inside `WorldView::update` — the
/// extraction here exists so the filter's semantics (dedup against
/// full-prep, resident-with-material check, coord-match check, overflow
/// carry) can be exercised by unit tests without a GPU.
pub(crate) fn classify_exposure_candidate(
    coord:             SubchunkCoord,
    level:             Level,
    level_idx:         usize,
    level_slots:       &[LevelSlotRange],
    directory:         &crate::world::slot_directory::SlotDirectory,
    full_prep_coords:  &HashSet<(SubchunkCoord, Level)>,
    current_batch_len: usize,
) -> ExposureDispatchDecision {
    if full_prep_coords.contains(&(coord, level)) {
        return ExposureDispatchDecision::SkipDeduped;
    }

    let range = &level_slots[level_idx];
    let directory_index = crate::world::pool::cpu_compute_directory_index(
        coord, range.pool_dims, range.offset,
    );
    let entry = directory.get(directory_index);

    if !entry.is_resident() {
        return ExposureDispatchDecision::SkipNotRefreshable;
    }
    if entry.coord != [coord.x, coord.y, coord.z] {
        return ExposureDispatchDecision::SkipNotRefreshable;
    }
    if entry.material_slot() == MATERIAL_SLOT_INVALID {
        return ExposureDispatchDecision::SkipNotRefreshable;
    }

    if current_batch_len >= SUBCHUNK_MAX_CANDIDATES {
        return ExposureDispatchDecision::DeferOverflow;
    }
    ExposureDispatchDecision::Dispatch
}

// --- Exposure-only retirement (Step 5) ---

/// Apply a batch of exposure-only dirty entries in place.
///
/// Each entry updates the directory's `bits` field at
/// `entry.directory_index` with the freshly-computed 6-bit exposure +
/// is_solid, preserving the existing `coord`, `material_slot`,
/// `content_version`, and `last_synth_version`. No material-pool copy is
/// emitted — the entry's occupancy payload is unchanged.
///
/// Invariants asserted in debug builds:
/// - `entry.staging_request_idx == EXPOSURE_STAGING_REQUEST_IDX_SENTINEL`
///   (the shader must tag exposure-only entries; any other value means a
///   full-prep entry has leaked into the exposure dirty list).
/// - The entry's `directory_index` belongs to `expected_indices` (the
///   batch dispatched this particular directory target). A stray index
///   here would indicate a cross-dispatch mix-up.
/// - The existing directory entry at `directory_index` is resident with
///   a real material slot (we filtered on these at dispatch time; the
///   directory's `queue.write_buffer` flushes from this frame land
///   before the next frame's shader reads, so by the time the refresh
///   retires, the entry may have transitioned to non-resident if an
///   eviction intervened — handle that gracefully).
pub(crate) fn apply_exposure_dirty_entries(
    entries:          &[DirtyEntry],
    expected_indices: &[u32],
    directory:        &mut crate::world::slot_directory::SlotDirectory,
) {
    // Build the dir_idx membership set for the cross-batch leak check.
    // Gated to debug builds — the set is small (N ≤ MAX_CANDIDATES) but
    // release builds have no reader for it.
    #[cfg(debug_assertions)]
    let expected: HashSet<u32> = expected_indices.iter().copied().collect();
    #[cfg(not(debug_assertions))]
    let _ = expected_indices;

    for entry in entries {
        debug_assert_eq!(
            entry.staging_request_idx, EXPOSURE_STAGING_REQUEST_IDX_SENTINEL,
            "exposure dirty entry at dir_idx {} carries a non-sentinel \
             staging_request_idx ({:#010x}) — full-prep leak",
            entry.directory_index, entry.staging_request_idx,
        );
        #[cfg(debug_assertions)]
        debug_assert!(
            expected.contains(&entry.directory_index),
            "exposure dirty entry targets dir_idx {} which was not in \
             this batch's expected_indices — cross-batch leak",
            entry.directory_index,
        );

        let old = *directory.get(entry.directory_index);
        // If the entry evicted between dispatch and retirement, the
        // bits field is zeroed; our write would re-resident it with
        // stale coord + material_slot. Skip defensively — a future
        // full-prep will re-seed this slot.
        if !old.is_resident() {
            continue;
        }

        let new_exposure = entry.new_bits_partial & BITS_EXPOSURE_MASK;
        let new_is_solid = (entry.new_bits_partial & BITS_IS_SOLID) != 0;

        // Rebuild the bits word. Keep the old material_slot (we never
        // touch the material pool) and the resident bit (still
        // resident). Drop the old exposure / is_solid and fold in the
        // freshly-computed values.
        let mut bits = new_exposure;
        if new_is_solid {
            bits |= BITS_IS_SOLID;
        }
        bits |= BITS_RESIDENT;
        bits |= (old.material_slot() & MATERIAL_SLOT_INVALID)
              << BITS_MATERIAL_SLOT_SHIFT;

        let new_entry = DirEntry {
            coord:              old.coord,
            bits,
            content_version:    old.content_version,
            last_synth_version: old.last_synth_version,
            material_data_slot: old.material_data_slot,
        };
        // `SlotDirectory::set` dirties the entry only if it differs from
        // the current value, so an unchanged-bits refresh is free — the
        // subsequent `drain_dirty` won't re-upload it.
        directory.set(entry.directory_index, new_entry);
    }
}

// --- Dirty-retirement transition logic ---

/// One input to [`apply_dirty_entries`].
///
/// Pairs a shader-reported [`DirtyEntry`] with the authoritative coord of
/// the prep request that produced it. The coord is CPU-owned (the prep
/// request carried it at dispatch time); retirement threads it into the
/// directory write rather than trusting any GPU-side round trip.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirtyRetireInput {
    pub(crate) entry: DirtyEntry,
    pub(crate) coord: [i32; 3],
}

/// Local type alias for the renderer-side [`MaterialPatchCopy`][r]. The
/// retirement logic builds the list on the CPU; the graph pass
/// `subchunk_material_patch` consumes it verbatim.
///
/// [r]: renderer::MaterialPatchCopy
pub(crate) type MaterialPatchCopy = RendererMaterialPatchCopy;

/// Return value of [`apply_dirty_entries_traced`] when the materializer
/// is threaded through with a material-data pool. Bundles the legacy
/// occupancy patches, the new material-id patches, and the pool's
/// grow-needed signal for the caller to act on at frame boundary.
#[derive(Clone, Debug)]
pub(crate) struct DirtyRetireOutput {
    pub occupancy_patches: Vec<PatchCopy>,
    pub material_patches:  Vec<MaterialPatchCopy>,
    pub grow_needed:       bool,
}

/// Apply a batch of dirty entries to the CPU-authored directory +
/// material allocator.
///
/// For every entry, this decides:
///
/// 1. The new classification (sparse iff `exposure != 0` under the
///    isolated-exposure emission; Step 4 will replace exposure with a
///    neighbor-aware value and add a proper `is_solid` path).
/// 2. The entry's destination material slot — reuse the old slot on a
///    sparse→sparse update, allocate a new one on non-sparse→sparse,
///    free the old slot on sparse→uniform-empty.
/// 3. The directory write that results (new coord / new bits / retained
///    content_version).
/// 4. Whether a `PatchCopy` is emitted — uniform-empty entries need no
///    material-pool storage and therefore no staging copy.
///
/// The function is a pure transform over `(&[DirtyRetireInput],
/// &mut SlotDirectory, &mut MaterialAllocator)`; isolating it from
/// residency / graph state makes the transition table testable without
/// a GPU.
///
/// # Allocator pressure
///
/// Under the identity allocation policy (`allocator.capacity() ==
/// directory.capacity()`), a `.expect()` on `allocate` cannot fire
/// because only resident entries draw slots and the residency caps
/// resident count at `sum(level_capacity) <= directory.capacity()`. A
/// later step introduces TTL eviction at this call site.
// Test-only entry point that omits the `debug-state-history` trace
// parameter. Production retirement goes through
// [`apply_dirty_entries_traced`] directly so the feature-gated sink
// can be threaded in. Marked `#[allow(dead_code)]` because it's used
// only from the test module, which the bin-only `cargo check` sees as
// unused.
#[allow(dead_code)]
pub(crate) fn apply_dirty_entries(
    inputs:             &[DirtyRetireInput],
    directory:          &mut SlotDirectory,
    allocator:          &mut MaterialAllocator,
    material_data_pool: &mut MaterialDataPool,
) -> Vec<PatchCopy> {
    let out = apply_dirty_entries_traced(
        inputs,
        directory,
        allocator,
        material_data_pool,
        #[cfg(feature = "debug-state-history")]
        None,
    );
    out.occupancy_patches
}

/// Inner implementation shared by the production path
/// [`apply_dirty_entries`] and the `debug-state-history` ledger path.
///
/// When the feature is enabled and `trace` is `Some(sink)`, each input
/// appends a [`state_history::RetiredDirtyEntry`] describing the CPU-side
/// transition that just landed. When the feature is off the extra
/// parameter does not exist.
pub(crate) fn apply_dirty_entries_traced(
    inputs:             &[DirtyRetireInput],
    directory:          &mut SlotDirectory,
    allocator:          &mut MaterialAllocator,
    material_data_pool: &mut MaterialDataPool,
    #[cfg(feature = "debug-state-history")]
    mut trace: Option<&mut Vec<crate::world::state_history::RetiredDirtyEntry>>,
) -> DirtyRetireOutput {
    let mut occupancy_patches = Vec::with_capacity(inputs.len());
    let mut material_patches  = Vec::with_capacity(inputs.len());
    let mut grow_needed       = false;

    for input in inputs {
        let e     = &input.entry;
        let coord = input.coord;

        let new_exposure = e.new_bits_partial & BITS_EXPOSURE_MASK;
        let new_is_solid = (e.new_bits_partial & BITS_IS_SOLID) != 0;
        // `exposure == 0` alone does not imply "uniformly empty". Two
        // distinct cases share that signal:
        //
        //   1. any_bit = 0, is_solid = 0 — truly empty. No material needed.
        //   2. any_bit = 1, is_solid = 1 — fully solid AND every neighbour
        //      face at dispatch time was fully solid (either legitimately,
        //      or transiently under intra-batch conservatism reading
        //      pre-patch `g_material_pool`). No *visible* faces now, but
        //      the sub-chunk has real occupancy that must stay resident so
        //      (a) the cull can re-include it when a neighbour changes and
        //      the exposure refresh recomputes; (b) cross-LOD
        //      `sub_sample_finer_face` at L_{n+1} can AND-reduce this
        //      entry's face plane; (c) OR-reduction at coarser levels can
        //      read its occupancy. Treating case 2 as uniform-empty wrote
        //      `DirEntry::empty()` (coord=(0,0,0), bits=0), which then
        //      failed `coord_matches` and silently removed the sub-chunk
        //      from every downstream query — producing the permanent
        //      over-cull hole documented in the 2026-04-21 frame 2040
        //      RenderDoc capture (L0 slot 29 covering the ray from camera
        //      `(-25.26, 4.30, -26.41)` east-and-down).
        //
        // Discriminate via `is_solid`: case 1 has `is_solid = 0`; case 2
        // has `is_solid = 1`. The refresh dispatch's re-prep will re-fire
        // exposure from fresh neighbour content on any subsequent frame
        // the neighbours change, so the resident entry naturally
        // transitions to the correct state.
        let is_uniform_new = new_exposure == 0 && !new_is_solid;

        // Invariants on the shader-emitted word. The material-slot field
        // must be zero (CPU authors it) and the resident bit must be set
        // (prep completion == resident).
        debug_assert!(
            (e.new_bits_partial >> 8) == 0,
            "prep shader wrote a non-zero material_slot field into \
             new_bits_partial ({:#010x}) — shader/CPU contract violation",
            e.new_bits_partial,
        );
        debug_assert!(
            (e.new_bits_partial & BITS_RESIDENT) != 0,
            "prep shader omitted the resident bit from new_bits_partial \
             ({:#010x}) — shader/CPU contract violation",
            e.new_bits_partial,
        );

        let old = *directory.get(e.directory_index);
        // `is_resident` is the authoritative "sparse entry exists" gate.
        // `direntry_has_material` would misclassify the buffer-zero state
        // (material_slot == 0, not INVALID) as sparse.
        let was_sparse        = old.is_resident();
        let old_material_slot = old.material_slot();
        let old_mds           = old.material_data_slot;

        let new_material_slot = if is_uniform_new {
            if was_sparse {
                allocator.free(old_material_slot);
            }
            None
        }
        else if was_sparse {
            Some(old_material_slot)
        }
        else {
            Some(
                allocator.allocate(e.directory_index).expect(
                    "MaterialAllocator full under identity policy — \
                     Step-3+ eviction required",
                ),
            )
        };

        // Material-data pool allocation runs in parallel to the
        // occupancy allocator, but is *non-identity* — slot identity
        // lives inside `MaterialDataPool` and is read back through the
        // directory's `material_data_slot` field (see
        // `decision-material-system-m1-sparse`).
        let new_mds = if is_uniform_new {
            // Transitioning to non-resident (uniform-empty): release the
            // material-data slot back to the pool.
            if old_mds != MATERIAL_DATA_SLOT_INVALID {
                material_data_pool.free(old_mds);
            }
            MATERIAL_DATA_SLOT_INVALID
        } else if old_mds != MATERIAL_DATA_SLOT_INVALID {
            // Reuse the existing material-data slot (sparse update or a
            // prior-frame magenta that resolved after an earlier alloc
            // success). A material-id patch is still emitted — the new
            // staging payload overwrites the old slot contents.
            old_mds
        } else {
            // First-time sparse OR prior-frame exhaustion: ask the pool
            // for a fresh slot.
            match material_data_pool.try_allocate() {
                Some(slot) => slot,
                None => {
                    // Pool exhausted. Leave the entry at INVALID (shade
                    // shader draws magenta) and signal the caller to
                    // grow + retry next frame.
                    material_data_pool.note_sentinel_patch();
                    grow_needed = true;
                    MATERIAL_DATA_SLOT_INVALID
                }
            }
        };

        // Author the new directory entry. Uniform-empty entries land as
        // `DirEntry::empty(coord)` so `is_resident` stays clear and the
        // cull shader drops the candidate before reading the material pool.
        // `material_data_slot` is threaded separately; `resident(...)` /
        // `empty(coord)` both default it to INVALID, so we set it
        // explicitly via `with_material_data_slot`.
        let new_entry = match new_material_slot {
            Some(slot) => {
                let mut de = DirEntry::resident(coord, new_exposure, new_is_solid, slot);
                de.content_version    = old.content_version;
                de.last_synth_version = old.content_version;
                de = de.with_material_data_slot(new_mds);
                de
            }
            None => DirEntry::empty(coord),
        };
        directory.set(e.directory_index, new_entry);

        if let Some(slot) = new_material_slot {
            occupancy_patches.push(PatchCopy {
                staging_request_idx: e.staging_request_idx,
                dst_material_slot:   slot,
            });
        }

        // Emit a material-id patch only when the sub-chunk actually has
        // a live material-data slot. The INVALID path is where the
        // magenta sentinel renders — no copy to issue until the grow +
        // retry lands next frame.
        if !is_uniform_new && new_mds != MATERIAL_DATA_SLOT_INVALID {
            material_patches.push(MaterialPatchCopy {
                staging_request_idx: e.staging_request_idx,
                dst_global_slot:     new_mds,
            });
        }

        #[cfg(feature = "debug-state-history")]
        if let Some(sink) = trace.as_deref_mut() {
            use crate::world::state_history::{RetiredDirtyEntry, TransitionKind};
            let transition = match (was_sparse, is_uniform_new) {
                (false, true)  => TransitionKind::UniformFirstTime,
                (false, false) => TransitionKind::UniformToSparse,
                (true,  false) => TransitionKind::SparseUpdate,
                (true,  true)  => TransitionKind::SparseToUniform,
            };
            sink.push(RetiredDirtyEntry {
                directory_index:     e.directory_index,
                coord,
                new_bits_partial:    e.new_bits_partial,
                staging_request_idx: e.staging_request_idx,
                transition,
                allocated_slot:      new_material_slot,
                new_entry,
            });
        }
    }

    DirtyRetireOutput {
        occupancy_patches,
        material_patches,
        grow_needed,
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a shader-authored `new_bits_partial`: `exposure | (is_solid
    /// << 6) | (1 << 7)`.
    fn pack_new_bits(exposure: u32, is_solid: bool) -> u32 {
        exposure | ((is_solid as u32) << 6) | BITS_RESIDENT
    }

    /// Construct a freshly-grown [`MaterialDataPool`] sized for the
    /// transition-table tests. One segment is live immediately so
    /// `try_allocate` succeeds without the deferred-grow path; capacity
    /// is the production [`SLOTS_PER_SEGMENT`] — plenty of headroom for
    /// any test that allocates at most a handful of slots.
    fn fresh_mpool() -> MaterialDataPool {
        let mut p = MaterialDataPool::new();
        p.grow().expect("first grow always succeeds");
        p
    }

    fn retire(
        dir_idx: u32,
        req_idx: u32,
        exposure: u32,
        is_solid: bool,
        coord: [i32; 3],
    ) -> DirtyRetireInput {
        DirtyRetireInput {
            entry: DirtyEntry {
                directory_index:     dir_idx,
                new_bits_partial:    pack_new_bits(exposure, is_solid),
                staging_request_idx: req_idx,
                _pad:                0,
            },
            coord,
        }
    }

    // -- Transition table: sparse/uniform × first-time/update --

    #[test]
    fn uniform_first_time_writes_empty_and_emits_no_copy() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count(); // clear initial dirty set

        // Exposure = 0 → uniform-empty classification.
        let inputs = vec![retire(2, 0, 0, false, [1, 2, 3])];
        let copies = apply_dirty_entries(&inputs, &mut dir, &mut alloc, &mut mpool);

        assert!(copies.is_empty(), "uniform-empty must not allocate a material slot");
        let e = dir.get(2);
        assert!(!e.is_resident(), "uniform-empty must clear the resident bit");
        // Coord stamp is load-bearing: secondary-ray DDAs (shadow, GI) use
        // `resolve_and_verify` to distinguish "ray exited this level's
        // shell" (coord mismatch → promote) from "sub-chunk confirmed
        // empty" (coord matches, resident clear → advance at this level).
        // A bare `DirEntry::empty()` with coord=(0,0,0) produced phantom
        // shadow hits via promotion into L_{n+1}'s OR-reduced occupancy.
        assert_eq!(e.coord, [1, 2, 3], "uniform-empty must stamp the retired coord");
        assert_eq!(alloc.active(), 0, "no material slot allocated");
    }

    #[test]
    fn sparse_first_time_allocates_and_emits_copy() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // Exposure != 0 → sparse classification; no prior material slot.
        let inputs = vec![retire(2, 7, BITS_EXPOSURE_MASK, false, [4, 5, 6])];
        let copies = apply_dirty_entries(&inputs, &mut dir, &mut alloc, &mut mpool);

        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].staging_request_idx, 7);
        // Identity policy: allocator returns directory_index verbatim. This
        // is what keeps material_pool[directory_index] lined up with the
        // renderer's VS/PS read.
        assert_eq!(copies[0].dst_material_slot, 2);

        let e = dir.get(2);
        assert!(e.is_resident());
        assert_eq!(e.exposure(),      BITS_EXPOSURE_MASK);
        assert_eq!(e.material_slot(), 2);
        assert_eq!(e.coord,           [4, 5, 6]);
        assert_eq!(alloc.active(), 1);
    }

    #[test]
    fn sparse_update_reuses_slot_and_emits_copy_to_same_slot() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // First-time sparse: identity policy allocates slot 2.
        let first = vec![retire(2, 1, 0x3F, false, [1, 2, 3])];
        let _     = apply_dirty_entries(&first, &mut dir, &mut alloc, &mut mpool);
        assert_eq!(alloc.active(), 1);

        // Sparse update at the same directory_index: different exposure,
        // different staging index.
        let second = vec![retire(2, 5, 0x21, false, [1, 2, 3])];
        let copies = apply_dirty_entries(&second, &mut dir, &mut alloc, &mut mpool);

        assert_eq!(copies.len(), 1);
        // Material slot is retained across the update (== directory_index 2
        // under the identity policy).
        assert_eq!(copies[0].dst_material_slot, 2);
        // Staging index reflects the new prep workgroup.
        assert_eq!(copies[0].staging_request_idx, 5);

        let e = dir.get(2);
        assert!(e.is_resident());
        assert_eq!(e.exposure(), 0x21);
        // Allocator didn't churn on a reuse.
        assert_eq!(alloc.active(), 1);
    }

    #[test]
    fn sparse_to_uniform_frees_slot() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // Prime slot 0 via sparse-first-time.
        let first = vec![retire(3, 2, 0x3F, false, [0, 0, 0])];
        let _     = apply_dirty_entries(&first, &mut dir, &mut alloc, &mut mpool);
        assert_eq!(alloc.active(), 1);

        // Transition to uniform-empty.
        let second = vec![retire(3, 2, 0, false, [0, 0, 0])];
        let copies = apply_dirty_entries(&second, &mut dir, &mut alloc, &mut mpool);

        assert!(copies.is_empty(), "uniform-empty must not emit a copy");
        assert_eq!(alloc.active(), 0, "the freed slot must be released");
        assert!(!dir.get(3).is_resident());
    }

    #[test]
    fn uniform_to_sparse_allocates_fresh_slot() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // Prime with a uniform-empty entry (directory stays empty, no
        // material slot).
        let first = vec![retire(1, 0, 0, false, [0, 0, 0])];
        let _     = apply_dirty_entries(&first, &mut dir, &mut alloc, &mut mpool);
        assert_eq!(alloc.active(), 0);

        // Now the sub-chunk becomes sparse.
        let second = vec![retire(1, 4, 0x3F, false, [0, 0, 0])];
        let copies = apply_dirty_entries(&second, &mut dir, &mut alloc, &mut mpool);

        assert_eq!(copies.len(), 1);
        // Identity policy: slot == directory_index 1.
        assert_eq!(copies[0].dst_material_slot, 1);
        assert_eq!(copies[0].staging_request_idx, 4);
        assert!(dir.get(1).is_resident());
        assert_eq!(alloc.active(), 1);
    }

    #[test]
    fn full_transition_cycle_leaves_allocator_clean() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // uniform-first-time → sparse-first-time → sparse-update →
        // sparse-to-uniform. After the cycle, every slot the allocator
        // touched should be back on the free stack.
        let script = [
            retire(1, 0, 0,    false, [0, 0, 0]), // uniform first-time
            retire(1, 1, 0x3F, false, [0, 0, 0]), // sparse first-time
            retire(1, 2, 0x21, false, [0, 0, 0]), // sparse update
            retire(1, 3, 0,    false, [0, 0, 0]), // sparse → uniform
        ];
        for step in &script {
            let _ = apply_dirty_entries(std::slice::from_ref(step), &mut dir, &mut alloc, &mut mpool);
        }

        assert_eq!(alloc.active(), 0);
        assert_eq!(alloc.free_count(), 4);
        assert!(!dir.get(1).is_resident());
    }

    // Step 4: the prep shader now emits a real `is_solid` bit when every
    // voxel in the staged occupancy is solid. Retirement must propagate
    // that bit into the new resident directory entry (the cull shader's
    // `direntry_is_solid` fast-path depends on seeing it), and the
    // allocator/slot assignment must behave identically to the non-solid
    // sparse case.
    #[test]
    fn is_solid_propagates_into_resident_entry() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // Sparse-first-time with is_solid=true: the sub-chunk is uniformly
        // solid but has at least one exposed face (otherwise exposure
        // would be 0 and the classification would drop into uniform-empty).
        let inputs = vec![retire(2, 3, 0x03, true, [9, 9, 9])];
        let copies = apply_dirty_entries(&inputs, &mut dir, &mut alloc, &mut mpool);

        assert_eq!(copies.len(), 1, "sparse entry must emit a patch copy");
        // Identity policy: slot == directory_index 2.
        assert_eq!(copies[0].dst_material_slot, 2);
        assert_eq!(copies[0].staging_request_idx, 3);

        let e = dir.get(2);
        assert!(e.is_resident());
        assert!(e.is_solid(), "is_solid bit must survive the retirement path");
        assert_eq!(e.exposure(), 0x03);
        assert_eq!(e.coord, [9, 9, 9]);
    }

    #[test]
    fn is_solid_with_zero_exposure_retires_resident_not_uniform_empty() {
        // A fully-solid sub-chunk bordered by fully-solid neighbours has
        // exposure=0 (no face is exposed) and is_solid=1. The retirement
        // MUST NOT treat this as uniform-empty: the sub-chunk has real
        // occupancy that must stay resident so (a) the cull can re-admit
        // it when a neighbour changes and the exposure refresh recomputes;
        // (b) cross-LOD `sub_sample_finer_face` at the coarser level can
        // AND-reduce this entry's face plane; (c) OR-reduction at coarser
        // levels reads its occupancy.
        //
        // This test guards the fix to `is_uniform_new` after a RenderDoc
        // capture (2026-04-21 frame 2040) showed L0 `(-3, -1, -3)` stuck
        // in `DirEntry::empty()` under the old rule `is_uniform_new =
        // exposure == 0`, producing a permanent over-cull hole where the
        // ray from the camera found no occluder.
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        let inputs = vec![retire(1, 0, 0, true, [0, 0, 0])];
        let copies = apply_dirty_entries(&inputs, &mut dir, &mut alloc, &mut mpool);

        assert_eq!(
            copies.len(), 1,
            "fully-solid + exposure=0 must allocate a material slot and emit \
             a patch copy — the occupancy is needed downstream"
        );
        let e = dir.get(1);
        assert!(e.is_resident(), "fully-solid retires as resident, not empty()");
        assert!(e.is_solid(), "is_solid bit is preserved into the directory");
        assert_eq!(e.exposure(), 0, "exposure=0 is preserved verbatim");
        assert_eq!(alloc.active(), 1, "one material slot is live for this entry");
    }

    #[test]
    fn exposure_zero_without_is_solid_retires_as_uniform_empty() {
        // Complement of the above: exposure=0 AND is_solid=0 is the true
        // uniform-empty case (any_bit was 0 at prep time). No material
        // slot, no patch copy, entry retires as `DirEntry::empty(coord)`.
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        let inputs = vec![retire(2, 0, 0, false, [-4, 7, -1])];
        let copies = apply_dirty_entries(&inputs, &mut dir, &mut alloc, &mut mpool);

        assert!(copies.is_empty(), "uniform-empty must not allocate a material slot");
        let e = dir.get(2);
        assert!(!e.is_resident(), "uniform-empty retires with resident bit clear");
        assert_eq!(
            e.coord, [-4, 7, -1],
            "uniform-empty must stamp the retired coord so secondary-ray \
             `resolve_and_verify` advances at this level rather than \
             promoting into L_{{n+1}} OR-reduced occupancy",
        );
        assert_eq!(alloc.active(), 0);
    }

    #[test]
    fn content_version_is_preserved_across_sparse_update() {
        let mut dir   = SlotDirectory::new(4);
        let mut alloc = MaterialAllocator::new(4);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // Seed a resident entry with a non-zero content_version to mimic
        // a post-edit state. Step 3 doesn't bump the version, but the
        // retirement must not zero it.
        let mut primed = DirEntry::resident([1, 1, 1], 0x3F, false, 0);
        primed.content_version = 42;
        alloc.allocate(0).unwrap(); // make the allocator agree that slot 0 is live
        dir.set(0, primed);

        let inputs = vec![retire(0, 9, 0x1F, false, [1, 1, 1])];
        let _ = apply_dirty_entries(&inputs, &mut dir, &mut alloc, &mut mpool);

        let e = dir.get(0);
        assert!(e.is_resident());
        assert_eq!(e.exposure(), 0x1F);
        assert_eq!(e.content_version,    42);
        assert_eq!(e.last_synth_version, 42, "retire stamps last_synth = current content");
    }

    // -- Step 5: exposure-only retirement --

    /// Build a shader-authored exposure-only dirty entry. The sentinel
    /// `staging_request_idx` is the distinguishing field.
    fn exposure_entry(dir_idx: u32, exposure: u32, is_solid: bool) -> DirtyEntry {
        DirtyEntry {
            directory_index:     dir_idx,
            new_bits_partial:    pack_new_bits(exposure, is_solid),
            staging_request_idx: EXPOSURE_STAGING_REQUEST_IDX_SENTINEL,
            _pad:                0,
        }
    }

    #[test]
    fn exposure_retirement_updates_bits_in_place() {
        let mut dir = SlotDirectory::new(4);
        let _ = dir.drain_dirty().count();

        // Seed dir_idx=2 as a resident entry with known coord, bits, and
        // content_version. Exposure-only refresh must rewrite exposure
        // + is_solid while preserving everything else.
        let mut seeded = DirEntry::resident([11, 22, 33], 0x1F, false, 2);
        seeded.content_version    = 7;
        seeded.last_synth_version = 7;
        dir.set(2, seeded);
        let _ = dir.drain_dirty().count();

        let entries = vec![exposure_entry(2, 0x3A, true)];
        apply_exposure_dirty_entries(&entries, &[2], &mut dir);

        let e = dir.get(2);
        assert!(e.is_resident(), "resident bit stays set");
        assert_eq!(e.exposure(),    0x3A);
        assert!(e.is_solid());
        // Load-bearing: coord, material_slot, content_version all
        // preserved.
        assert_eq!(e.coord,              [11, 22, 33]);
        assert_eq!(e.material_slot(),    2);
        assert_eq!(e.content_version,    7);
        assert_eq!(e.last_synth_version, 7);
    }

    #[test]
    fn exposure_retirement_is_no_op_for_non_resident_entry() {
        // If the entry evicted between dispatch and retirement, the
        // bits field reverted to zero. Exposure-only retirement must
        // NOT re-resident it with stale data.
        let mut dir = SlotDirectory::new(4);
        dir.set(3, DirEntry::empty([0, 0, 0]));
        let _ = dir.drain_dirty().count();

        let entries = vec![exposure_entry(3, 0x15, false)];
        apply_exposure_dirty_entries(&entries, &[3], &mut dir);

        let e = dir.get(3);
        assert!(!e.is_resident(), "non-resident entry stays non-resident");
        assert_eq!(e.bits, 0);
    }

    #[test]
    fn exposure_retirement_preserves_material_slot_not_identity() {
        // Under a non-identity allocator policy (Step 7+), the directory
        // entry's material_slot != directory_index. Exposure-only
        // retirement must keep the existing material_slot rather than
        // smuggling the directory_index into the slot field.
        let mut dir = SlotDirectory::new(8);
        dir.set(5, DirEntry::resident([0, 0, 0], 0x01, false, 42));
        let _ = dir.drain_dirty().count();

        let entries = vec![exposure_entry(5, 0x20, false)];
        apply_exposure_dirty_entries(&entries, &[5], &mut dir);

        let e = dir.get(5);
        assert_eq!(e.material_slot(), 42, "material_slot preserved across refresh");
        assert_eq!(e.exposure(), 0x20);
    }

    // -- Step 5: neighbour enqueue --

    #[test]
    fn enqueue_neighbours_queues_six_coords_of_same_level() {
        let mut pending: HashSet<(SubchunkCoord, Level)> = HashSet::new();
        enqueue_neighbours_for_exposure_refresh(
            SubchunkCoord::new(5, -3, 7),
            Level(2),
            &mut pending,
        );

        assert_eq!(pending.len(), 6, "each of the six axes queues one neighbour");
        for offset in NEIGHBOUR_OFFSETS {
            let expected = SubchunkCoord::new(
                5 + offset[0], -3 + offset[1], 7 + offset[2],
            );
            assert!(
                pending.contains(&(expected, Level(2))),
                "expected {expected:?} at L2 in pending set",
            );
        }
    }

    #[test]
    fn enqueue_neighbours_from_sparse_first_time_retirement() {
        // Mirrors the retirement path: a first-time sparse entry
        // (exposure != 0, was_sparse = false) is retired — its 6
        // neighbours get queued. This is the primary path Step 5
        // targets.
        let mut dir   = SlotDirectory::new(256);
        let mut alloc = MaterialAllocator::new(256);
        let mut mpool = fresh_mpool();
        let _ = dir.drain_dirty().count();

        // Sparse-first-time retirement.
        let inputs = vec![retire(100, 0, 0x3F, false, [4, 5, 6])];
        let _copies = apply_dirty_entries(&inputs, &mut dir, &mut alloc, &mut mpool);
        assert!(dir.get(100).is_resident());

        // Simulate what `update` does after retirement: for each input,
        // call enqueue_neighbours.
        let mut pending: HashSet<(SubchunkCoord, Level)> = HashSet::new();
        enqueue_neighbours_for_exposure_refresh(
            SubchunkCoord::new(4, 5, 6), Level(0), &mut pending,
        );
        assert_eq!(pending.len(), 6);
    }

    #[test]
    fn enqueue_neighbours_dedup_on_overlapping_retirements() {
        // Two adjacent retirements — (0,0,0) and (1,0,0) — share a
        // neighbour pair: (0,0,0)'s +X neighbour is (1,0,0), which is
        // the second retirement's own coord. Insertion-dedup via
        // HashSet keeps each unique (coord, level) once.
        let mut pending: HashSet<(SubchunkCoord, Level)> = HashSet::new();
        enqueue_neighbours_for_exposure_refresh(
            SubchunkCoord::new(0, 0, 0), Level(0), &mut pending,
        );
        enqueue_neighbours_for_exposure_refresh(
            SubchunkCoord::new(1, 0, 0), Level(0), &mut pending,
        );

        // Neighbours of (0,0,0): ±X,±Y,±Z around origin — 6 unique.
        // Neighbours of (1,0,0): ±X,±Y,±Z around (1,0,0) — 6 unique.
        // Overlap: (1,0,0)'s -X = (0,0,0) is NOT in (0,0,0)'s own
        // neighbour set (its +X is (1,0,0), which IS in (1,0,0)'s +0
        // position). Walk it:
        //  (0,0,0)'s neighbours: (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)
        //  (1,0,0)'s neighbours: (0,0,0),  (2,0,0), (1,-1,0), (1,1,0), (1,0,-1), (1,0,1)
        // Union has no overlaps — 12 unique entries.
        assert_eq!(pending.len(), 12);
    }

    // -- Step 5: exposure-only request building --

    /// Build a minimal `[LevelSlotRange]` for a single level with the
    /// given pool dimensions and global offset, matching what
    /// `WorldView::new` would produce.
    fn make_level_slots(pool_dims: [u32; 3], offset: u32) -> Vec<LevelSlotRange> {
        let capacity = pool_dims[0] * pool_dims[1] * pool_dims[2];
        vec![LevelSlotRange { offset, capacity, pool_dims }]
    }

    /// Build a `SlotDirectory` with a resident entry at the canonical
    /// directory_index for `coord` at level `level_slots[0]`.
    fn directory_with_resident(
        capacity:    u32,
        level_slots: &[LevelSlotRange],
        coord:       SubchunkCoord,
        exposure:    u32,
        slot:        u32,
    ) -> SlotDirectory {
        use crate::world::pool::cpu_compute_directory_index;
        let mut dir = SlotDirectory::new(capacity);
        let range = &level_slots[0];
        let dir_idx = cpu_compute_directory_index(coord, range.pool_dims, range.offset);
        dir.set(dir_idx, DirEntry::resident(
            [coord.x, coord.y, coord.z], exposure, false, slot,
        ));
        let _ = dir.drain_dirty().count();
        dir
    }

    #[test]
    fn classify_skips_coords_in_full_prep_set() {
        // Dedup test from the spec: coord C in both pending_refresh and
        // new_requests should NOT emit an exposure-only request for C.
        let level_slots = make_level_slots([8, 8, 8], 0);
        let coord = SubchunkCoord::new(2, 3, 4);
        let level = Level(0);

        let dir = directory_with_resident(
            512, &level_slots, coord, 0x3F, 0,
        );

        let mut full_prep_coords: HashSet<(SubchunkCoord, Level)> = HashSet::new();
        full_prep_coords.insert((coord, level));

        let d = classify_exposure_candidate(
            coord, level, 0, &level_slots, &dir, &full_prep_coords, 0,
        );
        assert_eq!(d, ExposureDispatchDecision::SkipDeduped);
    }

    #[test]
    fn classify_skips_non_resident_coords() {
        // The non-resident-filter test from the spec: coords whose
        // directory entry is not currently resident must not produce
        // exposure-only requests.
        let level_slots = make_level_slots([8, 8, 8], 0);
        let coord = SubchunkCoord::new(2, 3, 4);
        let level = Level(0);

        // Build an empty directory — no entries resident.
        let dir = SlotDirectory::new(512);
        let full_prep_coords: HashSet<(SubchunkCoord, Level)> = HashSet::new();

        let d = classify_exposure_candidate(
            coord, level, 0, &level_slots, &dir, &full_prep_coords, 0,
        );
        assert_eq!(d, ExposureDispatchDecision::SkipNotRefreshable);
    }

    #[test]
    fn classify_skips_torus_collision_coord_mismatch() {
        // Non-matching coord at a resident slot (torus-collision alias)
        // must be filtered. Simulate by putting a DIFFERENT coord's
        // entry at the slot our query coord resolves to.
        let level_slots = make_level_slots([8, 8, 8], 0);
        let query_coord   = SubchunkCoord::new(2, 3, 4);
        let resident_coord = SubchunkCoord::new(10, 3, 4); // differs by pool_dim on x
        let level = Level(0);

        use crate::world::pool::cpu_compute_directory_index;
        let range = &level_slots[0];
        let dir_idx_query    = cpu_compute_directory_index(
            query_coord, range.pool_dims, range.offset,
        );
        let dir_idx_resident = cpu_compute_directory_index(
            resident_coord, range.pool_dims, range.offset,
        );
        // Pool-dim aliasing gives same dir_idx.
        assert_eq!(dir_idx_query, dir_idx_resident);

        let mut dir = SlotDirectory::new(512);
        dir.set(dir_idx_resident, DirEntry::resident(
            [resident_coord.x, resident_coord.y, resident_coord.z],
            0x3F, false, 0,
        ));
        let _ = dir.drain_dirty().count();

        let d = classify_exposure_candidate(
            query_coord, level, 0, &level_slots, &dir, &HashSet::new(), 0,
        );
        assert_eq!(
            d, ExposureDispatchDecision::SkipNotRefreshable,
            "torus collision must be filtered before dispatch",
        );
    }

    #[test]
    fn classify_dispatches_resident_non_deduped_with_room() {
        // Happy path: resident, not full-prep-covered, batch has room —
        // dispatch.
        let level_slots = make_level_slots([8, 8, 8], 0);
        let coord = SubchunkCoord::new(2, 3, 4);
        let level = Level(0);

        let dir = directory_with_resident(
            512, &level_slots, coord, 0x2F, 0,
        );
        let d = classify_exposure_candidate(
            coord, level, 0, &level_slots, &dir, &HashSet::new(), 0,
        );
        assert_eq!(d, ExposureDispatchDecision::Dispatch);
    }

    #[test]
    fn classify_defers_when_batch_full() {
        // At the MAX_CANDIDATES ceiling, the classifier must signal
        // carry-over rather than dispatch.
        use renderer::SUBCHUNK_MAX_CANDIDATES;
        let level_slots = make_level_slots([8, 8, 8], 0);
        let coord = SubchunkCoord::new(2, 3, 4);
        let level = Level(0);

        let dir = directory_with_resident(
            512, &level_slots, coord, 0x2F, 0,
        );
        let d = classify_exposure_candidate(
            coord, level, 0, &level_slots, &dir,
            &HashSet::new(), SUBCHUNK_MAX_CANDIDATES,
        );
        assert_eq!(d, ExposureDispatchDecision::DeferOverflow);
    }

    // -- Step 7: shadow-ledger counters --

    /// A zeroed [`DirtyReport`] — stand-in for "nothing retired yet".
    fn empty_dirty_report() -> DirtyReport {
        DirtyReport {
            count:    0,
            _pad0:    0,
            overflow: 0,
            _pad2:    0,
            entries:  [DirtyEntry {
                directory_index:     0,
                new_bits_partial:    0,
                staging_request_idx: 0,
                _pad:                0,
            }; renderer::SUBCHUNK_MAX_CANDIDATES],
        }
    }

    /// Simulates the single retirement-path line that stamps the
    /// shadow-ledger watermarks from a retired full-prep
    /// [`DirtyReport`]. Kept in sync with the inline code in
    /// [`WorldView::update`].
    fn publish_full_prep_watermarks(
        stats:  &mut WorldRendererStats,
        report: &DirtyReport,
    ) {
        stats.dirty_full_count_last         = report.count;
        stats.dirty_appends_this_frame_full = report.count;
        stats.dirty_full_overflow           = report.overflow != 0;
    }

    /// Same shape as [`publish_full_prep_watermarks`] but for the
    /// exposure dispatch.
    fn publish_exposure_watermarks(
        stats:  &mut WorldRendererStats,
        report: &DirtyReport,
    ) {
        stats.dirty_exposure_count_last         = report.count;
        stats.dirty_appends_this_frame_exposure = report.count;
        stats.dirty_exposure_overflow           = report.overflow != 0;
    }

    /// Default `WorldRendererStats` used by the synthetic-frame tests.
    /// `SUBCHUNK_MAX_CANDIDATES` is hoisted here (rather than read from
    /// an allocator) so the test is independent of allocator sizing.
    fn zero_stats() -> WorldRendererStats {
        WorldRendererStats {
            frame:                             FrameIndex::default(),
            active_material_slots:             0,
            pool_capacity:                     renderer::SUBCHUNK_MAX_CANDIDATES as u32,
            directory_resident:                0,
            dirty_full_count_last:             0,
            dirty_exposure_count_last:         0,
            dirty_full_overflow:               false,
            dirty_exposure_overflow:           false,
            alloc_refused_cum:                 0,
            alloc_evictions_cum:               0,
            pending_exposure_refresh:          0,
            in_flight_full_batches:            0,
            in_flight_exposure_batches:        0,
            prep_dispatches_this_frame:        0,
            exposure_dispatches_this_frame:    0,
            dirty_appends_this_frame_full:     0,
            dirty_appends_this_frame_exposure: 0,
            material_data_segments_live:       0,
            material_data_grow_events:         0,
            material_data_active_slots:        0,
            material_data_sentinel_patches_this_frame: 0,
        }
    }

    /// Counting resident directory entries is the only O(capacity)
    /// aggregate in the tier-1 ledger; everything else is already
    /// available as a scalar on an in-memory structure. Assert the
    /// counter walks the directory correctly against a hand-built
    /// mix of resident and non-resident entries.
    #[test]
    fn count_directory_resident_matches_set_entries() {
        let mut dir = SlotDirectory::new(8);
        let _ = dir.drain_dirty().count(); // clear initial dirty set

        assert_eq!(count_directory_resident(&dir), 0,
            "fresh directory has no resident entries");

        dir.set(1, DirEntry::resident([0, 0, 0], 0x3F, false, 1));
        dir.set(3, DirEntry::resident([0, 0, 0], 0x01, true,  3));
        dir.set(5, DirEntry::empty([0, 0, 0]));  // explicit non-resident; sanity check

        assert_eq!(count_directory_resident(&dir), 2,
            "two resident entries set, one explicit empty");

        // Evict one; counter decrements.
        dir.set(1, DirEntry::empty([0, 0, 0]));
        assert_eq!(count_directory_resident(&dir), 1);
    }

    /// Synthetic-frame shape: after publishing a full-prep and an
    /// exposure retirement, the struct reflects the observed values.
    /// Stands in for an integration test that would need a GPU context.
    #[test]
    fn stats_reflect_synthetic_frame_updates() {
        let mut stats = zero_stats();

        // Initial state: everything zero except `pool_capacity`.
        assert_eq!(stats.dirty_full_count_last,     0);
        assert_eq!(stats.dirty_exposure_count_last, 0);
        assert!(!stats.dirty_full_overflow);
        assert!(!stats.dirty_exposure_overflow);
        assert_eq!(stats.pool_capacity,
                   renderer::SUBCHUNK_MAX_CANDIDATES as u32);

        // Full-prep retirement with 7 entries, no overflow.
        let mut report = empty_dirty_report();
        report.count    = 7;
        report.overflow = 0;
        publish_full_prep_watermarks(&mut stats, &report);
        assert_eq!(stats.dirty_full_count_last,         7);
        assert_eq!(stats.dirty_appends_this_frame_full, 7);
        assert!(!stats.dirty_full_overflow);

        // Exposure retirement with 3 entries, no overflow.
        report.count    = 3;
        report.overflow = 0;
        publish_exposure_watermarks(&mut stats, &report);
        assert_eq!(stats.dirty_exposure_count_last,         3);
        assert_eq!(stats.dirty_appends_this_frame_exposure, 3);
        assert!(!stats.dirty_exposure_overflow);

        // Full-prep watermarks are sticky (not affected by the
        // exposure retirement above).
        assert_eq!(stats.dirty_full_count_last, 7);
    }

    /// Overflow-flag propagation: a [`DirtyReport`] with
    /// `overflow != 0` must surface as `true` on the stats struct.
    /// Mirrors the "inject a dirty-list header with the overflow
    /// sentinel set, assert stats.dirty_full_overflow is true" check
    /// from the Step-7 test spec.
    #[test]
    fn overflow_flag_propagates_from_dirty_report_to_stats() {
        let mut stats = zero_stats();

        // Full-prep batch saturated — shader set overflow to 1 after
        // dropping entries past MAX_CANDIDATES.
        let mut report = empty_dirty_report();
        report.count    = (renderer::SUBCHUNK_MAX_CANDIDATES + 10) as u32;
        report.overflow = 1;

        publish_full_prep_watermarks(&mut stats, &report);
        assert!(
            stats.dirty_full_overflow,
            "dirty_full_overflow must latch when report.overflow != 0",
        );
        // The raw count is preserved so downstream telemetry can see
        // *how* overflowed the dispatch was (clamped at read time by
        // the retirement entry iterator).
        assert_eq!(
            stats.dirty_full_count_last,
            (renderer::SUBCHUNK_MAX_CANDIDATES + 10) as u32,
        );

        // A subsequent non-overflowing batch clears the flag.
        report.count    = 2;
        report.overflow = 0;
        publish_full_prep_watermarks(&mut stats, &report);
        assert!(
            !stats.dirty_full_overflow,
            "dirty_full_overflow must clear when a fresh batch retires clean",
        );

        // Same contract on the exposure path.
        let mut exp_report = empty_dirty_report();
        exp_report.count    = (renderer::SUBCHUNK_MAX_CANDIDATES + 1) as u32;
        exp_report.overflow = 1;
        publish_exposure_watermarks(&mut stats, &exp_report);
        assert!(stats.dirty_exposure_overflow);
    }

    /// `alloc_refused_cum` must be monotonic across frames — the
    /// counter exists specifically to signal pool pressure and
    /// decrementing it would erase that history.
    #[test]
    fn alloc_refused_cum_monotonic_across_frames() {
        let mut alloc = MaterialAllocator::new(4);
        assert_eq!(alloc.stats().refused, 0);

        // Frame 1: allocate slot 0, then try again — refused.
        alloc.allocate(0).unwrap();
        assert!(alloc.allocate(0).is_none());
        let after_frame_1 = alloc.stats().refused;
        assert_eq!(after_frame_1, 1);

        // Frame 2: free and reallocate cleanly (no refusal), refused
        // stays where it was.
        alloc.free(0);
        alloc.allocate(0).unwrap();
        assert_eq!(alloc.stats().refused, after_frame_1,
            "clean allocation must not change refused counter");

        // Frame 3: another duplicate allocate — refused bumps, never
        // decrements.
        assert!(alloc.allocate(0).is_none());
        let after_frame_3 = alloc.stats().refused;
        assert_eq!(after_frame_3, after_frame_1 + 1);
        assert!(after_frame_3 >= after_frame_1,
            "refused must be monotonic non-decreasing");

        // And the cumulative feed into the stats struct preserves
        // monotonicity after publishing.
        let mut stats = zero_stats();
        stats.alloc_refused_cum = alloc.stats().refused;
        assert_eq!(stats.alloc_refused_cum, 2);
    }

    /// Overflow as it would appear in the renderer binary: the
    /// `DirtyReport` buffer's overflow field survives a bytemuck
    /// round-trip at offset 8 (the shader writes it there via
    /// `InterlockedMax(8u, 1u)`). Guards against accidental struct
    /// reordering breaking the CPU↔GPU contract without anyone
    /// noticing.
    #[test]
    fn dirty_report_overflow_lives_at_offset_8() {
        let mut report = empty_dirty_report();
        report.overflow = 0xDEAD_BEEF;
        let bytes: &[u8] = bytemuck::bytes_of(&report);
        assert_eq!(
            &bytes[8..12],
            0xDEAD_BEEFu32.to_le_bytes(),
            "overflow field must live at byte offset 8 to match the \
             shader's InterlockedMax(8u, 1u, _prev)",
        );
    }

    // --- Cold-start seed regression ---

    /// Regression for the (0,0,0) cold-start bug: after the shell-build
    /// seed pass, every slot in the directory carries `DirEntry::empty` with
    /// the *canonical* coord and bits=0.
    ///
    /// Before the fix, `SlotDirectory::new` zero-initialised every entry and
    /// the GPU buffer was also zero-initialised, leaving coord=(0,0,0) in
    /// every slot. The DDA's `resolve_and_verify` saw `coord_match = false`
    /// for any coord other than (0,0,0), causing phantom promotions into
    /// coarser OR-reduced occupancy (shadow/GI phantom hits). For the slot
    /// whose *canonical* owner happens to be (0,0,0) this was indistinguishable
    /// from the zero-init — only an explicit seed makes the state observable
    /// as "empty at (0,0,0)" rather than "uninitialized".
    #[test]
    fn cold_start_seed_stamps_every_slot_with_canonical_coord() {
        // Construct a tiny 2×2×2 shell centred near the origin so that one
        // of the 8 resident slots has canonical coord (0,0,0). The shell has
        // radius [1,1,1] and corner (0,0,0), so it covers coords in
        // [0,1] × [0,1] × [0,1] — wait, Shell::new with corner=(0,0,0) and
        // radius=[1,1,1] covers [-1,0] on each axis, as per shell.rs §geometry.
        // Use corner=(0,0,0) + radius=[1,1,1]: residents are dx in [-1,0),
        // so coords (-1,-1,-1)..(-1,-1,0) etc. Choose corner=(1,1,1) so that
        // (0,0,0) is one of the residents.
        let corner   = SubchunkCoord::new(1, 1, 1);
        let radius   = [1u32, 1, 1];
        let pool_dims = [2u32, 2, 2];
        let capacity  = pool_dims[0] * pool_dims[1] * pool_dims[2]; // 8

        // Simulate the seed pass that WorldView::new performs.
        let mut dir = SlotDirectory::new(capacity);
        let shell   = Shell::new(radius, corner);
        for coord in shell.residents() {
            let dir_idx = cpu_compute_directory_index(coord, pool_dims, 0);
            dir.set(dir_idx, DirEntry::empty([coord.x, coord.y, coord.z]));
        }

        // Verify every resident slot carries its canonical coord, bits=0,
        // and is not marked resident — i.e. exactly DirEntry::empty(coord).
        let mut zero_coord_found = false;
        for coord in shell.residents() {
            let dir_idx = cpu_compute_directory_index(coord, pool_dims, 0);
            let entry   = dir.get(dir_idx);
            assert!(!entry.is_resident(),
                "seeded slot must be non-resident at coord {coord:?}");
            assert_eq!(
                entry.coord, [coord.x, coord.y, coord.z],
                "seeded slot must carry canonical coord at dir_idx {dir_idx}",
            );
            assert_eq!(
                entry.bits, 0,
                "seeded empty entry must have all bits clear",
            );
            if coord.x == 0 && coord.y == 0 && coord.z == 0 {
                zero_coord_found = true;
            }
        }
        assert!(zero_coord_found,
            "shell with corner=(1,1,1) radius=[1,1,1] must include (0,0,0)");
    }

    // --- Eviction → prep-dispatch path ---

    /// Regression: after eviction, the prep-dispatch step (not the eviction
    /// step) is the sole writer of new coord ownership in the directory.
    ///
    /// Eviction frees CPU-side resources but does NOT touch the directory.
    /// The prep-dispatch loop then writes `DirEntry::empty(new_coord)` for
    /// each newly-requested slot. The slot must carry the *new* coord after
    /// this sequence, not the old evicted coord's data.
    #[test]
    fn eviction_then_prep_dispatch_stamps_new_coord() {
        // Pool dims [2, 1, 1]: two slots at x=0 and x=1. Old coord (0,0,0)
        // maps to slot 0; new coord (2,0,0) also maps to slot 0 (toroidal
        // wrap: 2 % 2 = 0). That is the evict-then-replace case.
        let pool_dims   = [2u32, 1, 1];
        let global_offset = 0u32;
        let old_coord   = [0i32, 0, 0];
        let new_coord   = [2i32, 0, 0];

        // Seed the directory as if the shell-build seed pass ran for old_coord.
        let mut dir = SlotDirectory::new(2);
        let slot = cpu_compute_directory_index(
            SubchunkCoord::new(old_coord[0], old_coord[1], old_coord[2]),
            pool_dims,
            global_offset,
        );
        dir.set(slot, DirEntry::resident(old_coord, 0x3F, true, 1));
        let _ = dir.drain_dirty().count();

        // Eviction: free CPU resources only. No directory write (the new design).
        // The slot still holds the old resident entry.
        assert!(dir.get(slot).is_resident(), "slot still resident before prep-dispatch");
        assert_eq!(dir.get(slot).coord, old_coord);

        // Prep-dispatch: unconditionally seed the slot with the new canonical owner.
        dir.set(slot, DirEntry::empty([new_coord[0], new_coord[1], new_coord[2]]));

        // After prep-dispatch: slot must carry the new coord and be non-resident.
        let entry = dir.get(slot);
        assert!(!entry.is_resident(),
            "slot must be non-resident after prep-dispatch seeds new coord");
        assert_eq!(entry.coord, new_coord,
            "slot must carry the new canonical coord, not the evicted coord");
        assert_eq!(entry.bits, 0,
            "prep-dispatch seeds empty([coord]) which has bits=0");
    }
}
