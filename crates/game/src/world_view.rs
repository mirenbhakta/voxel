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

use std::collections::VecDeque;
use std::sync::Arc;

use renderer::{
    FrameIndex, LodMaskUniform, OverflowPolicy, PrepRequest as GpuPrepRequest,
    ReadbackChannel, RendererContext, SUBCHUNK_MAX_CANDIDATES, SUBCHUNK_MAX_LEVELS,
    SubchunkInstance, WorldRenderer, nodes,
};
use renderer::graph::{BufferPool, RenderGraph};
use renderer::DirtyReport;

use crate::world::coord::{Level, SubchunkCoord};
use crate::world::pool::SlotId;
use crate::world::residency::{LevelConfig, OccupancySummary, Residency, RequestId};
use crate::world::subchunk::SUBCHUNK_EDGE;

// --- WorldView ---

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

    // --- Eviction bookkeeping ---
    evicted_last_update: usize,
    evicted_total:       u64,
}

/// One outstanding prep dispatch. Pops off the front of `in_flight` when
/// its readback retires.
struct InFlightBatch {
    frame:       FrameIndex,
    completions: Vec<(RequestId, u32)>,
}

impl WorldView {
    /// Construct the view with the given levels, centered on
    /// `initial_camera`.
    ///
    /// The first frame has no retired readbacks yet, so the live GPU
    /// occupancy buffer stays zero-initialised until the prep dispatch
    /// recorded on frame 0 retires (typically 1–2 frames later). The
    /// render pass still runs each frame — the cull shader reads each
    /// slot's exposure from `live_exposure_buf`, which is zero-initialised
    /// until a patch lands, so every slot is rejected trivially. Expect
    /// the first one or two rendered frames to be empty before content
    /// starts to appear.
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
            let capacity = (2 * cfg.radius[0])
                         * (2 * cfg.radius[1])
                         * (2 * cfg.radius[2]);
            level_slots.push(LevelSlotRange { offset, capacity });
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
        let renderer  = Arc::new(WorldRenderer::new(ctx));
        let channel   = ReadbackChannel::<DirtyReport>::new(
            ctx, pool, "subchunk_prep_readback", OverflowPolicy::Panic,
        );

        Self {
            residency,
            renderer,
            configs:     configs.to_vec(),
            level_slots,
            instances:   [SubchunkInstance::padding(); SUBCHUNK_MAX_CANDIDATES],
            channel,
            in_flight:   VecDeque::new(),
            evicted_last_update: 0,
            evicted_total:       0,
        }
    }

    /// Borrow the GPU renderer for render-graph registration.
    pub fn renderer(&self) -> &Arc<WorldRenderer> {
        &self.renderer
    }

    pub fn evicted_last_update(&self) -> usize {
        self.evicted_last_update
    }

    pub fn evicted_total(&self) -> u64 {
        self.evicted_total
    }

    /// Top-of-frame tick.
    ///
    /// 1. Polls wgpu so any pending `map_async` callbacks fire.
    /// 2. Retires completed prep readbacks into residency, patching their
    ///    dirty slots through the graph.
    /// 3. Recenters the residency against `world_pos`, accumulating
    ///    evictions into the diagnostic counters.
    /// 4. Dispatches any newly-queued prep requests through the graph,
    ///    reserving a readback slot for the resulting dirty list.
    /// 5. Uploads the fresh instance array and LOD mask uniform.
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
        // --- 1. Poll for readback callbacks ---
        ctx.device().poll(wgpu::PollType::Poll).expect("device.poll failed");

        // --- 2. Retire completed readbacks ---
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

            // The exposure byte itself now lives in the renderer's
            // `live_exposure_buf` and is patched by the same copy pass
            // that commits the occupancy bytes. The CPU residency only
            // needs to mark the slot resident — no summary payload is
            // stored here.
            for (id, _slot) in batch.completions {
                self.residency.complete_prep(
                    id,
                    OccupancySummary::Mixed,
                    (),
                );
            }

            // Patch the live occupancy buffer from the staging writes the
            // prep shader made. `entries[..count]` are the shader-reported
            // dirty slots for this dispatch.
            let count = (report.count as usize).min(report.entries.len());
            let dirty_slots: Vec<u32> = report.entries[..count]
                .iter()
                .map(|e| e.slot)
                .collect();
            nodes::subchunk_patch(graph, &self.renderer, &dirty_slots);
        }

        // --- 3. Recenter residency ---
        let evictions = self.residency.update_camera(world_pos);
        self.evicted_last_update = evictions.len();
        self.evicted_total       = self.evicted_total.saturating_add(evictions.len() as u64);
        drop(evictions);

        // --- 4. Dispatch new prep requests ---
        let new_requests = self.residency.take_prep_requests();
        if !new_requests.is_empty() {
            let mut gpu_requests = Vec::with_capacity(new_requests.len());
            let mut completions  = Vec::with_capacity(new_requests.len());

            for req in &new_requests {
                let Some(level_idx) = self.level_index(req.level) else {
                    debug_assert!(false, "prep request for unknown level {:?}", req.level);
                    continue;
                };
                let global = self.global_slot(level_idx, req.slot);

                gpu_requests.push(GpuPrepRequest {
                    coord: [req.coord.x, req.coord.y, req.coord.z],
                    level: req.level.0 as u32,
                    slot:  global,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                });
                completions.push((req.id, global));
            }

            // Upload the CPU-side request list into the renderer's prep
            // buffer so the prep compute pass reads it.
            self.renderer.write_prep_requests(ctx, &gpu_requests);

            // Reserve this frame's readback slot and wire the prep pass
            // to write into it. Panic on overflow is appropriate: reserve
            // returning None here would mean the previous cycle's readback
            // never retired — a subsystem bug, not a load condition.
            let dst = self.channel
                .reserve(frame)
                .expect("ReadbackChannel::reserve: previous prep readback never retired")
                .clone();
            let dst_h = graph.import_buffer(dst);
            nodes::subchunk_prep(graph, &self.renderer, dst_h, gpu_requests.len() as u32);

            self.in_flight.push_back(InFlightBatch { frame, completions });
        }

        // --- 5. Upload instance array + LOD mask ---
        self.rebuild_instances(ctx);
        self.upload_lod_mask(ctx);
    }

    /// Post-`end_frame` tick: arm the channel's `map_async` callback on
    /// the submission fence so the reserved slot retires at the right
    /// time. No-op if no prep request was dispatched this frame.
    pub fn commit_submit(&mut self, frame: FrameIndex, submission: wgpu::SubmissionIndex) {
        // Only arm the callback if this frame's update actually reserved
        // a slot — otherwise the channel is still in `Empty` for this
        // slot and `commit_submit` would panic.
        let dispatched_this_frame = self.in_flight.back()
            .map(|b| b.frame == frame)
            .unwrap_or(false);
        if dispatched_this_frame {
            self.channel.commit_submit(frame, submission);
        }
    }

    // --- internal ---

    /// Rebuild `instances[..]` from every level's occupied slots and push
    /// the full array to the GPU.
    fn rebuild_instances(&mut self, ctx: &RendererContext) {
        // Reset to padding so any unused tail is rejected by the cull
        // shader before it reads any per-slot buffer.
        self.instances.fill(SubchunkInstance::padding());

        let mut i = 0usize;
        for (level_idx, cfg) in self.configs.iter().enumerate() {
            let level = cfg.level;
            let Some(pool) = self.residency.pool(level) else { continue; };
            let base = self.level_slots[level_idx].offset;
            for (coord, pool_slot, _) in pool.occupied() {
                if i >= SUBCHUNK_MAX_CANDIDATES {
                    debug_assert!(false, "resident count exceeds MAX_CANDIDATES");
                    break;
                }
                self.instances[i] = SubchunkInstance::new(
                    world_origin(coord, level),
                    base + pool_slot.0,
                    level.0,
                );
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
}

// --- LevelSlotRange ---

struct LevelSlotRange {
    offset:   u32,
    capacity: u32,
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
