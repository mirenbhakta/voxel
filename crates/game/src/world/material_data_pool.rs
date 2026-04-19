//! CPU-authoritative allocator for per-voxel material-data slots.
//!
//! Companion to [`MaterialAllocator`](super::material_pool::MaterialAllocator)
//! but **fundamentally different** in contract:
//!
//! - `MaterialAllocator` is an **identity** allocator for 64-byte occupancy
//!   blocks. Its output equals `directory_index` under today's 1:1 policy.
//! - `MaterialDataPool` is a **non-identity** allocator for 1 KB material-
//!   data blocks (512 × u16 per-voxel material IDs). Its output is a flat
//!   global slot index opaque to callers. `DirEntry.material_data_slot` is
//!   the single source of truth for "which slot owns this sub-chunk's
//!   per-voxel materials".
//!
//! The non-identity policy is deliberate: material-data storage scales with
//! the resident set (capped by GPU memory) rather than with the directory's
//! world-extent-driven capacity. Conflating the two pools would foot-gun
//! future extensions (M2 bindless textures, M3 PBR) into
//! `failure-allocator-identity-drift` territory the moment the shape
//! diverges.
//!
//! # Segmentation
//!
//! The pool is backed by a **binding array of 64 MB segments** on the GPU
//! ([`SLOTS_PER_SEGMENT`] slots × 1 KB per slot = 64 MB). The CPU allocator
//! is segmentation-aware: the addressable slot space grows by
//! [`SLOTS_PER_SEGMENT`] every time [`MaterialDataPool::grow`] is called,
//! and every slot the allocator returns is a **flat global index** valid
//! across all live segments. Segment/local decomposition is the shader's
//! concern, not the caller's.
//!
//! Ceiling: [`MAX_SEGMENTS`] (= 16, = 1 GB addressable). Beyond that,
//! [`MaterialDataPool::grow`] errors with [`PoolCeiling`]; the materializer
//! leaves the affected sub-chunks at the `INVALID` sentinel so the shade
//! shader draws magenta on un-allocatable sub-chunks.
//!
//! # Free-list semantics
//!
//! A single global LIFO free-list. Recently-freed slots are reused before
//! the allocator grows, which gives natural low-index compaction without
//! per-segment bookkeeping — the residency working set stays in the
//! low-index segments during steady-state motion.
//!
//! # Ordering contract
//!
//! The allocator is **pure**: it decides slot identity, nothing more. The
//! decision to defer a sub-chunk one frame under exhaustion, the decision to
//! append a new GPU segment, and the order in which DirEntry / free-list
//! updates are issued, all live in the materializer caller
//! (`crates/game/src/world_view.rs`) — not here. Keeping the allocator free
//! of frame-boundary concerns makes it unit-testable in isolation and keeps
//! its invariants cleanly stated.

#![allow(dead_code)]

// --- Public constants ---

/// Number of 1 KB material-data slots per 64 MB GPU segment.
///
/// Matches the binding-array shader declaration (`material_data_pool[MAX_SEGMENTS]`
/// where each element is `StructuredBuffer<MaterialBlock>` sized for
/// `SLOTS_PER_SEGMENT` elements of stride 1024). Kept compile-time so the
/// shader's segment/local decompose (`slot / SLOTS_PER_SEGMENT`) lowers to
/// a branch-free shift+mask.
///
/// Tests that want to exercise the segmented behaviour at a tractable
/// capacity construct a `MaterialDataPool` with a test-local
/// `slots_per_segment` via [`MaterialDataPool::with_slots_per_segment`].
pub const SLOTS_PER_SEGMENT: u32 = 65536;

/// Maximum number of live 64 MB segments the pool can grow to.
///
/// 16 × 64 MB = 1 GB hard ceiling, matching the scaffold renderer's
/// proven sizing (see `decision-volumetric-material-system`). Raising this
/// is a shader recompile — the binding-array declaration carries the
/// count.
pub const MAX_SEGMENTS: u32 = 16;

/// Sentinel value for [`DirEntry::material_data_slot`] under a
/// non-resident or ceiling-deferred entry. Chosen as `u32::MAX` so no real
/// global-slot index collides (capacity can never exceed
/// `SLOTS_PER_SEGMENT * MAX_SEGMENTS < u32::MAX`).
pub const MATERIAL_DATA_SLOT_INVALID: u32 = 0xFFFF_FFFF;

// --- PoolCeiling ---

/// Emitted by [`MaterialDataPool::grow`] when the pool has reached
/// [`MAX_SEGMENTS`] and cannot expand further. The caller should accept
/// persistent sentinel output for the deferred sub-chunks and log the
/// transition once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolCeiling;

impl std::fmt::Display for PoolCeiling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "material-data pool ceiling reached ({} segments, {} slots)",
            MAX_SEGMENTS,
            MAX_SEGMENTS as u64 * SLOTS_PER_SEGMENT as u64,
        )
    }
}

impl std::error::Error for PoolCeiling {}

// --- MaterialDataPoolStats ---

/// Snapshot of allocator state for the per-frame stats overlay. Cheap
/// copy-out — no allocations, no locks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialDataPoolStats {
    pub segments_live: u32,
    pub grow_events:   u64,
    pub active:        u32,
    pub sentinel_patches_this_frame: u32,
}

// --- SlotRecord ---

/// Per-slot bookkeeping. Today just tracks whether a slot is live.
/// Kept as a struct (rather than a `Vec<bool>` or bitset) so future fields
/// — e.g. the directory_index that owns the slot, or a version for
/// leak-check assertions — can land without changing the allocator's
/// surface area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotRecord {
    occupied: bool,
}

impl SlotRecord {
    const fn free() -> Self {
        Self { occupied: false }
    }
}

// --- MaterialDataPool ---

/// Segmented, unbounded-with-ceiling LIFO allocator over a flat global
/// slot-index space.
///
/// The allocator owns slot identity. Callers call
/// [`MaterialDataPool::try_allocate`] to request a slot, [`Self::free`] to
/// release, and [`Self::grow`] to extend capacity by one segment. No caller
/// ever constructs a slot index or assumes a specific identity — the
/// allocator's output is the authoritative value and is round-tripped
/// through `DirEntry`.
pub struct MaterialDataPool {
    /// Number of live 64 MB segments. Invariants:
    ///   `records.len() == segments_live * slots_per_segment`
    ///   `segments_live <= MAX_SEGMENTS`
    segments_live: u32,

    /// Per-slot bookkeeping. `records[i].occupied == true` ⇒ slot `i` is
    /// allocated to some caller.
    records: Vec<SlotRecord>,

    /// LIFO free list of global slot indices. Every free slot appears
    /// exactly once. The allocator pops from the back on allocate and
    /// pushes on free; the LIFO order keeps recently-freed slots hot so
    /// the allocator naturally reuses the low-index segments before
    /// growing.
    free_list: Vec<u32>,

    /// Count of live allocations (`records` entries with `occupied == true`).
    /// Incrementally maintained so [`Self::stats`] is O(1).
    active: u32,

    /// Monotonic count of successful [`Self::grow`] calls. Surfaced via
    /// [`MaterialDataPoolStats`] for the one-shot grow log.
    grow_events: u64,

    /// Counter for this-frame "try_allocate returned None" events. The
    /// materializer calls [`Self::reset_frame_sentinel_counter`] at the
    /// start of each materialization pass and reads the tally for the
    /// log-once-per-transition trace.
    sentinel_patches_this_frame: u32,

    /// Test-overridable slot count per segment. Production callers pass
    /// [`SLOTS_PER_SEGMENT`]; unit tests pass a smaller value to exercise
    /// the segmented grow path at tractable sizes.
    slots_per_segment: u32,

    /// Test-overridable segment ceiling. Production callers pass
    /// [`MAX_SEGMENTS`]; unit tests pass a smaller value to exercise the
    /// ceiling path without allocating a GB of records.
    max_segments: u32,
}

impl MaterialDataPool {
    /// Construct an allocator with zero live segments.
    ///
    /// The first [`Self::try_allocate`] returns `None` until the caller
    /// calls [`Self::grow`] once (typical renderer init). This matches the
    /// decision-material-system-m1-sparse contract: the allocator never
    /// pre-commits GPU buffers it cannot see, and the renderer's segment
    /// list must come up in lockstep with the allocator's segment count.
    pub fn new() -> Self {
        Self::with_capacity_params(SLOTS_PER_SEGMENT, MAX_SEGMENTS)
    }

    /// Test-only constructor that lets unit tests dial in a smaller
    /// segment size / ceiling. Production code uses [`Self::new`].
    #[cfg(test)]
    pub fn with_slots_per_segment(slots_per_segment: u32) -> Self {
        Self::with_capacity_params(slots_per_segment, MAX_SEGMENTS)
    }

    /// Test-only constructor that lets unit tests dial in both the
    /// segment size and the segment ceiling. Production code uses
    /// [`Self::new`].
    #[cfg(test)]
    pub fn with_capacity(slots_per_segment: u32, max_segments: u32) -> Self {
        Self::with_capacity_params(slots_per_segment, max_segments)
    }

    fn with_capacity_params(slots_per_segment: u32, max_segments: u32) -> Self {
        assert!(slots_per_segment > 0, "slots_per_segment must be positive");
        assert!(max_segments > 0, "max_segments must be positive");
        Self {
            segments_live: 0,
            records: Vec::new(),
            free_list: Vec::new(),
            active: 0,
            grow_events: 0,
            sentinel_patches_this_frame: 0,
            slots_per_segment,
            max_segments,
        }
    }

    /// Number of live segments (read by the renderer to decide how many
    /// entries of the binding array to populate).
    pub fn segments_live(&self) -> u32 {
        self.segments_live
    }

    /// Number of currently-allocated slots.
    pub fn active(&self) -> u32 {
        self.active
    }

    /// Snapshot of allocator state for the per-frame stats overlay.
    pub fn stats(&self) -> MaterialDataPoolStats {
        MaterialDataPoolStats {
            segments_live: self.segments_live,
            grow_events:   self.grow_events,
            active:        self.active,
            sentinel_patches_this_frame: self.sentinel_patches_this_frame,
        }
    }

    /// Attempt to allocate one slot. Returns `Some(slot)` on success —
    /// the returned `slot` is a flat global index valid across all live
    /// segments. Returns `None` when the free list is empty, signalling
    /// the caller that [`Self::grow`] is required before retry.
    ///
    /// This method does not itself call `grow`. The grow decision is
    /// the materializer's: it batches exhaustion across a frame, logs
    /// stats, and rebinds the GPU segment list at the frame boundary.
    /// Coupling grow to allocate would force every caller into a
    /// rebind-mid-frame path that this architecture explicitly rejects.
    #[must_use = "callers must route the allocated slot into DirEntry; dropping it leaks the slot"]
    pub fn try_allocate(&mut self) -> Option<u32> {
        let slot = self.free_list.pop()?;
        debug_assert!(
            (slot as usize) < self.records.len(),
            "free_list held slot {slot} past records.len() {}",
            self.records.len(),
        );
        let record = &mut self.records[slot as usize];
        debug_assert!(
            !record.occupied,
            "free_list contained already-occupied slot {slot}",
        );
        record.occupied = true;
        self.active += 1;

        if self.active_exceeds_counter_capacity() {
            self.sentinel_patches_this_frame = 0;
        }

        Some(slot)
    }

    /// Extend the pool's addressable capacity by one segment
    /// ([`SLOTS_PER_SEGMENT`] slots). All new slot indices are pushed
    /// onto the free list in ascending order so the first subsequent
    /// [`Self::try_allocate`] pops the *highest* new index — acceptable
    /// because the LIFO reuse behaviour only matters after at least one
    /// free has happened. For a freshly-grown segment, low-index
    /// compaction is preserved by subsequent alloc/free cycles rather
    /// than by the push order.
    ///
    /// Returns [`PoolCeiling`] when the pool has already reached
    /// [`MAX_SEGMENTS`]. The caller is expected to log once and accept
    /// persistent sentinel output for the affected sub-chunks.
    pub fn grow(&mut self) -> Result<(), PoolCeiling> {
        if self.segments_live >= self.max_segments {
            return Err(PoolCeiling);
        }

        let first_new = self.segments_live * self.slots_per_segment;
        let last_new  = first_new + self.slots_per_segment;

        self.records.resize(last_new as usize, SlotRecord::free());
        self.free_list.reserve(self.slots_per_segment as usize);
        for slot in first_new..last_new {
            self.free_list.push(slot);
        }

        self.segments_live += 1;
        self.grow_events   += 1;
        Ok(())
    }

    /// Release `slot` back to the free list and clear its record.
    ///
    /// # Panics
    ///
    /// Debug builds panic on:
    /// - `slot` past the allocator's addressable range, or
    /// - `slot` whose record is already free (double-free / caller bug).
    pub fn free(&mut self, slot: u32) {
        debug_assert!(
            (slot as usize) < self.records.len(),
            "free: slot {slot} past records.len() {}",
            self.records.len(),
        );
        let record = &mut self.records[slot as usize];
        debug_assert!(
            record.occupied,
            "free: slot {slot} was already free",
        );
        record.occupied = false;
        self.active = self.active.saturating_sub(1);
        self.free_list.push(slot);
    }

    /// Increment the "try_allocate returned None" counter by one. The
    /// materializer calls this when it observes a sentinel-requiring
    /// deferral; the counter is drained each frame by
    /// [`Self::reset_frame_sentinel_counter`] and surfaces on the stats
    /// overlay.
    pub fn note_sentinel_patch(&mut self) {
        self.sentinel_patches_this_frame =
            self.sentinel_patches_this_frame.saturating_add(1);
    }

    /// Reset the per-frame sentinel counter. Called once per frame by
    /// the materializer at the top of its dirty-list pass, immediately
    /// before it begins handing out slots.
    pub fn reset_frame_sentinel_counter(&mut self) {
        self.sentinel_patches_this_frame = 0;
    }

    /// Current value of the per-frame sentinel counter.
    pub fn sentinel_patches_this_frame(&self) -> u32 {
        self.sentinel_patches_this_frame
    }

    /// Total slot capacity currently addressable (= `segments_live *
    /// slots_per_segment`).
    pub fn capacity(&self) -> u64 {
        self.segments_live as u64 * self.slots_per_segment as u64
    }

    #[inline]
    fn active_exceeds_counter_capacity(&self) -> bool {
        // The active counter is a plain u32 and can't realistically
        // overflow (capacity caps at MAX_SEGMENTS × SLOTS_PER_SEGMENT =
        // 1M slots < 2^32). Helper kept for clarity at the single call
        // site; always returns false in practice.
        false
    }
}

impl Default for MaterialDataPool {
    fn default() -> Self {
        Self::new()
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for `failure-allocator-identity-drift`. The
    /// allocator's slot identity is opaque to callers — but out-of-order
    /// free/realloc cycles must yield slots that the caller can recover
    /// via a DirEntry mirror. We simulate that by recording every
    /// (caller_id → allocated_slot) mapping and asserting it matches the
    /// allocator's final `active`-set membership.
    #[test]
    fn allocated_slot_is_recorded_and_not_drifted() {
        // Small segment so we don't spend 64 MB on a unit test.
        let mut pool = MaterialDataPool::with_slots_per_segment(16);
        pool.grow().expect("first grow always succeeds under default ceiling");

        // 4 callers allocate in order and record their slots in a map.
        let mut owner_to_slot: [Option<u32>; 4] = [None; 4];
        for slot_ref in owner_to_slot.iter_mut() {
            let s = pool.try_allocate().expect("capacity 16 holds 4 allocs");
            *slot_ref = Some(s);
        }

        // Callers 1 and 3 free. Owner-to-slot remains the truth.
        pool.free(owner_to_slot[1].unwrap());
        pool.free(owner_to_slot[3].unwrap());
        owner_to_slot[1] = None;
        owner_to_slot[3] = None;

        // Two new callers allocate and record their slots.
        let s4 = pool.try_allocate().unwrap();
        let s5 = pool.try_allocate().unwrap();
        owner_to_slot[1] = Some(s4);
        owner_to_slot[3] = Some(s5);

        // Every owner's slot is distinct and corresponds to an allocator
        // decision — no drift.
        let mut slots: Vec<u32> = owner_to_slot.iter().filter_map(|s| *s).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), 4, "4 owners must hold 4 distinct live slots");
        assert_eq!(pool.active(), 4);
    }

    #[test]
    fn try_allocate_on_empty_pool_returns_none() {
        let mut pool = MaterialDataPool::with_slots_per_segment(4);
        assert_eq!(pool.try_allocate(), None);
    }

    #[test]
    fn try_allocate_returns_none_when_full() {
        let mut pool = MaterialDataPool::with_slots_per_segment(3);
        pool.grow().unwrap();
        let _ = pool.try_allocate().unwrap();
        let _ = pool.try_allocate().unwrap();
        let _ = pool.try_allocate().unwrap();
        assert_eq!(pool.try_allocate(), None);
    }

    #[test]
    fn grow_extends_addressable_capacity() {
        let mut pool = MaterialDataPool::with_slots_per_segment(2);
        pool.grow().unwrap();
        let _ = pool.try_allocate().unwrap();
        let _ = pool.try_allocate().unwrap();
        assert_eq!(pool.try_allocate(), None, "segment 0 full");

        pool.grow().unwrap();
        let slot = pool.try_allocate().expect("segment 1 has capacity");
        assert!(
            slot >= 2,
            "slot from segment 1 must be >= slots_per_segment ({slot} < 2)",
        );
    }

    #[test]
    fn free_list_is_lifo_for_low_fragmentation() {
        let mut pool = MaterialDataPool::with_slots_per_segment(4);
        pool.grow().unwrap();
        let a = pool.try_allocate().unwrap();
        let b = pool.try_allocate().unwrap();
        let c = pool.try_allocate().unwrap();
        // Sanity: first segment's alloc order is sequential because the
        // grow path pushed the free list in ascending order.
        assert_ne!(a, b);
        assert_ne!(b, c);

        pool.free(b);
        let reused = pool.try_allocate().unwrap();
        assert_eq!(reused, b, "LIFO reuse of the just-freed middle slot");
    }

    #[test]
    fn grow_past_max_segments_errors() {
        // Cap at 4 segments of 2 slots each = 8 slots total.
        let mut pool = MaterialDataPool::with_capacity(2, 4);
        for _ in 0..4 {
            pool.grow().expect("each grow up to MAX_SEGMENTS succeeds");
        }
        assert_eq!(pool.grow(), Err(PoolCeiling));
        assert_eq!(pool.segments_live(), 4);
    }

    #[test]
    fn sentinel_counter_reset_and_note() {
        let mut pool = MaterialDataPool::with_slots_per_segment(2);
        assert_eq!(pool.sentinel_patches_this_frame(), 0);
        pool.note_sentinel_patch();
        pool.note_sentinel_patch();
        assert_eq!(pool.sentinel_patches_this_frame(), 2);
        pool.reset_frame_sentinel_counter();
        assert_eq!(pool.sentinel_patches_this_frame(), 0);
    }

    #[test]
    fn stats_reports_live_segments_and_grow_events() {
        let mut pool = MaterialDataPool::with_slots_per_segment(2);
        let s0 = pool.stats();
        assert_eq!(s0.segments_live, 0);
        assert_eq!(s0.grow_events,   0);
        assert_eq!(s0.active,        0);

        pool.grow().unwrap();
        let s1 = pool.stats();
        assert_eq!(s1.segments_live, 1);
        assert_eq!(s1.grow_events,   1);

        let _ = pool.try_allocate().unwrap();
        let s2 = pool.stats();
        assert_eq!(s2.active, 1);
    }

    #[test]
    fn free_then_realloc_reuses_the_same_slot() {
        let mut pool = MaterialDataPool::with_slots_per_segment(2);
        pool.grow().unwrap();
        let a = pool.try_allocate().unwrap();
        pool.free(a);
        let a2 = pool.try_allocate().unwrap();
        assert_eq!(a, a2, "most-recently-freed slot reused first");
    }

    #[test]
    fn try_allocate_signature_has_no_caller_supplied_slot_id() {
        // Specifically guard against anyone inadvertently reintroducing
        // identity `allocate(dir_idx)` semantics here. The caller never
        // supplies an argument to `try_allocate` — slot identity is the
        // allocator's. This test asserts the contract at the use-site
        // level: arbitrary distinct callers get distinct slots without
        // having to coordinate indices.
        let mut pool = MaterialDataPool::with_slots_per_segment(4);
        pool.grow().unwrap();
        let mut slots = Vec::new();
        for _ in 0..4 {
            slots.push(pool.try_allocate().unwrap());
        }
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots.len(),
            4,
            "4 try_allocate calls against a 4-slot pool must yield 4 distinct slots",
        );
    }
}
