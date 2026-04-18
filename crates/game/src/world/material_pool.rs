//! CPU-authoritative allocator for material-storage slot indices.
//!
//! The sub-chunk material pool has fixed capacity and the allocator decides
//! which `directory_index` is currently using each slot. Through Step 4 of
//! the directory-indirection refactor there is a trivial 1:1 mapping between
//! directory entries and material slots (`capacity == total_directory_capacity`),
//! and the allocator enforces this **identity policy**:
//!
//! > `allocate(dir_idx)` always returns `Some(dir_idx)`, provided that slot
//! > is currently free. It never returns some other slot and it never remaps.
//!
//! This is load-bearing for the renderer: the VS/PS stack reads
//! `g_material_pool[directory_index]` directly (through the instance's
//! `slot_mask`), so for a patch that writes into `material_pool[allocated_slot]`
//! to be visible at the correct draw instance, `allocated_slot` must equal
//! `directory_index`. A free-stack-style LIFO allocator breaks this invariant
//! after the first free/reallocate cycle — live patches land at the wrong
//! pool offset and the cross-wiring manifests as "right sub-chunk position,
//! wrong voxels" in the rendered frame.
//!
//! Later steps (TTL eviction, decoupled pool capacity) replace this identity
//! policy with a real allocator. At that point, VS/PS must also be re-routed
//! to read through the directory's `material_slot` field — see the `todo!`
//! breadcrumb noted in Step 7.
//!
//! # Data flow
//!
//! The allocator is the *single writer* for the "which directory owns which
//! material slot" relationship. Residency calls [`MaterialAllocator::allocate`]
//! when it inserts a new resident entry; [`MaterialAllocator::free`] when it
//! evicts one. [`MaterialAllocator::touch`] is called from the per-frame
//! residency walk so the TTL eviction policy (later step) can see which slots
//! have been visible recently.
//!
//! The allocator does not touch the directory or the GPU buffers directly.
//! Callers route its outputs into [`SlotDirectory`](super::slot_directory::SlotDirectory)
//! and from there into `WorldRenderer::write_directory_entries`.

#![allow(dead_code)]

use renderer::FrameIndex;

// --- MaterialAllocator ---

/// Fixed-capacity allocator enforcing the identity mapping
/// `directory_index → directory_index` on every `allocate` call.
///
/// Backed by a dense `Vec<SlotRecord>`. The "which slot is free" question
/// is answered by looking up the record for that slot — no free stack is
/// needed because callers already know which slot they want.
///
/// Future work (Step 7+): replace this with a real eviction-capable allocator
/// when `pool_capacity < directory_capacity`. At that point, the identity
/// property goes away and VS/PS must be re-routed to read the allocated
/// slot from the directory's `material_slot` field rather than relying on
/// `directory_index == material_slot`.
pub struct MaterialAllocator {
    capacity: u32,
    records:  Vec<SlotRecord>,

    // --- statistics for the shadow-ledger in Step 7 ---
    active:    u32,
    evictions: u64,
    refused:   u64,
}

impl MaterialAllocator {
    /// Construct an allocator with all `capacity` slots free.
    pub fn new(capacity: u32) -> Self {
        let capacity_usize = capacity as usize;
        let mut records = Vec::with_capacity(capacity_usize);
        records.resize_with(capacity_usize, SlotRecord::free);

        Self {
            capacity,
            records,
            active:    0,
            evictions: 0,
            refused:   0,
        }
    }

    /// Total slot count this allocator manages.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of currently-allocated slots.
    pub fn active(&self) -> u32 {
        self.active
    }

    /// Number of free slots.
    pub fn free_count(&self) -> u32 {
        self.capacity - self.active
    }

    /// Allocate the slot whose index equals `directory_index` (identity
    /// policy). Returns `Some(directory_index)` on success.
    ///
    /// # Identity invariant
    ///
    /// The returned slot is always exactly `directory_index`. Callers
    /// depend on this: the rendering VS/PS reads `material_pool[directory_index]`
    /// directly, so the patch copy MUST land at the same index. Breaking
    /// the identity mapping cross-wires material data to the wrong draw
    /// instance.
    ///
    /// # Panics / Returns None
    ///
    /// - Panics (debug build) if `directory_index >= capacity` — the caller
    ///   is asking for a slot outside this pool. This is a programmer error;
    ///   residency / directory layers are sized to fit.
    /// - Returns `None` if the slot is already allocated. Under the Step 3
    ///   residency invariants this cannot happen (each distinct `directory_index`
    ///   is allocated/freed in balanced pairs and residency never double-allocates).
    ///   Callers treat `None` as an unreachable state.
    pub fn allocate(&mut self, directory_index: u32) -> Option<u32> {
        debug_assert!(
            directory_index < self.capacity,
            "allocate: directory_index {directory_index} >= capacity {}",
            self.capacity,
        );

        let record = &mut self.records[directory_index as usize];
        if !record.is_free() {
            self.refused = self.refused.saturating_add(1);
            return None;
        }

        *record = SlotRecord {
            directory_index,
            last_rendered_frame: FrameIndex::default(),
            edit_locked:         false,
        };
        self.active += 1;

        Some(directory_index)
    }

    /// Release `material_slot` and clear its record.
    ///
    /// # Panics
    ///
    /// Debug builds panic if `material_slot >= capacity` or if the slot is
    /// already free — both are caller-side programming errors (the
    /// directory / residency layer is supposed to track ownership and
    /// should never double-free).
    pub fn free(&mut self, material_slot: u32) {
        debug_assert!(
            material_slot < self.capacity,
            "free: material_slot {material_slot} >= capacity {}",
            self.capacity,
        );
        debug_assert!(
            !self.records[material_slot as usize].is_free(),
            "free: material_slot {material_slot} is already free",
        );

        self.records[material_slot as usize] = SlotRecord::free();
        self.active = self.active.saturating_sub(1);
    }

    /// Bump `last_rendered_frame` for `material_slot`.
    ///
    /// The per-frame residency walk calls this once per slot it emits into
    /// the instance array, so the TTL eviction policy (later step) can pick
    /// a slot that has not rendered recently. A call on a free slot is a
    /// silent no-op — the per-frame walk derives its slot list from the
    /// directory, and by the time a slot is freed its directory entry has
    /// already been cleared, so the no-op only fires if callers wire the
    /// touch before the free (order-dependent bug) rather than indicating a
    /// real state this allocator needs to handle.
    pub fn touch(&mut self, material_slot: u32, frame: FrameIndex) {
        debug_assert!(
            material_slot < self.capacity,
            "touch: material_slot {material_slot} >= capacity {}",
            self.capacity,
        );
        let record = &mut self.records[material_slot as usize];
        if !record.is_free() {
            record.last_rendered_frame = frame;
        }
    }

    /// Flip the edit-lock bit on `material_slot`.
    ///
    /// Edit-locked slots are excluded from TTL eviction candidates in
    /// Step 3. No callers yet — the hook exists so Step 3's edit path can
    /// pin a slot while the player edits it.
    pub fn set_edit_locked(&mut self, material_slot: u32, locked: bool) {
        debug_assert!(
            material_slot < self.capacity,
            "set_edit_locked: material_slot {material_slot} >= capacity {}",
            self.capacity,
        );
        let record = &mut self.records[material_slot as usize];
        debug_assert!(
            !record.is_free(),
            "set_edit_locked: material_slot {material_slot} is free",
        );
        record.edit_locked = locked;
    }

    /// Snapshot of allocator state for the Step-7 shadow ledger.
    pub fn stats(&self) -> AllocatorStats {
        AllocatorStats {
            active:    self.active,
            evictions: self.evictions,
            refused:   self.refused,
            capacity:  self.capacity,
        }
    }

    /// The directory index currently owning `material_slot`, or `None` if
    /// the slot is free. Under the identity policy, `owner(slot)` always
    /// returns `Some(slot)` when allocated. Kept as a public getter for the
    /// Step-7 eviction path, which will need to know the outgoing
    /// directory entry before reusing a slot.
    pub fn owner(&self, material_slot: u32) -> Option<u32> {
        debug_assert!(
            material_slot < self.capacity,
            "owner: material_slot {material_slot} >= capacity {}",
            self.capacity,
        );
        let record = &self.records[material_slot as usize];
        if record.is_free() { None } else { Some(record.directory_index) }
    }
}

// --- SlotRecord ---

/// Per-slot allocator bookkeeping. Private — callers never inspect a
/// `SlotRecord`; the allocator exposes targeted getters for each field it
/// supports reading.
struct SlotRecord {
    /// Directory index that currently owns this material slot, or
    /// `DIRECTORY_INDEX_FREE` when the slot is unallocated. Under the
    /// identity policy this field always equals the slot's own index when
    /// allocated; it is stored explicitly so the Step-7 eviction path can
    /// diverge from the identity mapping without rewriting the bookkeeping.
    directory_index:     u32,
    last_rendered_frame: FrameIndex,
    edit_locked:         bool,
}

/// Sentinel `directory_index` stored in a free [`SlotRecord`]. Chosen as
/// `u32::MAX` so any collision with a real directory index is impossible —
/// the directory is sized far below that (MAX_CANDIDATES = 256 today).
const DIRECTORY_INDEX_FREE: u32 = u32::MAX;

impl SlotRecord {
    fn free() -> Self {
        Self {
            directory_index:     DIRECTORY_INDEX_FREE,
            last_rendered_frame: FrameIndex::default(),
            edit_locked:         false,
        }
    }

    const fn is_free(&self) -> bool {
        self.directory_index == DIRECTORY_INDEX_FREE
    }
}

// --- AllocatorStats ---

/// Compact snapshot of [`MaterialAllocator`] state. Consumed by the
/// Step-7 shadow-ledger readback; unused today.
#[derive(Clone, Copy, Debug)]
pub struct AllocatorStats {
    pub active:    u32,
    pub evictions: u64,
    pub refused:   u64,
    pub capacity:  u32,
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocator_has_all_slots_free() {
        let a = MaterialAllocator::new(4);
        assert_eq!(a.capacity(),   4);
        assert_eq!(a.active(),     0);
        assert_eq!(a.free_count(), 4);
    }

    /// Load-bearing: the renderer's VS/PS read `material_pool[directory_index]`,
    /// so the allocator MUST return `directory_index` verbatim. If this test
    /// fails, the rendered frame cross-wires material data between sub-chunks.
    #[test]
    fn allocate_returns_directory_index_verbatim() {
        let mut a = MaterialAllocator::new(8);
        // Out-of-order allocations must each yield their own directory_index.
        assert_eq!(a.allocate(3), Some(3));
        assert_eq!(a.allocate(0), Some(0));
        assert_eq!(a.allocate(7), Some(7));
        assert_eq!(a.allocate(5), Some(5));
        // Freeing a slot and reallocating a different directory_index must
        // still return that directory_index, not the just-freed slot.
        a.free(0);
        assert_eq!(a.allocate(2), Some(2));
        // And reallocating the freed slot's directory_index returns it.
        assert_eq!(a.allocate(0), Some(0));
    }

    #[test]
    fn allocate_then_free_roundtrips() {
        let mut a = MaterialAllocator::new(4);
        let s = a.allocate(2).unwrap();
        assert_eq!(s, 2, "identity policy: allocate(2) -> 2");
        assert_eq!(a.active(), 1);
        assert_eq!(a.owner(s), Some(2));

        a.free(s);
        assert_eq!(a.active(), 0);
        assert_eq!(a.owner(s), None);
    }

    #[test]
    fn allocate_same_directory_index_without_free_returns_none() {
        // Under the identity policy, each directory_index owns exactly one
        // slot. Attempting to allocate the same directory_index twice
        // without an intervening free is a caller-side bug and must not
        // silently hand out a different slot (that would re-introduce the
        // cross-wiring this allocator exists to prevent).
        let mut a = MaterialAllocator::new(4);
        assert_eq!(a.allocate(2), Some(2));
        assert_eq!(a.allocate(2), None);
        assert_eq!(a.stats().refused, 1);
        assert_eq!(a.active(), 1, "refused allocation must not bump active");
    }

    #[test]
    fn freed_slot_becomes_reallocatable() {
        let mut a = MaterialAllocator::new(4);
        let s = a.allocate(3).unwrap();
        assert_eq!(s, 3);
        a.free(s);
        assert_eq!(a.allocate(3), Some(3));
    }

    #[test]
    fn touch_is_noop_for_free_slot() {
        let mut a = MaterialAllocator::new(2);
        // Slot 0 is free — touching must not panic or change state.
        a.touch(0, FrameIndex::default());
    }

    #[test]
    fn owner_tracks_directory_index_across_free_and_realloc() {
        let mut a = MaterialAllocator::new(4);
        let s = a.allocate(2).unwrap();
        assert_eq!(a.owner(s), Some(2));
        a.free(s);
        assert_eq!(a.owner(s), None);
        let s2 = a.allocate(2).unwrap();
        assert_eq!(s2, s);
        assert_eq!(a.owner(s2), Some(2));
    }

    #[test]
    fn filling_the_pool_yields_sequential_identity_allocations() {
        let mut a = MaterialAllocator::new(4);
        for dir_idx in 0..4 {
            assert_eq!(a.allocate(dir_idx), Some(dir_idx));
        }
        assert_eq!(a.active(), 4);
        assert_eq!(a.free_count(), 0);
    }
}
