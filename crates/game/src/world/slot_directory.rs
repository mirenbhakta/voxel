//! CPU-authoritative sub-chunk directory.
//!
//! [`SlotDirectory`] is the single CPU-side owner of the `DirEntry` array
//! the GPU consumes through `WorldRenderer::slot_directory_buf`. Per the
//! "no CPU mirrors of GPU state" principle, this is *not* a cache of GPU
//! contents — it is the source. The GPU reads it; the GPU never writes
//! back into it.
//!
//! # Dirty tracking
//!
//! Writes set a per-entry dirty bit. [`SlotDirectory::drain_dirty`]
//! walks the dirty set, producing the exact `(index, DirEntry)` tuples
//! that need to flow to `WorldRenderer::write_directory_entries` this
//! frame, and clears the bits as it goes. A single entry written N times
//! in the same frame flushes once.
//!
//! # Initial state
//!
//! All entries start as [`DirEntry::empty([0,0,0])`] (the canonical
//! pre-seeded state) and the dirty set starts *full*. The caller
//! (e.g. `WorldView::new`) immediately overwrites every slot with
//! `DirEntry::empty(canonical_coord)` via the seed pass — those writes land
//! in the dirty set and are flushed to the GPU on the first frame. The
//! initial `empty([0,0,0])` bit pattern is never GPU-observable because the
//! seed pass always fires before the first drain.

#![allow(dead_code)]

use renderer::DirEntry;

// --- SlotDirectory ---

/// CPU-authored directory, parallel to the GPU `slot_directory_buf`.
///
/// Indexed by `directory_index` (= `level_offset + pool_slot`). The
/// backing store is a `Vec<DirEntry>` for direct random access; dirty
/// flags are tracked as a bit-packed `Vec<u64>` so a single-instruction
/// `BMI1` TZCNT scan can pull out the next dirty entry.
pub struct SlotDirectory {
    entries: Vec<DirEntry>,
    dirty:   Vec<u64>,

    /// Running count of set bits in `dirty`. Maintained incrementally by
    /// `set` / `drain_dirty` so the caller can skip the drain when the
    /// directory is clean, without scanning the bitmap.
    dirty_count: u32,
}

impl SlotDirectory {
    /// Construct a directory sized to `capacity` entries, all initialised
    /// to [`DirEntry::empty([0,0,0])`] (the pre-seed placeholder). Every
    /// entry is flagged dirty so the caller's seed pass (which writes
    /// `DirEntry::empty(canonical_coord)` for every slot) flows through the
    /// dirty set and reaches the GPU on the first flush. The initial
    /// `empty([0,0,0])` bit pattern is never GPU-observable: the seed pass
    /// always runs before the first drain.
    pub fn new(capacity: u32) -> Self {
        let capacity_usize = capacity as usize;
        let entries = vec![DirEntry::empty([0, 0, 0]); capacity_usize];

        // One bit per entry, rounded up to a multiple of 64.
        let words = capacity_usize.div_ceil(64);
        let mut dirty = vec![0u64; words];
        // Mark every bit in `[0, capacity)` as dirty.
        for (w, word) in dirty.iter_mut().enumerate() {
            let base = (w * 64) as u32;
            if base + 64 <= capacity {
                *word = u64::MAX;
            }
            else {
                let bits = capacity - base;
                *word = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            }
        }

        Self {
            entries,
            dirty,
            dirty_count: capacity,
        }
    }

    /// Total entry capacity.
    pub fn capacity(&self) -> u32 {
        self.entries.len() as u32
    }

    /// Number of currently-dirty entries.
    pub fn dirty_count(&self) -> u32 {
        self.dirty_count
    }

    /// Read-only access to the entry at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.capacity()`.
    pub fn get(&self, index: u32) -> &DirEntry {
        &self.entries[index as usize]
    }

    /// Borrow the full entry array.
    ///
    /// Used by the `debug-state-history` snapshot path to clone the
    /// CPU-authoritative directory state into a per-frame record; the slice
    /// is also convenient for tests that want to read out the post-batch
    /// directory without knowing individual indices. The slice length equals
    /// [`Self::capacity`].
    pub fn entries_view(&self) -> &[DirEntry] {
        &self.entries
    }

    /// Write `entry` at `index`, flagging it dirty.
    ///
    /// Writing the same entry twice before a drain flushes once — the
    /// dirty bit is idempotent. Setting a value that equals the current
    /// entry is not elided; the caller is the one best positioned to
    /// decide whether to skip a redundant write (the allocator's
    /// allocate / free pattern always writes different entries).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.capacity()`.
    pub fn set(&mut self, index: u32, entry: DirEntry) {
        let idx = index as usize;
        self.entries[idx] = entry;

        let word_idx = idx / 64;
        let bit      = 1u64 << (idx % 64);
        if (self.dirty[word_idx] & bit) == 0 {
            self.dirty[word_idx] |= bit;
            self.dirty_count += 1;
        }
    }

    /// Walk the dirty set, yielding `(index, entry)` for every dirty slot
    /// and clearing the dirty bits as it goes. After the iterator is
    /// exhausted the directory is in a fully-clean state.
    ///
    /// Non-lazy — constructs a `Vec` so the iterator does not borrow the
    /// directory while the caller mutates it. The `Vec` capacity tracks
    /// `dirty_count`, so the allocation is zero bytes when nothing is
    /// dirty.
    pub fn drain_dirty(&mut self) -> impl Iterator<Item = (u32, DirEntry)> {
        let mut out = Vec::with_capacity(self.dirty_count as usize);
        let capacity = self.entries.len() as u32;

        for (w, word) in self.dirty.iter_mut().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let bit_idx = bits.trailing_zeros();
                let entry_idx = (w as u32) * 64 + bit_idx;
                // Guard against trailing bits past the logical capacity —
                // they cannot be set by `set` but a hypothetical `new`
                // bug would otherwise leak non-existent entries.
                if entry_idx < capacity {
                    out.push((entry_idx, self.entries[entry_idx as usize]));
                }
                bits &= bits - 1;
            }
            *word = 0;
        }

        self.dirty_count = 0;
        out.into_iter()
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_marks_every_entry_dirty() {
        let mut d = SlotDirectory::new(3);
        assert_eq!(d.capacity(),   3);
        assert_eq!(d.dirty_count(), 3);
        let drained: Vec<(u32, DirEntry)> = d.drain_dirty().collect();
        assert_eq!(drained.len(), 3);
        // All initial entries are empty.
        for (_, entry) in &drained {
            assert!(!entry.is_resident());
        }
        assert_eq!(d.dirty_count(), 0);
    }

    #[test]
    fn set_then_drain_reports_changed_entry() {
        let mut d = SlotDirectory::new(4);
        let _ = d.drain_dirty().count(); // clear the initial dirty set

        let e = DirEntry::resident([1, 2, 3], 0x3F, false, 5);
        d.set(2, e);

        let drained: Vec<_> = d.drain_dirty().collect();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, 2);
        assert_eq!(drained[0].1.coord, [1, 2, 3]);
        assert!(drained[0].1.is_resident());
        assert_eq!(d.dirty_count(), 0);
    }

    #[test]
    fn set_reports_dirty_exactly_once_after_multiple_writes() {
        let mut d = SlotDirectory::new(4);
        let _ = d.drain_dirty().count();

        d.set(1, DirEntry::empty([0, 0, 0]));
        d.set(1, DirEntry::resident([0, 0, 0], 1, false, 0));
        d.set(1, DirEntry::resident([0, 0, 0], 2, false, 0));

        let drained: Vec<_> = d.drain_dirty().collect();
        assert_eq!(drained.len(), 1, "three writes to slot 1 should flush once");
        assert_eq!(drained[0].0, 1);
        assert_eq!(drained[0].1.exposure(), 2, "last write wins");
    }

    #[test]
    fn drain_dirty_clears_the_bitset() {
        let mut d = SlotDirectory::new(4);
        let _ = d.drain_dirty().count();

        d.set(0, DirEntry::empty([0, 0, 0]));
        d.set(3, DirEntry::empty([0, 0, 0]));

        let first: Vec<_> = d.drain_dirty().collect();
        assert_eq!(first.len(), 2);

        // Nothing should be dirty after a drain.
        let second: Vec<_> = d.drain_dirty().collect();
        assert_eq!(second.len(), 0);
    }

    #[test]
    fn drain_dirty_reports_all_set_indices_in_order() {
        let mut d = SlotDirectory::new(130);
        let _ = d.drain_dirty().count();

        // Hit indices spanning two full dirty words + the tail.
        let indices = [0u32, 1, 63, 64, 65, 127, 128, 129];
        for &i in &indices {
            d.set(i, DirEntry::empty([0, 0, 0]));
        }

        let drained: Vec<u32> = d.drain_dirty().map(|(i, _)| i).collect();
        assert_eq!(drained, indices);
    }

    #[test]
    fn get_reflects_latest_write() {
        let mut d = SlotDirectory::new(4);
        let e = DirEntry::resident([0, 0, 0], 0x3F, true, 7);
        d.set(1, e);
        assert_eq!(d.get(1).material_slot(), 7);
        assert!(d.get(1).is_resident());
        assert!(d.get(1).is_solid());
    }
}
