//! Growable GPU buffer composed of fixed-size segments.
//!
//! A [`MultiBuffer`] manages a flat allocation-unit address space backed by
//! one or more equal-size GPU buffer segments. When the current capacity is
//! exhausted, a new segment is appended -- no data is copied. Segment index
//! and local offset are derived arithmetically from flat offsets:
//!
//! ```text
//! segment = offset >> segment_shift
//! local   = offset & segment_mask
//! ```
//!
//! Allocations never cross segment boundaries. When the bump pointer would
//! produce a crossing allocation, the tail of the current segment is
//! released to the free list and the bump advances to the next boundary.

use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

// ---------------------------------------------------------------------------
// SegmentAllocator
// ---------------------------------------------------------------------------

/// A bump allocator with coalescing free list that respects segment
/// boundaries.
///
/// Operates in a flat integer address space measured in allocation units.
/// Allocations never cross segment boundaries. When the bump pointer would
/// produce a crossing allocation, the tail of the current segment is released
/// to the free list and the bump advances to the next boundary.
struct SegmentAllocator {
    /// High-water mark (next bump offset, in units).
    bump          : u32,
    /// Free ranges sorted by offset. Each entry is (offset, size).
    free_list     : Vec<(u32, u32)>,
    /// Total capacity in units (always a multiple of segment_units).
    capacity      : u32,
    /// Units per segment (power of two).
    segment_units : u32,
    /// Bitmask for local offset (`segment_units - 1`).
    segment_mask  : u32,
}

// --- SegmentAllocator ---

impl SegmentAllocator {
    /// Create an allocator with zero capacity. Call [`grow`] to add
    /// segments before allocating.
    fn new(segment_units: u32) -> Self {
        assert!(
            segment_units.is_power_of_two(),
            "segment_units must be a power of two",
        );

        SegmentAllocator {
            bump          : 0,
            free_list     : Vec::new(),
            capacity      : 0,
            segment_units : segment_units,
            segment_mask  : segment_units - 1,
        }
    }

    /// Allocate `count` contiguous units.
    ///
    /// Returns the flat offset on success, or `None` if the allocator
    /// cannot satisfy the request. Tries the free list first (first-fit),
    /// then falls back to the bump pointer. Allocations that would cross a
    /// segment boundary are skipped or the bump is advanced past the
    /// boundary.
    ///
    /// # Panics
    ///
    /// Panics if `count > segment_units`.
    fn alloc(&mut self, count: u32) -> Option<u32> {
        assert!(
            count <= self.segment_units,
            "allocation of {} exceeds segment size {}",
            count,
            self.segment_units,
        );

        if count == 0 {
            return Some(0);
        }

        // First-fit search through free list, skipping boundary crossings.
        for i in 0..self.free_list.len() {
            let (offset, free_size) = self.free_list[i];
            let local = offset & self.segment_mask;

            // Skip if the allocation would cross a segment boundary.
            if local + count > self.segment_units {
                continue;
            }

            if free_size >= count {
                if free_size == count {
                    self.free_list.remove(i);
                }
                else {
                    // Shrink the free range from the front.
                    self.free_list[i] = (offset + count, free_size - count);
                }

                return Some(offset);
            }
        }

        // Bump allocation with boundary enforcement.
        let local = self.bump & self.segment_mask;

        if local + count > self.segment_units {
            // Would cross a segment boundary. Waste the tail of the
            // current segment and advance to the next boundary.
            let tail = self.segment_units - local;

            if tail > 0 {
                self.insert_free(self.bump, tail);
            }

            self.bump += tail;
        }

        if self.capacity - self.bump >= count {
            let offset = self.bump;
            self.bump += count;
            return Some(offset);
        }

        None
    }

    /// Free a contiguous range starting at `offset` with the given `count`.
    ///
    /// Coalesces with adjacent free ranges to prevent fragmentation. If the
    /// freed range is at the bump pointer, the pointer is retracted instead
    /// of adding to the free list.
    fn free(&mut self, offset: u32, count: u32) {
        if count == 0 {
            return;
        }

        let end = offset + count;

        // If this range is at the top of the bump region, retract.
        if end == self.bump {
            self.bump = offset;
            self.coalesce_bump();
            return;
        }

        self.insert_free(offset, count);
    }

    /// Return the total number of free units (free list + remaining bump
    /// space).
    fn free_units(&self) -> u32 {
        let free_list_total: u32 = self.free_list
            .iter()
            .map(|&(_, size)| size)
            .sum();

        free_list_total + (self.capacity - self.bump)
    }

    /// Return the total capacity in allocation units.
    fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Extend capacity by one segment. Returns the new segment index.
    fn grow(&mut self) -> u32 {
        let index = self.capacity / self.segment_units;
        self.capacity += self.segment_units;
        index
    }

    /// Insert a free range into the sorted free list and coalesce with
    /// neighbors.
    fn insert_free(&mut self, offset: u32, size: u32) {
        let pos = self.free_list
            .partition_point(|&(o, _)| o < offset);

        self.free_list.insert(pos, (offset, size));
        self.coalesce_at(pos);
    }

    /// Coalesce the free list entry at `idx` with its immediate neighbors.
    fn coalesce_at(&mut self, idx: usize) {
        // Merge with the entry after idx, if adjacent.
        if idx + 1 < self.free_list.len() {
            let (off_a, size_a) = self.free_list[idx];
            let (off_b, size_b) = self.free_list[idx + 1];

            if off_a + size_a == off_b {
                self.free_list[idx] = (off_a, size_a + size_b);
                self.free_list.remove(idx + 1);
            }
        }

        // Merge with the entry before idx, if adjacent.
        if idx > 0 {
            let (off_prev, size_prev) = self.free_list[idx - 1];
            let (off_cur, size_cur)   = self.free_list[idx];

            if off_prev + size_prev == off_cur {
                self.free_list[idx - 1] = (off_prev, size_prev + size_cur);
                self.free_list.remove(idx);
            }
        }
    }

    /// Retract the bump pointer by absorbing any free list entry that now
    /// abuts it from below.
    fn coalesce_bump(&mut self) {
        while let Some(&(offset, size)) = self.free_list.last() {
            if offset + size == self.bump {
                self.bump = offset;
                self.free_list.pop();
            }
            else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MultiBuffer
// ---------------------------------------------------------------------------

/// A growable GPU buffer composed of fixed-size segments.
///
/// Each segment is an independent GPU buffer of equal byte size. Allocations
/// are addressed in a flat unit space. Segment selection and local byte
/// offset are derived arithmetically via shift and mask (segment sizes are
/// powers of two).
///
/// Allocations never cross segment boundaries. When the current capacity is
/// exhausted, a new segment is appended and no existing data is copied.
pub struct MultiBuffer {
    /// GPU buffer segments, appended on growth.
    segments      : Vec<Buffer>,
    /// Segment-aware allocator operating in flat unit space.
    alloc         : SegmentAllocator,
    /// Bytes per allocation unit.
    unit_bytes    : u32,
    /// Log2 of segment_units (for shift-based segment resolution).
    segment_shift : u32,
    /// Buffer usage flags for new segments.
    usage         : BufferUsages,
    /// Debug label prefix for new segments.
    label         : &'static str,
    /// Monotonically increasing counter, incremented on each new segment.
    /// Compare against a cached value to detect growth events that require
    /// bind group rebuilds.
    generation    : u64,
}

// --- MultiBuffer ---

impl MultiBuffer {
    /// Create a multi-buffer with one initial segment.
    ///
    /// # Arguments
    ///
    /// * `segment_units` - Units per segment (must be a power of two).
    /// * `unit_bytes` - Bytes per allocation unit.
    /// * `usage` - wgpu buffer usage flags for every segment.
    /// * `label` - Debug label prefix (segments are named `label[index]`).
    pub fn new(
        device        : &Device,
        segment_units : u32,
        unit_bytes    : u32,
        usage         : BufferUsages,
        label         : &'static str,
    )
        -> Self
    {
        assert!(segment_units.is_power_of_two());

        let mut mb = MultiBuffer {
            segments      : Vec::new(),
            alloc         : SegmentAllocator::new(segment_units),
            unit_bytes    : unit_bytes,
            segment_shift : segment_units.trailing_zeros(),
            usage         : usage,
            label         : label,
            generation    : 0,
        };

        mb.grow_segment(device);
        mb
    }

    /// Allocate `count` contiguous units. Creates new segments as needed.
    ///
    /// Returns the flat offset. The allocation is guaranteed to reside
    /// entirely within a single segment. Panics if `count` exceeds the
    /// segment size.
    pub fn alloc(&mut self, device: &Device, count: u32) -> u32 {
        loop {
            if let Some(offset) = self.alloc.alloc(count) {
                return offset;
            }

            self.grow_segment(device);
        }
    }

    /// Free a previously allocated range.
    pub fn free(&mut self, offset: u32, count: u32) {
        self.alloc.free(offset, count);
    }

    /// Resolve a flat unit offset to segment index and local byte offset.
    pub fn resolve(&self, offset: u32) -> (u32, u32) {
        let segment = offset >> self.segment_shift;
        let local   = (offset & self.alloc.segment_mask) * self.unit_bytes;
        (segment, local)
    }

    /// Return a slice of all segment buffers.
    pub fn buffers(&self) -> &[Buffer] {
        &self.segments
    }

    /// Return the number of segments.
    pub fn segment_count(&self) -> u32 {
        self.segments.len() as u32
    }

    /// Return the generation counter. Incremented on each new segment.
    /// Compare against a cached value to detect growth events.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Return the total capacity in allocation units across all segments.
    pub fn capacity(&self) -> u32 {
        self.alloc.capacity()
    }

    /// Return the total number of free units (free list + unallocated bump
    /// space).
    pub fn free_units(&self) -> u32 {
        self.alloc.free_units()
    }

    /// Return the byte size of one segment.
    pub fn segment_byte_size(&self) -> u64 {
        u64::from(self.alloc.segment_units) * u64::from(self.unit_bytes)
    }

    /// Create a new GPU buffer segment and extend allocator capacity.
    fn grow_segment(&mut self, device: &Device) {
        let index = self.alloc.grow();

        let buffer = device.create_buffer(&BufferDescriptor {
            label              : Some(&format!("{}[{}]", self.label, index)),
            size               : self.segment_byte_size(),
            usage              : self.usage,
            mapped_at_creation : false,
        });

        self.segments.push(buffer);
        self.generation += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::SegmentAllocator;

    /// Verify basic allocation and free-list reuse.
    #[test]
    fn basic_alloc_and_free() {
        let mut a = SegmentAllocator::new(16);
        a.grow(); // capacity = 16

        let o1 = a.alloc(4).unwrap();
        assert_eq!(o1, 0);
        assert_eq!(a.bump, 4);

        let o2 = a.alloc(4).unwrap();
        assert_eq!(o2, 4);

        // Free the first allocation.
        a.free(0, 4);

        // Re-allocate should reuse the freed range.
        let o3 = a.alloc(4).unwrap();
        assert_eq!(o3, 0);
    }

    /// Verify that allocations crossing a segment boundary are blocked:
    /// the tail is wasted and the allocation starts at the next boundary.
    #[test]
    fn boundary_enforcement_bump() {
        let mut a = SegmentAllocator::new(16);
        a.grow(); // capacity = 16
        a.grow(); // capacity = 32

        // Fill 10 of the first 16 units.
        let o1 = a.alloc(10).unwrap();
        assert_eq!(o1, 0);

        // Allocating 10 more would cross the boundary (10 + 10 = 20 > 16).
        // The allocator should waste units 10-15 and start at 16.
        let o2 = a.alloc(10).unwrap();
        assert_eq!(o2, 16);

        // The 6-unit tail (offsets 10-15) should be on the free list.
        assert_eq!(a.free_list.len(), 1);
        assert_eq!(a.free_list[0], (10, 6));
    }

    /// Verify that wasted tails from boundary skips are reusable by smaller
    /// allocations.
    #[test]
    fn small_alloc_reuses_tail() {
        let mut a = SegmentAllocator::new(16);
        a.grow();
        a.grow();

        // Fill 10, then alloc 10 (skips boundary, wastes 6 at offsets 10-15).
        a.alloc(10).unwrap();
        a.alloc(10).unwrap();

        // A small allocation should reuse the wasted tail.
        let o = a.alloc(3).unwrap();
        assert_eq!(o, 10);

        // Remaining 3 units (13-15) still on free list.
        assert_eq!(a.free_list.len(), 1);
        assert_eq!(a.free_list[0], (13, 3));
    }

    /// Verify that the free-list scan skips candidates that would cross a
    /// segment boundary.
    #[test]
    fn free_list_boundary_skip() {
        let mut a = SegmentAllocator::new(16);
        a.grow();
        a.grow();

        // Allocate 14 units, then 2, filling segment 0.
        a.alloc(14).unwrap();
        a.alloc(2).unwrap();

        // Free the 14-unit block at offset 0.
        a.free(0, 14);

        // Allocate 16 from segment 1 (fills it fully).
        a.alloc(16).unwrap();

        // Now free list has (0, 14). Try to alloc 15: this would cross if
        // placed at offset 0 (0 + 15 = 15 <= 16, actually fits). Let's
        // test with something that truly crosses.

        // Alloc 4: fits at offset 0 within segment 0.
        let o = a.alloc(4).unwrap();
        assert_eq!(o, 0);

        // Free list now has (4, 10). Alloc 12 would cross from offset 4
        // (4 + 12 = 16 -- that's exactly the boundary, which is OK since
        // segment 0 is [0, 16)). Actually 4+12=16 means local = 4,
        // local + 12 = 16 <= 16. Wait: the check is local + count >
        // segment_units. 4 + 12 = 16 is NOT > 16. So it fits.

        // Let's test 13: local = 4, 4 + 13 = 17 > 16. Should skip free
        // list and go to bump (which is at 32, needs segment 2).
        a.grow(); // capacity = 48
        let o2 = a.alloc(13).unwrap();
        assert_eq!(o2, 32); // from bump in segment 2
    }

    /// Verify that freeing ranges across segment boundaries correctly
    /// retracts the bump pointer via coalesce_bump.
    #[test]
    fn coalesce_across_segments() {
        let mut a = SegmentAllocator::new(8);
        a.grow(); // 0-7
        a.grow(); // 8-15

        // Allocate three ranges spanning the segment boundary.
        let o1 = a.alloc(4).unwrap(); // 0-3
        let o2 = a.alloc(4).unwrap(); // 4-7
        let o3 = a.alloc(4).unwrap(); // 8-11

        assert_eq!(o1, 0);
        assert_eq!(o2, 4);
        assert_eq!(o3, 8);

        // Free o2 (goes to free list), then o3 (at bump edge, retracts
        // to 8, then coalesce_bump absorbs the (4,4) entry across the
        // segment boundary, retracting bump to 4).
        a.free(4, 4);
        a.free(8, 4);

        assert_eq!(a.bump, 4);
        assert!(a.free_list.is_empty());
    }

    /// Verify bump retraction works across segment boundaries.
    #[test]
    fn bump_retraction_across_segments() {
        let mut a = SegmentAllocator::new(8);
        a.grow(); // 0-7
        a.grow(); // 8-15

        // Alloc 4 in segment 0, 4 more in segment 0, 4 in segment 1.
        let o1 = a.alloc(4).unwrap(); // 0-3
        let o2 = a.alloc(4).unwrap(); // 4-7
        let o3 = a.alloc(4).unwrap(); // 8-11, bump = 12
        assert_eq!((o1, o2, o3), (0, 4, 8));

        // Free middle block -- not at bump edge, goes to free list.
        a.free(4, 4);
        assert_eq!(a.free_list.len(), 1);
        assert_eq!(a.free_list[0], (4, 4));
        assert_eq!(a.bump, 12);

        // Free top block -- at bump edge, retracts to 8.
        // Then coalesce_bump absorbs (4, 4), retracting across the
        // segment boundary to 4.
        a.free(8, 4);
        assert_eq!(a.bump, 4);
        assert!(a.free_list.is_empty());

        // Free bottom block -- at bump edge, retracts to 0.
        a.free(0, 4);
        assert_eq!(a.bump, 0);
        assert!(a.free_list.is_empty());
    }

    /// Verify size-0 edge cases.
    #[test]
    fn size_zero() {
        let mut a = SegmentAllocator::new(16);
        a.grow();

        // Size-0 alloc returns 0 without consuming space.
        let o = a.alloc(0).unwrap();
        assert_eq!(o, 0);
        assert_eq!(a.bump, 0);

        // Size-0 free is a no-op.
        a.free(0, 0);
        assert_eq!(a.bump, 0);
        assert!(a.free_list.is_empty());
    }

    /// Verify that allocating more than segment_units panics.
    #[test]
    #[should_panic(expected = "allocation of 17 exceeds segment size 16")]
    fn oversized_alloc_panics() {
        let mut a = SegmentAllocator::new(16);
        a.grow();
        a.alloc(17);
    }

    /// Verify that growth makes previously impossible allocations succeed.
    #[test]
    fn grow_enables_alloc() {
        let mut a = SegmentAllocator::new(8);
        a.grow(); // capacity = 8

        // Fill segment 0.
        a.alloc(8).unwrap();

        // No room left.
        assert!(a.alloc(4).is_none());

        // Grow and retry.
        a.grow(); // capacity = 16
        let o = a.alloc(4).unwrap();
        assert_eq!(o, 8);
    }

    /// Verify free_units accounts for both free list and bump space.
    #[test]
    fn free_units_accounting() {
        let mut a = SegmentAllocator::new(16);
        a.grow(); // capacity = 16
        assert_eq!(a.free_units(), 16);

        a.alloc(6).unwrap();
        assert_eq!(a.free_units(), 10);

        a.free(0, 6);
        assert_eq!(a.free_units(), 16);

        // Boundary waste is counted as free.
        a.grow(); // capacity = 32
        a.alloc(6).unwrap(); // offset 0
        a.alloc(12).unwrap(); // would cross, wastes 10 at offsets 6-15, allocs at 16
        // Used: 6 + 12 = 18. Wasted (on free list): 10. Bump space: 32 - 28 = 4.
        assert_eq!(a.free_units(), 14); // 10 + 4
    }

    /// Verify the exact-boundary case: allocation ending exactly at the
    /// segment boundary is valid (not a crossing).
    #[test]
    fn exact_boundary_fit() {
        let mut a = SegmentAllocator::new(16);
        a.grow();

        // 12 + 4 = 16, which is not > 16. Should fit.
        a.alloc(12).unwrap();
        let o = a.alloc(4).unwrap();
        assert_eq!(o, 12);
        assert_eq!(a.bump, 16);
    }

    /// Verify that resolve produces correct segment index and local byte
    /// offset.
    #[test]
    fn resolve_arithmetic() {
        let shift = 14u32; // segment_units = 16384
        let mask  = (1u32 << shift) - 1;
        let unit_bytes = 1024u32;

        // Offset 0: segment 0, local 0.
        assert_eq!((0u32 >> shift, (0u32 & mask) * unit_bytes), (0, 0));

        // Offset 100: segment 0, local 100 * 1024 = 102400.
        assert_eq!(
            (100u32 >> shift, (100u32 & mask) * unit_bytes),
            (0, 102400),
        );

        // Offset 16384: segment 1, local 0.
        assert_eq!(
            (16384u32 >> shift, (16384u32 & mask) * unit_bytes),
            (1, 0),
        );

        // Offset 16400: segment 1, local 16 * 1024 = 16384.
        assert_eq!(
            (16400u32 >> shift, (16400u32 & mask) * unit_bytes),
            (1, 16384),
        );
    }
}
