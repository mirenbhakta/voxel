//! GPU-side bump + free-list allocator.
//!
//! Manages a contiguous range of allocation units. The GPU owns the bump
//! pointer and scans a free list seeded by the CPU before each dispatch.
//! The CPU tracks freed ranges persistently and mirrors the GPU's
//! consumption via [`consume`] after each build feedback readback.
//!
//! [`consume`]: GpuAllocator::consume

use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

/// A GPU-side bump allocator backed by a first-fit free list.
///
/// The GPU maintains a persistent bump pointer in [`bump_buf`] and a
/// free list in [`free_list_buf`]. The CPU keeps a persistent mirror of
/// the free list in [`free_list`]. Each dispatch uploads the full list
/// to the GPU. After the alloc pass, [`consume`] removes entries that
/// the GPU used (identified by the readback `base_offset`). Freed
/// ranges are added via [`push_free`] and survive across frames until
/// the GPU consumes them.
///
/// [`bump_buf`]:      Self::bump_buf
/// [`free_list_buf`]: Self::free_list_buf
/// [`free_list`]:     Self::free_list
/// [`push_free`]:     Self::push_free
/// [`consume`]:       Self::consume
pub(crate) struct GpuAllocator {
    /// GPU-side persistent bump pointer (single u32).
    bump_buf      : Buffer,
    /// GPU-visible free list buffer.
    /// Layout: `count(4)` followed by `(offset: u32, size: u32)` pairs.
    free_list_buf : Buffer,
    /// CPU-side persistent free list. Flat: `[off0, sz0, off1, sz1, ...]`.
    ///
    /// Entries accumulate via [`push_free`] and are removed by
    /// [`consume`] when the GPU alloc pass uses them. The full list is
    /// uploaded to the GPU before each dispatch.
    ///
    /// [`push_free`]: Self::push_free
    /// [`consume`]:   Self::consume
    free_list     : Vec<u32>,
    /// Maximum free list entries (caps upload to GPU buffer capacity).
    max_entries   : u32,
}

impl GpuAllocator {
    /// Create a new allocator with the given label prefix and capacity.
    ///
    /// Allocates two GPU buffers: a single-u32 bump pointer and a free
    /// list sized for `max_entries` `(offset, size)` pairs plus a
    /// leading count word.
    pub fn new(device: &Device, label: &str, max_entries: u32) -> Self {
        let bump_buf = device.create_buffer(&BufferDescriptor {
            label              : Some(&format!("{label}_bump")),
            size               : 4,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        let free_list_bytes = 4 + u64::from(max_entries) * 8;

        let free_list_buf = device.create_buffer(&BufferDescriptor {
            label              : Some(&format!("{label}_free_list")),
            size               : free_list_bytes,
            usage              : BufferUsages::STORAGE
                               | BufferUsages::COPY_DST,
            mapped_at_creation : false,
        });

        Self {
            bump_buf      : bump_buf,
            free_list_buf : free_list_buf,
            free_list     : Vec::new(),
            max_entries   : max_entries,
        }
    }

    /// Record a freed range.
    ///
    /// `offset` and `size` are in allocation units (quads, sub-blocks,
    /// etc.) and are opaque to this struct. The entry persists until
    /// consumed by the GPU (via [`consume`]) or explicitly removed.
    ///
    /// [`consume`]: Self::consume
    pub fn push_free(&mut self, offset: u32, size: u32) {
        self.free_list.push(offset);
        self.free_list.push(size);
    }

    /// Remove a free list entry that the GPU consumed.
    ///
    /// The GPU alloc shader uses first-fit and returns the matched
    /// entry's offset as `base_offset` in the staging readback. This
    /// method finds the entry starting at `offset`, then shrinks it
    /// (if larger than `count`) or swap-removes it (if exact fit),
    /// mirroring the GPU's in-place modifications.
    ///
    /// Returns `true` if a matching entry was found (free list
    /// allocation), `false` otherwise (bump allocation).
    pub fn consume(&mut self, offset: u32, count: u32) -> bool {
        let entries = self.free_list.len() / 2;

        for i in 0..entries {
            let fo = self.free_list[i * 2];
            let fs = self.free_list[i * 2 + 1];

            if fo == offset && fs >= count {
                if fs > count {
                    // Shrink: advance offset, reduce size.
                    self.free_list[i * 2]     = fo + count;
                    self.free_list[i * 2 + 1] = fs - count;
                }
                else {
                    // Exact fit: swap-remove (mirrors GPU shader).
                    let last = entries - 1;

                    if i != last {
                        self.free_list[i * 2]     = self.free_list[last * 2];
                        self.free_list[i * 2 + 1] = self.free_list[last * 2 + 1];
                    }

                    self.free_list.truncate(last * 2);
                }

                return true;
            }
        }

        false
    }

    /// Upload the full free list to the GPU.
    ///
    /// Writes a leading `count` word followed by `(offset, size)` pairs.
    /// Entries beyond [`max_entries`] remain in the CPU list and will be
    /// uploaded once earlier entries are consumed by the GPU.
    ///
    /// Unlike the previous implementation, this does **not** clear the
    /// CPU list. Entries persist until explicitly removed by [`consume`].
    ///
    /// [`max_entries`]: Self::max_entries
    /// [`consume`]:     Self::consume
    pub fn upload(&mut self, queue: &Queue) {
        let total   = self.free_list.len() / 2;
        let capped  = total.min(self.max_entries as usize);
        let entry_count = capped as u32;

        queue.write_buffer(
            &self.free_list_buf,
            0,
            bytemuck::bytes_of(&entry_count),
        );

        if entry_count > 0 {
            let word_count = capped * 2;

            queue.write_buffer(
                &self.free_list_buf,
                4,
                bytemuck::cast_slice(&self.free_list[..word_count]),
            );
        }
    }

    /// Number of entries currently in the CPU free list.
    pub fn entry_count(&self) -> usize {
        self.free_list.len() / 2
    }

    /// The GPU buffer holding the persistent bump pointer.
    pub fn bump_buf(&self) -> &Buffer {
        &self.bump_buf
    }

    /// The GPU buffer holding the free list.
    pub fn free_list_buf(&self) -> &Buffer {
        &self.free_list_buf
    }
}
