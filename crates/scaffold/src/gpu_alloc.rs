//! GPU-side bump + free-list buffer holder.
//!
//! The GPU is the sole authority on allocation state. The CPU only
//! provides the buffers; the free list and bump pointer are managed
//! entirely by the alloc shader.

use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

/// GPU-side allocation buffers (bump pointer + free list).
///
/// The GPU maintains a persistent bump pointer in [`bump_buf`] and a
/// free list in [`free_list_buf`]. Both are GPU-internal state. The
/// CPU never reads or writes the free list after initialization.
///
/// [`bump_buf`]:      Self::bump_buf
/// [`free_list_buf`]: Self::free_list_buf
pub(crate) struct GpuAllocBuffers {
    /// GPU-side persistent bump pointer (single u32).
    bump_buf      : Buffer,
    /// GPU-only free list buffer.
    /// Layout: `count(4)` followed by `(offset: u32, size: u32)` pairs.
    free_list_buf : Buffer,
}

impl GpuAllocBuffers {
    /// Create allocation buffers with the given label prefix and capacity.
    ///
    /// Allocates two GPU buffers: a single-u32 bump pointer and a free
    /// list sized for `max_entries` `(offset, size)` pairs plus a
    /// leading count word. The free list count is initialized to zero.
    pub fn new(
        device      : &Device,
        queue       : &Queue,
        label       : &str,
        max_entries : u32,
    ) -> Self
    {
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

        // Initialize the free list count to zero so the GPU starts
        // with an empty list.
        queue.write_buffer(
            &free_list_buf,
            0,
            bytemuck::bytes_of(&0u32),
        );

        Self {
            bump_buf      : bump_buf,
            free_list_buf : free_list_buf,
        }
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
