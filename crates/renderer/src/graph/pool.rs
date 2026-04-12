//! Buffer pool and deferred release for transient render graph resources.
//!
//! The pool manages a free-list of reusable [`wgpu::Buffer`]s keyed by
//! (size, usage).  Buffers are acquired during graph execution and returned
//! via [`PendingRelease`] once the GPU has finished the submit that
//! referenced them.
//!
//! The pool never shrinks — it converges to steady-state over the first
//! few frames as the graph's transient demand stabilises.

use std::collections::HashMap;

use super::resource::BufferDesc;

// --- PoolKey ---

/// Internal key for the free-list.  Uses raw `u32` for the usage bits to
/// avoid depending on `wgpu::BufferUsages` implementing `Hash`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PoolKey {
    size  : u64,
    usage : u32,
}

impl PoolKey {
    fn from_desc(desc: &BufferDesc) -> Self {
        Self { size: desc.size, usage: desc.usage.bits() }
    }
}

// --- BufferPool ---

/// Reusable pool of GPU buffers for transient render graph resources.
///
/// Keyed by exact (size, usage).  [`acquire`](Self::acquire) checks the
/// free-list first and only creates a new buffer when the list is empty.
/// [`release`](Self::release) returns a buffer to the free-list for future
/// frames.
pub struct BufferPool {
    free : HashMap<PoolKey, Vec<wgpu::Buffer>>,
}

impl BufferPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self { free: HashMap::new() }
    }

    /// Acquire a buffer matching `desc`.
    ///
    /// Returns a recycled buffer from the free-list when available,
    /// otherwise allocates a new one from `device`.
    pub fn acquire(
        &mut self,
        device : &wgpu::Device,
        desc   : &BufferDesc,
    )
        -> wgpu::Buffer
    {
        let key = PoolKey::from_desc(desc);

        if let Some(buf) = self.free.get_mut(&key).and_then(|v| v.pop()) {
            return buf;
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label              : None,
            size               : desc.size,
            usage              : desc.usage,
            mapped_at_creation : false,
        })
    }

    /// Return a buffer to the free-list for reuse.
    pub fn release(&mut self, buffer: wgpu::Buffer) {
        let key = PoolKey {
            size  : buffer.size(),
            usage : buffer.usage().bits(),
        };

        self.free.entry(key).or_default().push(buffer);
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

// --- PendingRelease ---

/// Transient buffers awaiting GPU completion before they can be recycled.
///
/// Returned by [`CompiledGraph::execute`](super::CompiledGraph::execute).
/// Hold this until the GPU submit that used these buffers has completed,
/// then call [`release`](Self::release) to return them to the pool.
///
/// Dropping without calling `release` is safe (wgpu cleans up the
/// underlying GPU resources) but wastes pool capacity — the pool will
/// allocate fresh buffers next frame instead of reusing these.
#[must_use = "call .release() after GPU completion to return buffers to the pool"]
pub struct PendingRelease {
    pub(super) buffers: Vec<wgpu::Buffer>,
}

impl PendingRelease {
    /// Return all held buffers to the pool.
    pub fn release(self, pool: &mut BufferPool) {
        for buffer in self.buffers {
            pool.release(buffer);
        }
    }

    /// Number of transient buffers awaiting release.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Whether this release batch is empty (no transient buffers).
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}
