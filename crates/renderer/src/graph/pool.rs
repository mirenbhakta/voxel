//! Buffer and texture pools and deferred release for transient render graph
//! resources.
//!
//! Each pool manages a free-list of reusable GPU resources keyed by their
//! descriptor.  Resources are acquired during graph execution and returned
//! via [`PendingRelease`] once the GPU has finished the submit that
//! referenced them.
//!
//! Pools never shrink — they converge to steady-state over the first few
//! frames as the graph's transient demand stabilises.

use std::collections::HashMap;

use super::resource::{BufferDesc, TextureDesc};

// --- PoolKey ---

/// Internal key for the buffer free-list.  Uses raw `u32` for the usage bits
/// to avoid depending on `wgpu::BufferUsages` implementing `Hash`.
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

// --- TexturePoolKey ---

/// Internal key for the texture free-list.  Uses raw `u32` for the usage
/// bits (no `Hash` on `TextureUsages`); format and dimension impl `Hash`
/// directly so they are stored by value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TexturePoolKey {
    width                 : u32,
    height                : u32,
    depth_or_array_layers : u32,
    format                : wgpu::TextureFormat,
    usage                 : u32,   // wgpu::TextureUsages::bits()
    dimension             : wgpu::TextureDimension,
    mip_level_count       : u32,
    sample_count          : u32,
}

impl TexturePoolKey {
    fn from_desc(desc: &TextureDesc) -> Self {
        Self {
            width                 : desc.size.width,
            height                : desc.size.height,
            depth_or_array_layers : desc.size.depth_or_array_layers,
            format                : desc.format,
            usage                 : desc.usage.bits(),
            dimension             : desc.dimension,
            mip_level_count       : desc.mip_level_count,
            sample_count          : desc.sample_count,
        }
    }

    fn from_texture(texture: &wgpu::Texture) -> Self {
        let size = texture.size();
        Self {
            width                 : size.width,
            height                : size.height,
            depth_or_array_layers : size.depth_or_array_layers,
            format                : texture.format(),
            usage                 : texture.usage().bits(),
            dimension             : texture.dimension(),
            mip_level_count       : texture.mip_level_count(),
            sample_count          : texture.sample_count(),
        }
    }
}

// --- TexturePool ---

/// Reusable pool of GPU textures for transient render graph resources.
///
/// Keyed by the full texture descriptor.  [`acquire`](Self::acquire) checks
/// the free-list first and only creates a new texture when the list is empty.
/// [`release`](Self::release) returns a texture to the free-list for future
/// frames.
pub struct TexturePool {
    free: HashMap<TexturePoolKey, Vec<wgpu::Texture>>,
}

impl TexturePool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self { free: HashMap::new() }
    }

    /// Acquire a texture matching `desc`.
    ///
    /// Returns a recycled texture from the free-list when available,
    /// otherwise allocates a new one from `device`.
    pub fn acquire(
        &mut self,
        device : &wgpu::Device,
        desc   : &TextureDesc,
    )
        -> wgpu::Texture
    {
        let key = TexturePoolKey::from_desc(desc);

        if let Some(tex) = self.free.get_mut(&key).and_then(|v| v.pop()) {
            return tex;
        }

        device.create_texture(&wgpu::TextureDescriptor {
            label           : None,
            size            : desc.size,
            mip_level_count : desc.mip_level_count,
            sample_count    : desc.sample_count,
            dimension       : desc.dimension,
            format          : desc.format,
            usage           : desc.usage,
            view_formats    : &[],
        })
    }

    /// Return a texture to the free-list for reuse.
    pub fn release(&mut self, texture: wgpu::Texture) {
        let key = TexturePoolKey::from_texture(&texture);
        self.free.entry(key).or_default().push(texture);
    }
}

impl Default for TexturePool {
    fn default() -> Self {
        Self::new()
    }
}

// --- PendingRelease ---

/// Transient resources awaiting GPU completion before they can be recycled.
///
/// Returned by [`CompiledGraph::execute`](super::CompiledGraph::execute).
/// Hold this until the GPU submit that used these resources has completed,
/// then call [`release`](Self::release) to return them to the pools.
///
/// Dropping without calling `release` is safe (wgpu cleans up the
/// underlying GPU resources) but wastes pool capacity — the pools will
/// allocate fresh resources next frame instead of reusing these.
#[must_use = "call .release() after GPU completion to return resources to the pools"]
pub struct PendingRelease {
    pub(super) buffers  : Vec<wgpu::Buffer>,
    pub(super) textures : Vec<wgpu::Texture>,
}

impl PendingRelease {
    /// Return all held resources to their respective pools.
    pub fn release(self, buf_pool: &mut BufferPool, tex_pool: &mut TexturePool) {
        for buffer in self.buffers {
            buf_pool.release(buffer);
        }
        for texture in self.textures {
            tex_pool.release(texture);
        }
    }

    /// Total number of transient resources awaiting release.
    pub fn len(&self) -> usize {
        self.buffers.len() + self.textures.len()
    }

    /// Whether this release batch is empty (no transient resources).
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty() && self.textures.is_empty()
    }

    /// Number of transient buffers awaiting release.
    pub fn buffers_len(&self) -> usize {
        self.buffers.len()
    }

    /// Number of transient textures awaiting release.
    pub fn textures_len(&self) -> usize {
        self.textures.len()
    }
}
