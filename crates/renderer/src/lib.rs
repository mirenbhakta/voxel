//! Renderer crate — primitives layer.

pub mod buffer;
pub mod commands;
pub mod error;
pub mod frame;
pub mod gpu_consts;
pub mod graph;
pub mod multi_buffer;
pub mod nodes;
pub mod pipeline;
pub mod readback;
pub mod shader;
pub mod worldgen;

mod device;
mod subchunk;

pub use buffer::StagedBuffer;
pub use multi_buffer::MultiBufferRing;
pub use readback::{OverflowPolicy, ReadbackChannel};
pub use commands::{
    ColorAttachment, Commands, DepthAttachment, RasterPass, RasterPassDesc,
};
pub use device::{FrameEncoder, RendererContext, SurfaceFrame};
pub use error::RendererError;
pub use frame::{FrameCount, FrameIndex};
pub use gpu_consts::{GpuConsts, GpuConstsData, LevelStatic};
pub use graph::PassContext;
pub use pipeline::{BindEntry, BindKind, PipelineBindLayout};
pub use pipeline::{ComputePipeline, ComputePipelineDescriptor};
pub use pipeline::{RenderPipeline, RenderPipelineDescriptor};
pub use shader::{ShaderModule, ShaderSource};
pub use subchunk::{
    DEPTH_FORMAT as SUBCHUNK_DEPTH_FORMAT,
    MATERIAL_BLOCK_BYTES,
    MATERIAL_DESC_CAPACITY,
    MATERIAL_POOL_SLOTS_PER_SEGMENT,
    MATERIAL_SEGMENT_BYTES,
    MAX_CANDIDATES as SUBCHUNK_MAX_CANDIDATES,
    MAX_LEVELS as SUBCHUNK_MAX_LEVELS,
    MAX_MATERIAL_POOL_SEGMENTS,
    BITS_EXPOSURE_MASK,
    BITS_IS_SOLID,
    BITS_MATERIAL_SLOT_SHIFT,
    BITS_RESIDENT,
    DirEntry,
    DirtyEntry,
    DirtyReport,
    EXPOSURE_STAGING_REQUEST_IDX_SENTINEL,
    LodMaskUniform,
    MATERIAL_DATA_SLOT_INVALID,
    MATERIAL_SLOT_INVALID,
    MaterialBlock,
    MaterialDesc,
    MaterialPatchCopy,
    PatchCopy,
    PrepRequest,
    SubchunkCamera,
    SubchunkInstance,
    SubchunkOccupancy,
    WorldRenderer,
    sphere_occupancy,
};
