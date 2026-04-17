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
pub use gpu_consts::{GpuConsts, GpuConstsData};
pub use graph::PassContext;
pub use pipeline::{BindEntry, BindKind, PipelineBindLayout};
pub use pipeline::{ComputePipeline, ComputePipelineDescriptor};
pub use pipeline::{RenderPipeline, RenderPipelineDescriptor};
pub use shader::{ShaderModule, ShaderSource};
pub use subchunk::{
    DEPTH_FORMAT as SUBCHUNK_DEPTH_FORMAT,
    MAX_CANDIDATES as SUBCHUNK_MAX_CANDIDATES,
    MAX_LEVELS as SUBCHUNK_MAX_LEVELS,
    DirtyEntry,
    DirtyReport,
    LodMaskUniform,
    PrepRequest,
    SubchunkCamera,
    SubchunkInstance,
    SubchunkOccupancy,
    WorldRenderer,
    sphere_occupancy,
};
