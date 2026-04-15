//! Renderer crate — primitives layer.

pub mod buffer;
pub mod commands;
pub mod error;
pub mod frame;
pub mod gpu_consts;
pub mod graph;
pub mod nodes;
pub mod pipeline;
pub mod shader;

mod device;
mod subchunk_test;

pub use buffer::StagedBuffer;
pub use commands::{
    ColorAttachment, Commands, DepthAttachment, RasterPass, RasterPassDesc,
};
pub use device::{FrameEncoder, RendererContext, SurfaceFrame};
pub use error::RendererError;
pub use frame::{FrameCount, FrameIndex};
pub use gpu_consts::{GpuConsts, GpuConstsData};
pub use graph::PassContext;
pub use pipeline::{BindEntry, BindKind, BindingLayout, BindingLayoutBuilder};
pub use pipeline::{ComputePipeline, ComputePipelineDescriptor};
pub use pipeline::{RenderPipeline, RenderPipelineDescriptor};
pub use shader::{ShaderModule, ShaderSource};
pub use subchunk_test::{
    DEPTH_FORMAT as SUBCHUNK_DEPTH_FORMAT,
    MAX_CANDIDATES as SUBCHUNK_MAX_CANDIDATES,
    SubchunkInstance,
    SubchunkOccupancy,
    SubchunkTest,
    TestCamera,
    occupancy_exposure,
    sphere_occupancy,
};
