//! Renderer crate — primitives layer.

pub mod buffer;
pub mod error;
pub mod frame;
pub mod gpu_consts;
pub mod graph;
pub mod pipeline;
pub mod shader;

mod device;
mod subchunk_test;

pub use buffer::StagedBuffer;
pub use device::{FrameEncoder, RendererContext, SurfaceFrame};
pub use error::RendererError;
pub use frame::{FrameCount, FrameIndex};
pub use gpu_consts::{GpuConsts, GpuConstsData};
pub use pipeline::{BindEntry, BindKind, BindingLayout, BindingLayoutBuilder};
pub use pipeline::{ComputePipeline, ComputePipelineDescriptor};
pub use pipeline::{RenderPipeline, RenderPipelineDescriptor};
pub use shader::{ShaderModule, ShaderSource};
pub use subchunk_test::{SubchunkTest, SubchunkOccupancy, TestCamera, sphere_occupancy};
