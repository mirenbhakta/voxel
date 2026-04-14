//! Pipeline and binding-layout abstractions.

pub mod binding;
pub mod compute;
pub mod reflect;
pub mod render;

pub use binding::{BindEntry, BindKind, BindingLayout, BindingLayoutBuilder};
pub use compute::{ComputePipeline, ComputePipelineDescriptor};
pub use reflect::Reflected;
pub use render::{RenderPipeline, RenderPipelineDescriptor};
