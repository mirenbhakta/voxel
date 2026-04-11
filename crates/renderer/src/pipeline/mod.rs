//! Pipeline and binding-layout abstractions.

pub mod binding;
pub mod compute;
pub mod reflect;
pub mod render;

pub use binding::{BindEntry, BindKind, BindingLayout, BindingLayoutBuilder};
