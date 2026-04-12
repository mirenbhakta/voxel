//! CPU↔GPU ring primitives with frame-in-flight semantics.
//!
//! The pure-logic core of both rings — slot state, command-index
//! assignment, watermark tracking, and overflow-policy decisions — lives
//! in [`bookkeeping::RingBookkeeping`] and is `pub(crate)`: the wgpu
//! wrappers in [`upload`] and [`readback`] own it and surface policy
//! behavior through their own APIs.
//!
//! Public surface:
//! - [`OverflowPolicy`] / [`PolicyError`] — the three-way policy choice
//!   and the allocation-free error type returned by ring operations.
//! - [`CommandIndex`] / [`CommandWatermark`] / [`CommandRange`] — the
//!   monotonic-index / GPU-progress / inclusive-range newtypes used
//!   throughout the ring APIs.
//! - [`UploadRing`] — CPU-write, GPU-read ring with per-frame-in-flight
//!   slot rotation and configurable overflow policy.
//!
//! See `.local/renderer_plan.md` §3.3–3.6 and principle 6 in
//! `docs/renderer_rewrite_principles.md`.

pub(crate) mod bookkeeping;
pub mod policy;
pub mod readback;
pub mod upload;
pub mod watermark;

pub use policy::{OverflowPolicy, PolicyError};
pub use upload::UploadRing;
pub use watermark::{CommandIndex, CommandRange, CommandWatermark};
