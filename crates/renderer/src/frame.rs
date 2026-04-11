//! Frame count and index primitives — the "how many frames in flight and
//! which one are we on" plumbing.
//!
//! Principle 1: the frame count is a runtime value, not a const generic. It
//! comes from surface configuration in windowed mode and is passed explicitly
//! in headless mode. See `.local/renderer_plan.md` §3.1.

use crate::error::RendererError;

/// The number of frames in flight. Runtime value, typically 2 or 3, derived
/// from surface configuration in windowed mode or passed explicitly in
/// headless mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameCount(u32);

impl FrameCount {
    /// Minimum supported frames in flight. Double-buffering is the lower bound
    /// below which the rings have nothing to rotate against.
    pub const MIN: u32 = 2;

    /// Maximum supported frames in flight. `4` gives one frame of headroom
    /// above the typical 3 without imposing a const-generic cap.
    pub const MAX: u32 = 4;

    /// Construct a `FrameCount` from a raw value. Returns
    /// [`RendererError::InvalidFrameCount`] if `n` is outside
    /// `Self::MIN..=Self::MAX`.
    pub fn new(n: u32) -> Result<Self, RendererError> {
        if (Self::MIN..=Self::MAX).contains(&n) {
            Ok(Self(n))
        } else {
            Err(RendererError::InvalidFrameCount {
                requested: n,
                min: Self::MIN,
                max: Self::MAX,
            })
        }
    }

    /// Get the raw count.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Monotonic frame counter. [`Self::slot`] returns `index % frame_count`,
/// which is the per-frame rotating slot index used by ring primitives.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FrameIndex(u64);

impl FrameIndex {
    /// Advance the counter by one. Saturates at `u64::MAX`, which at 60 FPS is
    /// ~9.7 billion years away — not a concern in practice, but saturating
    /// makes the "cannot overflow" contract explicit instead of relying on
    /// debug-mode panic + release-mode wrap.
    pub fn advance(&mut self) {
        self.0 = self.0.saturating_add(1);
    }

    /// Get the raw index.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Per-frame rotating slot index in `0..count.get()`.
    pub fn slot(self, count: FrameCount) -> u32 {
        (self.0 % count.get() as u64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_accepts_min_and_max() {
        assert_eq!(FrameCount::new(FrameCount::MIN).unwrap().get(), FrameCount::MIN);
        assert_eq!(FrameCount::new(FrameCount::MAX).unwrap().get(), FrameCount::MAX);
    }

    #[test]
    fn frame_count_rejects_below_min() {
        let err = FrameCount::new(1).unwrap_err();
        match err {
            RendererError::InvalidFrameCount { requested, min, max } => {
                assert_eq!(requested, 1);
                assert_eq!(min, FrameCount::MIN);
                assert_eq!(max, FrameCount::MAX);
            }
            other => panic!("expected InvalidFrameCount, got {other:?}"),
        }
    }

    #[test]
    fn frame_count_rejects_above_max() {
        assert!(matches!(
            FrameCount::new(FrameCount::MAX + 1),
            Err(RendererError::InvalidFrameCount { .. })
        ));
    }

    #[test]
    fn frame_count_rejects_zero() {
        assert!(matches!(
            FrameCount::new(0),
            Err(RendererError::InvalidFrameCount { .. })
        ));
    }

    #[test]
    fn frame_index_default_is_zero() {
        assert_eq!(FrameIndex::default().get(), 0);
    }

    #[test]
    fn frame_index_advance_is_monotonic() {
        let mut idx = FrameIndex::default();
        idx.advance();
        idx.advance();
        assert_eq!(idx.get(), 2);
    }

    #[test]
    fn frame_index_slot_wraps_correctly() {
        let count = FrameCount::new(2).unwrap();
        let mut idx = FrameIndex::default();
        assert_eq!(idx.slot(count), 0);
        idx.advance();
        assert_eq!(idx.slot(count), 1);
        idx.advance();
        assert_eq!(idx.slot(count), 0);
        idx.advance();
        assert_eq!(idx.slot(count), 1);
    }

    #[test]
    fn frame_index_slot_wraps_at_frame_count_3() {
        let count = FrameCount::new(3).unwrap();
        let mut idx = FrameIndex::default();
        for expected in [0u32, 1, 2, 0, 1, 2, 0] {
            assert_eq!(idx.slot(count), expected);
            idx.advance();
        }
    }
}
