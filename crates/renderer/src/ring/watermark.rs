//! [`CommandIndex`], [`CommandWatermark`], and [`CommandRange`] newtypes.
//!
//! Every message pushed into an upload ring is stamped with a monotonic
//! [`CommandIndex`]. The GPU reports its processing progress back through a
//! [`CommandWatermark`] — the highest command index it has finished
//! processing — and the CPU compares its per-slot watermarks against the
//! observed GPU watermark to retire slots once their messages are no longer
//! in flight.
//!
//! See `.local/renderer_plan.md` §3.6 and principle 6 in
//! `docs/renderer_rewrite_principles.md`.
//!
//! ## Zero is reserved
//!
//! Command indices are assigned starting from `1`. The zero value is a
//! sentinel meaning "unassigned" for [`CommandIndex`] and "the GPU has
//! processed nothing yet" for [`CommandWatermark`]. This lets
//! `RingBookkeeping` use `CommandWatermark::default()` as the "empty slot"
//! marker inside its per-slot watermark vector without an extra `Option`
//! layer.
//!
//! ## First-pass u32 limit
//!
//! The shader side stores the watermark as a `u32`; the CPU side keeps a
//! `u64` for monotonicity and bounds-checks every assignment so that no
//! index may cross `u32::MAX`. See `RingBookkeeping::push` and plan §3.6.
//! A follow-up pass can widen the shader side to `u64` atomics if the
//! workload demands it.

/// A monotonic, subsystem-local command index assigned to each pushed
/// message.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct CommandIndex(pub u64);

// --- CommandIndex ---

impl CommandIndex {
    /// Raw `u64` value. Zero is reserved as the "unassigned" sentinel.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// The highest [`CommandIndex`] the GPU has reported as fully processed.
///
/// Read out of a readback channel (or equivalent CPU-visible signal) and
/// passed to `RingBookkeeping::observe_gpu_watermark` to retire pending
/// slots.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct CommandWatermark(pub u64);

// --- CommandWatermark ---

impl CommandWatermark {
    /// Raw `u64` value. Zero means "the GPU has processed nothing yet."
    pub fn get(self) -> u64 {
        self.0
    }
}

/// An inclusive range of [`CommandIndex`] values assigned by a single
/// `push_many`-style call. Both endpoints are inclusive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandRange {
    pub first: CommandIndex,
    pub last: CommandIndex,
}

// --- CommandRange ---

impl CommandRange {
    /// Number of command indices in the range (always `>= 1`).
    pub fn count(self) -> u64 {
        self.last.0 - self.first.0 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_index_default_is_zero_sentinel() {
        assert_eq!(CommandIndex::default().get(), 0);
    }

    #[test]
    fn command_watermark_default_is_zero_sentinel() {
        assert_eq!(CommandWatermark::default().get(), 0);
    }

    #[test]
    fn command_index_ordering_is_numeric() {
        assert!(CommandIndex(1) < CommandIndex(2));
        assert!(CommandIndex(10) > CommandIndex(5));
        assert_eq!(CommandIndex(7), CommandIndex(7));
    }

    #[test]
    fn command_watermark_ordering_is_numeric() {
        assert!(CommandWatermark(0) < CommandWatermark(1));
        assert!(CommandWatermark(100) > CommandWatermark(99));
        assert_eq!(CommandWatermark(42), CommandWatermark(42));
    }

    #[test]
    fn command_range_count_is_inclusive_on_both_ends() {
        let r = CommandRange { first: CommandIndex(5), last: CommandIndex(5) };
        assert_eq!(r.count(), 1);

        let r = CommandRange { first: CommandIndex(5), last: CommandIndex(9) };
        assert_eq!(r.count(), 5);
    }
}
