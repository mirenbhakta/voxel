//! [`OverflowPolicy`] enum and [`PolicyError`] — the three-way choice for
//! how a ring handles a full slot or a pending rotation target.
//!
//! See `.local/renderer_plan.md` §3.3 and principle 6 in
//! `docs/renderer_rewrite_principles.md`.

/// Behavior when the CPU attempts to push a message into a full ring slot
/// or rotate into a slot whose highest command is still pending on the GPU.
///
/// Chosen per-ring at construction and fixed for the ring's lifetime: a
/// ring cannot switch policies at runtime. If a subsystem needs both
/// semantics, it should construct separate rings — see §3.3 for the
/// rationale.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverflowPolicy {
    /// Block until the GPU retires the target slot and the operation
    /// becomes possible. Appropriate when the work eventually completes and
    /// a CPU stall is the correct backpressure.
    ///
    /// The pure-logic [`RingBookkeeping`] layer does not contain the block
    /// loop itself — it returns [`PolicyError::RotateWouldBlock`] and
    /// relies on the wgpu-touching wrapper to call `device.poll(Wait)`,
    /// observe the GPU watermark via a readback channel, and retry.
    ///
    /// [`RingBookkeeping`]: crate::ring::bookkeeping::RingBookkeeping
    Block,

    /// Return [`PolicyError::PushWouldBlock`] so the caller can coalesce,
    /// back off, or drop the message entirely. Appropriate when message
    /// loss is tolerable (profiling samples, per-frame hints). Under this
    /// policy `rotate_frame` is infallible — the target slot is cleared
    /// unconditionally on rotation (see [`RingBookkeeping::rotate_frame`]).
    ///
    /// [`RingBookkeeping::rotate_frame`]: crate::ring::bookkeeping::RingBookkeeping::rotate_frame
    Drop,

    /// Track GPU progress and panic once `rotate_frame` has been called
    /// `frames` consecutive times without the GPU watermark advancing.
    /// Appropriate when "the GPU has fallen this far behind" is
    /// unrecoverable and the process should tear down with full diagnostic
    /// context rather than block indefinitely.
    ///
    /// Semantics of `frames`: it is a grace period measured in rotate
    /// attempts. The panic fires on the `(frames + 1)`-th consecutive
    /// no-progress attempt. A value of `frames: 3` therefore tolerates
    /// three no-progress rotations and panics on the fourth — this
    /// matches the test description in §9.6.
    TimeoutCrash {
        /// Maximum number of consecutive `rotate_frame` attempts tolerated
        /// without forward progress on the GPU watermark.
        frames: u32,
    },
}

/// Errors returned by `RingBookkeeping` operations.
///
/// `Copy`-able: no embedded strings. The wgpu wrapper layer adds label and
/// per-ring context when propagating errors outward, keeping the
/// pure-logic core allocation-free on the error path.
///
/// `TimeoutCrash` does not produce a variant here — it panics in place
/// with full diagnostic context per principle 6 ("loud failure"), rather
/// than threading through `Result`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PolicyError {
    /// A `push` call would place more messages into the current slot than
    /// `capacity_per_frame` permits.
    ///
    /// Under [`OverflowPolicy::Drop`] the caller is expected to ignore or
    /// coalesce the message; under [`OverflowPolicy::Block`] /
    /// [`OverflowPolicy::TimeoutCrash`] the wgpu wrapper rotates to the
    /// next slot and retries the push.
    #[error(
        "ring push would block: slot {slot} fill={current_fill} + \
         requested={requested} > capacity_per_frame={capacity}"
    )]
    PushWouldBlock {
        /// Slot being pushed into.
        slot: u32,
        /// Existing fill of that slot before the rejected push.
        current_fill: u32,
        /// Number of messages the rejected call tried to push.
        requested: u32,
        /// Ring capacity_per_frame.
        capacity: u32,
    },

    /// A `rotate_frame` call would overwrite a slot whose highest pending
    /// command index is still greater than the observed GPU watermark.
    ///
    /// Returned under [`OverflowPolicy::Block`] and
    /// [`OverflowPolicy::TimeoutCrash`]. Under [`OverflowPolicy::Drop`]
    /// rotation is infallible (the slot is cleared unconditionally) and
    /// this variant is never produced.
    #[error(
        "ring rotate would block: next slot {next_slot} has pending \
         watermark {pending} > observed GPU watermark {observed}"
    )]
    RotateWouldBlock {
        /// The slot the caller attempted to rotate into.
        next_slot: u32,
        /// That slot's highest pending command index.
        pending: u64,
        /// Last GPU watermark observed on this ring.
        observed: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_policy_is_value_type() {
        let a = OverflowPolicy::Block;
        let b = OverflowPolicy::Drop;
        let c = OverflowPolicy::TimeoutCrash { frames: 3 };

        // Copy + Eq: can compare freely.
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(a, OverflowPolicy::Block);
        assert_eq!(
            OverflowPolicy::TimeoutCrash { frames: 3 },
            OverflowPolicy::TimeoutCrash { frames: 3 },
        );
        assert_ne!(
            OverflowPolicy::TimeoutCrash { frames: 3 },
            OverflowPolicy::TimeoutCrash { frames: 4 },
        );
    }

    #[test]
    fn overflow_policy_debug_round_trips_structurally() {
        // Smoke test: Debug impl renders all three variants with no panics
        // and each is distinguishable in the output.
        let dbg_block = format!("{:?}", OverflowPolicy::Block);
        let dbg_drop  = format!("{:?}", OverflowPolicy::Drop);
        let dbg_tc    = format!("{:?}", OverflowPolicy::TimeoutCrash { frames: 5 });

        assert!(dbg_block.contains("Block"));
        assert!(dbg_drop.contains("Drop"));
        assert!(dbg_tc.contains("TimeoutCrash"));
        assert!(dbg_tc.contains('5'));
    }

    #[test]
    fn policy_error_push_variant_display_is_informative() {
        let e = PolicyError::PushWouldBlock {
            slot: 1,
            current_fill: 4,
            requested: 2,
            capacity: 4,
        };
        let msg = format!("{e}");
        assert!(msg.contains("slot 1"));
        assert!(msg.contains("4"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn policy_error_rotate_variant_display_is_informative() {
        let e = PolicyError::RotateWouldBlock {
            next_slot: 0,
            pending: 17,
            observed: 10,
        };
        let msg = format!("{e}");
        assert!(msg.contains("slot 0"));
        assert!(msg.contains("17"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn policy_error_is_copy() {
        let e = PolicyError::PushWouldBlock {
            slot: 1,
            current_fill: 4,
            requested: 2,
            capacity: 4,
        };
        let e2 = e;
        // If PolicyError weren't Copy this would move `e`.
        assert_eq!(e, e2);
    }
}
