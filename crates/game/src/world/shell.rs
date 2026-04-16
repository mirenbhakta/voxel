//! Single-level clipmap shell.
//!
//! A [`Shell`] tracks the resident set of sub-chunks around a camera-driven
//! "origin corner" — a world-space grid vertex where 8 sub-chunks meet —
//! at one LOD level. Moving the corner (via [`Shell::recenter`]) returns a
//! [`ShellDiff`] listing the coords that entered and left residency, which
//! the caller feeds to the slot pool and prep/commit pipeline.
//!
//! The shell has no knowledge of the LOD level it represents — the level
//! is carried by whatever structure owns the shell. For cross-level
//! nesting (an `L_(n+1)` shell cleanly containing an integer number of
//! `L_n` sub-chunks), radii should be even; see
//! [`crate::world::residency::Residency`] for the corner-derivation rule
//! that keeps levels aligned.

#![allow(dead_code)]

use crate::world::coord::SubchunkCoord;

// --- Shell ---

/// Residency window: the `(2r)³` axis-aligned block of sub-chunks whose
/// coords lie in `[corner - r, corner + r - 1]` per axis.
///
/// The corner is a grid-vertex coord — interpreted as the world point
/// `corner * extent`, the shell extends `r` sub-chunks symmetrically into
/// each of the 8 octants around that vertex. At `r = 2` this gives the
/// minimum shell size that nests cleanly inside a coarser level of the
/// same structure (a 4×4×4 block that equals 2×2×2 coarser sub-chunks).
pub struct Shell {
    radius: [u32; 3],
    corner: SubchunkCoord,
}

impl Shell {
    /// Create a shell with the given per-axis radius, anchored at `corner`.
    ///
    /// At `r = 1` the shell has 2 sub-chunks per axis (8 total). Every
    /// radius component must be at least 1 — a zero-radius shell has no
    /// residents in this scheme and would trip the pool-size assertion.
    pub fn new(radius: [u32; 3], corner: SubchunkCoord) -> Self {
        debug_assert!(
            radius[0] >= 1 && radius[1] >= 1 && radius[2] >= 1,
            "shell radius must be >= 1 per axis (dim = 2r); got {radius:?}",
        );
        Self { radius, corner }
    }

    /// Current anchor corner (grid vertex in this level's sub-chunk coord
    /// space). The shell extends `r` sub-chunks into each of the 8 octants
    /// around `corner * extent`.
    pub fn corner(&self) -> SubchunkCoord {
        self.corner
    }

    /// Per-axis residency radius.
    pub fn radius(&self) -> [u32; 3] {
        self.radius
    }

    /// The minimum pool dimensions that host this shell without collisions.
    ///
    /// Two resident coords differing by `pool_dims[i]` on axis `i` would
    /// map to the same toroidal slot. The shell spans `2 * radius[i]`
    /// coords per axis, so `pool_dims[i] = 2 * radius[i]` is the tightest
    /// safe size. Hysteresis pools use strictly larger dims.
    pub fn pool_dims(&self) -> [u32; 3] {
        [2 * self.radius[0], 2 * self.radius[1], 2 * self.radius[2]]
    }

    /// Resident count: `(2r[0]) * (2r[1]) * (2r[2])`.
    pub fn resident_count(&self) -> usize {
        let [a, b, c] = self.pool_dims();
        (a as usize) * (b as usize) * (c as usize)
    }

    /// Whether `coord` currently lies inside the shell.
    ///
    /// Resident iff `corner - r <= coord < corner + r` per axis. The upper
    /// bound is exclusive — a 2r-wide window places `r` coords on the
    /// negative side of the corner and `r` starting at the corner.
    pub fn contains(&self, coord: SubchunkCoord) -> bool {
        let dx = coord.x - self.corner.x;
        let dy = coord.y - self.corner.y;
        let dz = coord.z - self.corner.z;
        let rx = self.radius[0] as i32;
        let ry = self.radius[1] as i32;
        let rz = self.radius[2] as i32;
        (-rx..rx).contains(&dx) && (-ry..ry).contains(&dy) && (-rz..rz).contains(&dz)
    }

    /// Enumerate every resident coord. Order: X fastest, then Y, then Z.
    pub fn residents(&self) -> impl Iterator<Item = SubchunkCoord> + '_ {
        let c  = self.corner;
        let rx = self.radius[0] as i32;
        let ry = self.radius[1] as i32;
        let rz = self.radius[2] as i32;
        (-rz..rz).flat_map(move |dz| {
            (-ry..ry).flat_map(move |dy| {
                (-rx..rx).map(move |dx| {
                    SubchunkCoord::new(c.x + dx, c.y + dy, c.z + dz)
                })
            })
        })
    }

    /// Move the corner to `new_corner` and return the residency diff.
    ///
    /// `added` contains coords that are resident at the new corner but not
    /// the old. `removed` is the reverse. A no-op recenter (same coord)
    /// returns empty diffs without mutation.
    pub fn recenter(&mut self, new_corner: SubchunkCoord) -> ShellDiff {
        if new_corner == self.corner {
            return ShellDiff::empty();
        }

        let old = Shell::new(self.radius, self.corner);
        let new = Shell::new(self.radius, new_corner);

        let mut added   = Vec::new();
        let mut removed = Vec::new();

        for coord in new.residents() {
            if !old.contains(coord) {
                added.push(coord);
            }
        }
        for coord in old.residents() {
            if !new.contains(coord) {
                removed.push(coord);
            }
        }

        self.corner = new_corner;
        ShellDiff { added, removed }
    }
}

// --- ShellDiff ---

/// Changes to a shell's resident set produced by [`Shell::recenter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellDiff {
    pub added:   Vec<SubchunkCoord>,
    pub removed: Vec<SubchunkCoord>,
}

impl ShellDiff {
    pub fn empty() -> Self {
        Self { added: Vec::new(), removed: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::pool::SlotPool;

    // -- pool_dims / resident_count --

    #[test]
    fn pool_dims_is_two_r() {
        let s = Shell::new([1, 2, 3], SubchunkCoord::new(0, 0, 0));
        assert_eq!(s.pool_dims(),      [2, 4, 6]);
        assert_eq!(s.resident_count(), 2 * 4 * 6);
    }

    #[test]
    fn pool_dims_minimum_radius_is_eight() {
        // r=1 is the minimum: 2 sub-chunks per axis = 8 total.
        let s = Shell::new([1, 1, 1], SubchunkCoord::new(10, 10, 10));
        assert_eq!(s.pool_dims(),      [2, 2, 2]);
        assert_eq!(s.resident_count(), 8);
    }

    // -- contains / residents --

    #[test]
    fn contains_matches_residents_iter() {
        // r=[2,2,2] → 4 per axis → 64 residents. Shell at corner (5, -3, 0)
        // covers coords in [3, 6] × [-5, -2] × [-2, 1].
        let s = Shell::new([2, 2, 2], SubchunkCoord::new(5, -3, 0));
        let set: std::collections::HashSet<_> = s.residents().collect();
        assert_eq!(set.len(), 64);

        for c in &set {
            assert!(s.contains(*c));
        }
        // Outside the shell by one coord on each axis.
        assert!(!s.contains(SubchunkCoord::new(7, -3, 0)));   // x past upper (corner.x + r = 7)
        assert!(!s.contains(SubchunkCoord::new(5, -6, 0)));   // y past lower
        assert!(!s.contains(SubchunkCoord::new(5, -3, -3)));  // z past lower
        // Corner coord is always resident (it's the positive-octant anchor).
        assert!(s.contains(SubchunkCoord::new(5, -3, 0)));
    }

    #[test]
    fn residents_count_matches_resident_count() {
        let s = Shell::new([2, 1, 3], SubchunkCoord::new(0, 0, 0));
        assert_eq!(s.residents().count(), s.resident_count());
    }

    // -- recenter --

    #[test]
    fn recenter_noop_returns_empty_diff() {
        let mut s = Shell::new([1, 1, 1], SubchunkCoord::new(0, 0, 0));
        let diff  = s.recenter(SubchunkCoord::new(0, 0, 0));
        assert!(diff.is_empty());
        assert_eq!(s.corner(), SubchunkCoord::new(0, 0, 0));
    }

    #[test]
    fn recenter_one_step_rolls_one_face() {
        // 4×4×4 shell → moving the corner by +1 in X rolls 16 coords out and
        // 16 in. Old shell covers x ∈ [-2, 1]; new covers x ∈ [-1, 2].
        let mut s = Shell::new([2, 2, 2], SubchunkCoord::new(0, 0, 0));
        let diff  = s.recenter(SubchunkCoord::new(1, 0, 0));

        assert_eq!(diff.added.len(),   16);
        assert_eq!(diff.removed.len(), 16);

        // Added face: x = 2 (new_corner.x + r - 1).
        for c in &diff.added {
            assert_eq!(c.x, 2);
        }
        // Removed face: x = -2 (old_corner.x - r).
        for c in &diff.removed {
            assert_eq!(c.x, -2);
        }
    }

    #[test]
    fn recenter_far_jump_removes_old_adds_new_fully() {
        let mut s = Shell::new([2, 2, 2], SubchunkCoord::new(0, 0, 0));
        let diff  = s.recenter(SubchunkCoord::new(100, 0, 0));
        // Disjoint old and new shells: every coord in each is added/removed.
        assert_eq!(diff.added.len(),   64);
        assert_eq!(diff.removed.len(), 64);
    }

    // -- integration with SlotPool --

    #[test]
    fn shell_and_pool_roll_consistently() {
        let mut shell = Shell::new([2, 2, 2], SubchunkCoord::new(0, 0, 0));
        let mut pool: SlotPool<&'static str> = SlotPool::new(shell.pool_dims());

        // Populate initial residents.
        for c in shell.residents() {
            assert_eq!(pool.insert(c, "initial"), None);
        }
        assert_eq!(pool.occupied_count(), shell.resident_count());

        // Roll by +1 on X: evict removed, insert added.
        let diff = shell.recenter(SubchunkCoord::new(1, 0, 0));
        assert_eq!(diff.added.len(),   16);
        assert_eq!(diff.removed.len(), 16);

        for c in &diff.removed {
            assert_eq!(pool.remove(*c), Some("initial"));
        }
        for c in &diff.added {
            assert_eq!(pool.insert(*c, "rolled"), None);
        }
        assert_eq!(pool.occupied_count(), shell.resident_count());

        // Post-roll, every shell resident is stored under the correct label.
        // Added face in the new shell is at x=2; rest came from "initial".
        for c in shell.residents() {
            let v = pool.get(c).copied().expect("every resident stored");
            let label = if c.x == 2 { "rolled" } else { "initial" };
            assert_eq!(v, label, "at {c:?}");
        }
    }

    #[test]
    fn shell_and_pool_far_jump_resets_cleanly() {
        // Disjoint shells: every slot is evicted and repopulated.
        let mut shell = Shell::new([2, 2, 2], SubchunkCoord::new(0, 0, 0));
        let mut pool: SlotPool<u32> = SlotPool::new(shell.pool_dims());
        for c in shell.residents() {
            pool.insert(c, 0);
        }

        let diff = shell.recenter(SubchunkCoord::new(100, 0, 0));
        for c in &diff.removed {
            assert!(pool.remove(*c).is_some());
        }
        assert_eq!(pool.occupied_count(), 0);

        for c in &diff.added {
            pool.insert(*c, 1);
        }
        assert_eq!(pool.occupied_count(), shell.resident_count());
    }
}
