//! Single-level clipmap shell.
//!
//! A [`Shell`] tracks the resident set of sub-chunks around a camera-driven
//! center at one LOD level. Moving the center (via [`Shell::recenter`])
//! returns a [`ShellDiff`] listing the coords that entered and left
//! residency, which the caller feeds to the slot pool and prep/commit
//! pipeline.
//!
//! The shell has no knowledge of the LOD level it represents — the level
//! is carried by whatever structure owns the shell.

#![allow(dead_code)]

use crate::world::coord::SubchunkCoord;

// --- Shell ---

/// Residency window: every sub-chunk within `radius` (L∞ norm) of `center`
/// is resident. The resident set is a `(2r+1)³` axis-aligned block.
pub struct Shell {
    radius: [u32; 3],
    center: SubchunkCoord,
}

impl Shell {
    /// Create a shell with the given per-axis radius, centered on `center`.
    ///
    /// The radius is extent per axis (radius 1 → a 3×3×3 block). Every
    /// component must fit in an `i32` after doubling without overflow; for
    /// realistic clipmap radii (≤ a few hundred) this is never a concern.
    pub fn new(radius: [u32; 3], center: SubchunkCoord) -> Self {
        Self { radius, center }
    }

    /// Current center coord.
    pub fn center(&self) -> SubchunkCoord {
        self.center
    }

    /// Per-axis residency radius.
    pub fn radius(&self) -> [u32; 3] {
        self.radius
    }

    /// The minimum pool dimensions that host this shell without collisions.
    ///
    /// Two resident coords differing by `pool_dims[i]` on axis `i` would
    /// map to the same toroidal slot. The shell spans `2 * radius[i] + 1`
    /// coords per axis, so `pool_dims[i] = 2 * radius[i] + 1` is the
    /// tightest safe size. Hysteresis pools use strictly larger dims.
    pub fn pool_dims(&self) -> [u32; 3] {
        [
            2 * self.radius[0] + 1,
            2 * self.radius[1] + 1,
            2 * self.radius[2] + 1,
        ]
    }

    /// Resident count: `(2r[0]+1) * (2r[1]+1) * (2r[2]+1)`.
    pub fn resident_count(&self) -> usize {
        let [a, b, c] = self.pool_dims();
        (a as usize) * (b as usize) * (c as usize)
    }

    /// Whether `coord` currently lies inside the shell.
    pub fn contains(&self, coord: SubchunkCoord) -> bool {
        let dx = coord.x - self.center.x;
        let dy = coord.y - self.center.y;
        let dz = coord.z - self.center.z;
        dx.unsigned_abs() <= self.radius[0]
            && dy.unsigned_abs() <= self.radius[1]
            && dz.unsigned_abs() <= self.radius[2]
    }

    /// Enumerate every resident coord. Order: X fastest, then Y, then Z.
    pub fn residents(&self) -> impl Iterator<Item = SubchunkCoord> + '_ {
        let c  = self.center;
        let rx = self.radius[0] as i32;
        let ry = self.radius[1] as i32;
        let rz = self.radius[2] as i32;
        (-rz..=rz).flat_map(move |dz| {
            (-ry..=ry).flat_map(move |dy| {
                (-rx..=rx).map(move |dx| {
                    SubchunkCoord::new(c.x + dx, c.y + dy, c.z + dz)
                })
            })
        })
    }

    /// Move the center to `new_center` and return the residency diff.
    ///
    /// `added` contains coords that are resident at the new center but not
    /// the old. `removed` is the reverse. A no-op recenter (same coord)
    /// returns empty diffs without mutation.
    pub fn recenter(&mut self, new_center: SubchunkCoord) -> ShellDiff {
        if new_center == self.center {
            return ShellDiff::empty();
        }

        let old = Shell::new(self.radius, self.center);
        let new = Shell::new(self.radius, new_center);

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

        self.center = new_center;
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
    fn pool_dims_is_two_r_plus_one() {
        let s = Shell::new([1, 2, 3], SubchunkCoord::new(0, 0, 0));
        assert_eq!(s.pool_dims(),      [3, 5, 7]);
        assert_eq!(s.resident_count(), 3 * 5 * 7);
    }

    #[test]
    fn pool_dims_radius_zero_is_unit() {
        let s = Shell::new([0, 0, 0], SubchunkCoord::new(10, 10, 10));
        assert_eq!(s.pool_dims(),      [1, 1, 1]);
        assert_eq!(s.resident_count(), 1);
    }

    // -- contains / residents --

    #[test]
    fn contains_matches_residents_iter() {
        let s = Shell::new([1, 1, 1], SubchunkCoord::new(5, -3, 0));
        let set: std::collections::HashSet<_> = s.residents().collect();
        assert_eq!(set.len(), 27);

        for c in &set {
            assert!(s.contains(*c));
        }
        assert!(!s.contains(SubchunkCoord::new(7, -3, 0)));
        assert!(!s.contains(SubchunkCoord::new(5, -5, 0)));
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
        assert_eq!(s.center(), SubchunkCoord::new(0, 0, 0));
    }

    #[test]
    fn recenter_one_step_rolls_one_face() {
        // 3×3×3 shell → moving by +1 in X rolls 9 coords out and 9 in.
        let mut s = Shell::new([1, 1, 1], SubchunkCoord::new(0, 0, 0));
        let diff  = s.recenter(SubchunkCoord::new(1, 0, 0));

        assert_eq!(diff.added.len(),   9);
        assert_eq!(diff.removed.len(), 9);

        // Added face: x = 2 (new_center + radius).
        for c in &diff.added {
            assert_eq!(c.x, 2);
        }
        // Removed face: x = -1 (old_center - radius).
        for c in &diff.removed {
            assert_eq!(c.x, -1);
        }
    }

    #[test]
    fn recenter_far_jump_removes_old_adds_new_fully() {
        let mut s = Shell::new([1, 1, 1], SubchunkCoord::new(0, 0, 0));
        let diff  = s.recenter(SubchunkCoord::new(100, 0, 0));
        // Disjoint old and new shells: every coord in each is added/removed.
        assert_eq!(diff.added.len(),   27);
        assert_eq!(diff.removed.len(), 27);
    }

    // -- integration with SlotPool --

    #[test]
    fn shell_and_pool_roll_consistently() {
        let mut shell = Shell::new([1, 1, 1], SubchunkCoord::new(0, 0, 0));
        let mut pool: SlotPool<&'static str> = SlotPool::new(shell.pool_dims());

        // Populate initial residents.
        for c in shell.residents() {
            assert_eq!(pool.insert(c, "initial"), None);
        }
        assert_eq!(pool.occupied_count(), shell.resident_count());

        // Roll by +1 on X: evict removed, insert added.
        let diff = shell.recenter(SubchunkCoord::new(1, 0, 0));
        assert_eq!(diff.added.len(),   9);
        assert_eq!(diff.removed.len(), 9);

        for c in &diff.removed {
            assert_eq!(pool.remove(*c), Some("initial"));
        }
        for c in &diff.added {
            assert_eq!(pool.insert(*c, "rolled"), None);
        }
        assert_eq!(pool.occupied_count(), shell.resident_count());

        // Post-roll, every shell resident is stored under the correct label.
        for c in shell.residents() {
            let v = pool.get(c).copied().expect("every resident stored");
            let label = if c.x == 2 { "rolled" } else { "initial" };
            assert_eq!(v, label, "at {c:?}");
        }
    }

    #[test]
    fn shell_and_pool_far_jump_resets_cleanly() {
        // Disjoint shells: every slot is evicted and repopulated.
        let mut shell = Shell::new([1, 1, 1], SubchunkCoord::new(0, 0, 0));
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
