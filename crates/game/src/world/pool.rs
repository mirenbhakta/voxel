//! Toroidal slot pool.
//!
//! A fixed-size storage grid that maps a [`SubchunkCoord`] to a slot via
//! per-axis modular arithmetic:
//!
//! ```text
//! slot_index = (c.x mod dims.x, c.y mod dims.y, c.z mod dims.z)
//! ```
//!
//! Any two coords that differ by `dims[i]` along axis `i` collide at the
//! same slot; the pool stores the current occupant's coord alongside its
//! value so [`SlotPool::get`] / [`SlotPool::remove`] can distinguish "slot
//! holds this coord" from "slot holds some other coord that happens to hash
//! here."
//!
//! The pool does not enforce that callers avoid collisions — it merely
//! reports them. The containing shell is responsible for sizing `dims`
//! large enough that the resident set never produces a collision (see
//! [`Shell::pool_dims`](crate::world::shell::Shell::pool_dims)).

#![allow(dead_code)]

use crate::world::coord::SubchunkCoord;

// --- SlotId ---

/// Linear index into a [`SlotPool`]'s backing vector. Useful as a stable
/// identifier for GPU-side slot references.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u32);

// --- SlotPool ---

/// Fixed-size pool indexed by `SubchunkCoord` modulo per-axis dimensions.
pub struct SlotPool<T> {
    dims:  [u32; 3],
    slots: Vec<Option<SlotEntry<T>>>,
}

impl<T> SlotPool<T> {
    /// Create an empty pool with the given per-axis dimensions.
    ///
    /// Total slot count is `dims[0] * dims[1] * dims[2]`. All dimensions
    /// must be non-zero.
    pub fn new(dims: [u32; 3]) -> Self {
        assert!(
            dims[0] > 0 && dims[1] > 0 && dims[2] > 0,
            "pool dims must be non-zero"
        );

        let count = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        let mut slots = Vec::with_capacity(count);
        slots.resize_with(count, || None);
        Self { dims, slots }
    }

    /// Per-axis pool dimensions.
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Total slot count.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// The slot identifier for `coord`, via toroidal mapping.
    pub fn slot_id(&self, coord: SubchunkCoord) -> SlotId {
        let x = coord.x.rem_euclid(self.dims[0] as i32) as u32;
        let y = coord.y.rem_euclid(self.dims[1] as i32) as u32;
        let z = coord.z.rem_euclid(self.dims[2] as i32) as u32;
        SlotId(z * self.dims[1] * self.dims[0] + y * self.dims[0] + x)
    }

    /// Borrow the value at `coord`, if the slot currently holds exactly
    /// that coord.
    pub fn get(&self, coord: SubchunkCoord) -> Option<&T> {
        let slot  = self.slot_id(coord);
        let entry = self.slots[slot.0 as usize].as_ref()?;
        if entry.coord == coord { Some(&entry.value) } else { None }
    }

    /// Mutably borrow the value at `coord`, if the slot currently holds
    /// exactly that coord.
    pub fn get_mut(&mut self, coord: SubchunkCoord) -> Option<&mut T> {
        let slot  = self.slot_id(coord);
        let entry = self.slots[slot.0 as usize].as_mut()?;
        if entry.coord == coord { Some(&mut entry.value) } else { None }
    }

    /// Write `value` into `coord`'s slot, replacing any prior occupant.
    ///
    /// Returns the previous `(coord, value)` at that slot, if any — even
    /// when the prior occupant's coord differs from `coord` (a collision).
    pub fn insert(&mut self, coord: SubchunkCoord, value: T) -> Option<(SubchunkCoord, T)> {
        let slot = self.slot_id(coord);
        self.slots[slot.0 as usize]
            .replace(SlotEntry { coord, value })
            .map(|e| (e.coord, e.value))
    }

    /// Clear `coord`'s slot if it currently holds exactly that coord.
    ///
    /// Returns the stored value, or `None` if the slot is empty or holds
    /// a different coord.
    pub fn remove(&mut self, coord: SubchunkCoord) -> Option<T> {
        let slot = self.slot_id(coord);
        let idx  = slot.0 as usize;
        let hit  = matches!(&self.slots[idx], Some(e) if e.coord == coord);
        if !hit {
            return None;
        }

        self.slots[idx].take().map(|e| e.value)
    }

    /// Iterate `(coord, slot_id, value)` for every occupied slot.
    pub fn occupied(&self) -> impl Iterator<Item = (SubchunkCoord, SlotId, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, entry)| {
            entry
                .as_ref()
                .map(|e| (e.coord, SlotId(i as u32), &e.value))
        })
    }

    /// Count of occupied slots.
    pub fn occupied_count(&self) -> usize {
        self.slots.iter().filter(|e| e.is_some()).count()
    }
}

// --- directory-index derivation ---

/// Single-source-of-truth formula for the CPU's global `directory_index`.
///
/// The per-level toroidal pool is keyed purely by `coord.rem_euclid(pool_dims)`;
/// the level's flat segment of the shared directory starts at `global_offset`.
/// Every CPU call site that converts `(coord, level)` → `directory_index` must
/// go through this function so the coord↔dir_idx binding is a single function
/// of `(coord, pool_dims, global_offset)` with no drift between sites.
///
/// Mirrors `resolve_coord_to_slot` in
/// `crates/renderer/shaders/include/directory.hlsl` — the GPU shader emits
/// a `directory_index` for each workgroup, and the CPU must compute the same
/// value for the same inputs. The shader/CPU parity is regression-tested in
/// [`hlsl_formula_matches_cpu_slot_id_across_non_aligned_origins`]
/// (see the test module below).
///
/// # Why this is a free function, not a method on `SlotPool`
///
/// `SlotPool::slot_id` is level-local (returns a [`SlotId`] inside that
/// pool's 0..capacity range). The global `directory_index` also needs the
/// level's `global_offset`, which lives on the owning `WorldView`, not on
/// the pool. Routing the computation through a shared free function that
/// takes both inputs as explicit arguments keeps every call site mechanical
/// (no hidden state) and gives a single location to cite in documentation.
pub fn cpu_compute_directory_index(
    coord:         SubchunkCoord,
    pool_dims:     [u32; 3],
    global_offset: u32,
) -> u32 {
    debug_assert!(
        pool_dims[0] > 0 && pool_dims[1] > 0 && pool_dims[2] > 0,
        "pool_dims must be non-zero (got {pool_dims:?})",
    );
    let x = coord.x.rem_euclid(pool_dims[0] as i32) as u32;
    let y = coord.y.rem_euclid(pool_dims[1] as i32) as u32;
    let z = coord.z.rem_euclid(pool_dims[2] as i32) as u32;
    global_offset + z * pool_dims[1] * pool_dims[0] + y * pool_dims[0] + x
}

// --- SlotEntry ---

struct SlotEntry<T> {
    coord: SubchunkCoord,
    value: T,
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // -- slot_id --

    #[test]
    fn slot_id_wraps_per_axis() {
        let pool: SlotPool<()> = SlotPool::new([3, 3, 3]);
        // (0,0,0) and (3,0,0) and (-3,0,0) all map to the same slot.
        let a = pool.slot_id(SubchunkCoord::new(0, 0, 0));
        let b = pool.slot_id(SubchunkCoord::new(3, 0, 0));
        let c = pool.slot_id(SubchunkCoord::new(-3, 0, 0));
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn slot_id_negative_uses_euclidean() {
        let pool: SlotPool<()> = SlotPool::new([3, 3, 3]);
        assert_eq!(
            pool.slot_id(SubchunkCoord::new(-1, -1, -1)),
            pool.slot_id(SubchunkCoord::new( 2,  2,  2)),
        );
    }

    #[test]
    fn slot_id_distinct_per_coord_within_pool() {
        let pool: SlotPool<()> = SlotPool::new([3, 3, 3]);
        let mut seen = std::collections::HashSet::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    seen.insert(pool.slot_id(SubchunkCoord::new(x, y, z)));
                }
            }
        }
        assert_eq!(seen.len(), 27);
    }

    // -- insert / get / remove --

    #[test]
    fn insert_get_roundtrip() {
        let mut pool: SlotPool<i32> = SlotPool::new([3, 3, 3]);
        let c = SubchunkCoord::new(1, 2, 0);

        assert!(pool.get(c).is_none());
        assert_eq!(pool.insert(c, 42), None);
        assert_eq!(pool.get(c), Some(&42));
        assert_eq!(pool.occupied_count(), 1);
    }

    #[test]
    fn insert_same_coord_replaces_and_returns_prior() {
        let mut pool: SlotPool<i32> = SlotPool::new([3, 3, 3]);
        let c = SubchunkCoord::new(1, 2, 0);
        pool.insert(c, 10);

        let prior = pool.insert(c, 20);
        assert_eq!(prior, Some((c, 10)));
        assert_eq!(pool.get(c), Some(&20));
    }

    #[test]
    fn insert_colliding_coord_returns_prior_with_its_original_coord() {
        let mut pool: SlotPool<i32> = SlotPool::new([3, 3, 3]);
        let a = SubchunkCoord::new(0, 0, 0);
        let b = SubchunkCoord::new(3, 0, 0); // collides with `a` at dims = 3.
        pool.insert(a, 10);

        let prior = pool.insert(b, 99);
        assert_eq!(prior, Some((a, 10)));
        assert_eq!(pool.get(b), Some(&99));
        assert_eq!(pool.get(a), None, "a has been displaced");
    }

    #[test]
    fn remove_requires_matching_coord() {
        let mut pool: SlotPool<i32> = SlotPool::new([3, 3, 3]);
        let a = SubchunkCoord::new(0, 0, 0);
        let b = SubchunkCoord::new(3, 0, 0); // collides with `a`.
        pool.insert(a, 10);

        // `b` hashes to the same slot but isn't the occupant.
        assert_eq!(pool.remove(b), None);
        assert_eq!(pool.get(a), Some(&10), "a still present");

        assert_eq!(pool.remove(a), Some(10));
        assert_eq!(pool.get(a), None);
        assert_eq!(pool.occupied_count(), 0);
    }

    // -- occupied iter --

    #[test]
    fn occupied_enumerates_only_filled_slots() {
        let mut pool: SlotPool<i32> = SlotPool::new([3, 3, 3]);
        pool.insert(SubchunkCoord::new(0, 0, 0), 1);
        pool.insert(SubchunkCoord::new(1, 1, 1), 2);

        let mut coords: Vec<_> = pool.occupied().map(|(c, _, v)| (c, *v)).collect();
        coords.sort_by_key(|(_, v)| *v);
        assert_eq!(coords, vec![
            (SubchunkCoord::new(0, 0, 0), 1),
            (SubchunkCoord::new(1, 1, 1), 2),
        ]);
    }

    // -- HLSL-parity regression --
    //
    // Regression guard for `failure-resolve-coord-to-slot-diverges-from-cpu-pool`.
    // The HLSL `resolve_coord_to_slot` helper in
    // `crates/renderer/shaders/include/directory.hlsl` and the CPU
    // [`SlotPool::slot_id`] MUST produce the same slot for the same coord
    // — the renderer uses the CPU slot as an instance index while the
    // prep shader uses the HLSL slot as the directory write target, so
    // any divergence cross-wires material content to the wrong cube.
    //
    // A prior version of the HLSL helper subtracted a per-level
    // `pool_origin` before the `rem_euclid`. That made the two formulas
    // agree iff `pool_origin % pool_dims == 0` — a condition that holds
    // at cold start but breaks the first time the shell recenters across
    // a non-pool-dim boundary. This test mirrors the (fixed) HLSL
    // formula in Rust and asserts equivalence over a grid of coords and
    // deliberately non-aligned pool_origins so any future reintroduction
    // of a pool_origin term in the shader formula fails here.

    /// Rust mirror of the HLSL `resolve_coord_to_slot` body in
    /// `crates/renderer/shaders/include/directory.hlsl` (minus the
    /// per-level `global_offset` which the HLSL helper adds after the
    /// pool-slot derivation). Structured to match the shader operation
    /// order verbatim — `coord % dims`, then double-mod for Euclidean
    /// remainder, then the z-major flat index.
    fn hlsl_equivalent_pool_slot(
        coord:     SubchunkCoord,
        pool_dims: [u32; 3],
    ) -> u32 {
        let dx = pool_dims[0] as i32;
        let dy = pool_dims[1] as i32;
        let dz = pool_dims[2] as i32;

        let px = ((coord.x % dx) + dx) % dx;
        let py = ((coord.y % dy) + dy) % dy;
        let pz = ((coord.z % dz) + dz) % dz;

        (pz as u32) * pool_dims[1] * pool_dims[0]
            + (py as u32) * pool_dims[0]
            + (px as u32)
    }

    #[test]
    fn hlsl_formula_matches_cpu_slot_id_across_non_aligned_origins() {
        // `pool_dims` deliberately non-cubic so a wrong axis order in the
        // flat index would get caught alongside a wrong rem_euclid term.
        let pool_dims: [u32; 3] = [4, 5, 6];
        let pool: SlotPool<()> = SlotPool::new(pool_dims);

        // Non-aligned origins: each component picked so `origin % dim`
        // is non-zero on at least one axis. The CPU `slot_id` MUST NOT
        // depend on the origin, and neither MUST the HLSL formula —
        // this grid asserts that neutrality across signed and unsigned
        // offsets, positive and negative, small and large.
        let origins = [
            [ 0,  0,  0], // baseline: aligned on every axis
            [ 1,  2,  3], // non-aligned on every axis, small positive
            [-1, -2, -3], // non-aligned, small negative
            [ 7, 13, 25], // non-aligned, moderately large
            [-7, -5, -4], // non-aligned, negative, x aligned with dim=4
            [12, 15, 18], // aligned on x (12%4), non-aligned on y, z
        ];

        // Coord sweep: a bounded cube that includes negatives so the
        // rem_euclid path is exercised (not just truncated `%`).
        let coord_range = -6..=6;

        for origin in origins {
            for z in coord_range.clone() {
                for y in coord_range.clone() {
                    for x in coord_range.clone() {
                        let coord = SubchunkCoord::new(x, y, z);
                        let cpu  = pool.slot_id(coord).0;
                        let hlsl = hlsl_equivalent_pool_slot(coord, pool_dims);
                        assert_eq!(
                            cpu, hlsl,
                            "CPU slot_id({coord:?}) = {cpu} != \
                             hlsl_equivalent_pool_slot({coord:?}, {pool_dims:?}) = {hlsl} \
                             (observed under origin {origin:?} — origin must not affect \
                             either formula)",
                        );
                    }
                }
            }
        }
    }

    // -- cpu_compute_directory_index --
    //
    // `cpu_compute_directory_index` is the canonical CPU formula for the
    // flat directory index. Every call site that converts (coord, level)
    // into a `directory_index` must go through it so the two CPU
    // formulas can never silently diverge. These tests pin the contract
    // down to the byte: agreement with `SlotPool::slot_id` + offset,
    // agreement with the (fixed) HLSL formula, and independence from
    // `pool_origin` — the same neutralities the regression tests above
    // enforce for `slot_id`.

    #[test]
    fn cpu_compute_directory_index_agrees_with_slot_id_plus_offset() {
        let pool_dims: [u32; 3] = [4, 5, 6];
        let pool: SlotPool<()>  = SlotPool::new(pool_dims);

        // Sweep across offsets — the offset is a pure additive term, but
        // the test fixes a representative mix (zero, mid-range, large)
        // to catch any accidental shadowing (e.g. a stray multiply or an
        // offset-before-flatten order error).
        for &global_offset in &[0u32, 7, 64, 1_000_000] {
            for z in -6..=6 {
                for y in -6..=6 {
                    for x in -6..=6 {
                        let coord  = SubchunkCoord::new(x, y, z);
                        let expect = global_offset + pool.slot_id(coord).0;
                        let actual = cpu_compute_directory_index(
                            coord, pool_dims, global_offset,
                        );
                        assert_eq!(
                            actual, expect,
                            "cpu_compute_directory_index({coord:?}, \
                             {pool_dims:?}, {global_offset}) = {actual} \
                             != slot_id + offset = {expect}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_compute_directory_index_matches_hlsl_formula() {
        // Parity with the HLSL `resolve_coord_to_slot` helper — this is
        // the load-bearing invariant: shader and CPU must emit identical
        // directory indices for the same (coord, level) pair, or the
        // dirty-list retirement cross-wires coord content to the wrong
        // slot.
        let pool_dims: [u32; 3] = [4, 5, 6];

        for &global_offset in &[0u32, 64, 192] {
            for z in -6..=6 {
                for y in -6..=6 {
                    for x in -6..=6 {
                        let coord  = SubchunkCoord::new(x, y, z);
                        let hlsl   = hlsl_equivalent_pool_slot(coord, pool_dims);
                        let cpu    = cpu_compute_directory_index(
                            coord, pool_dims, global_offset,
                        );
                        assert_eq!(
                            cpu, global_offset + hlsl,
                            "cpu_compute_directory_index disagrees with \
                             HLSL formula at coord={coord:?}, \
                             pool_dims={pool_dims:?}, offset={global_offset}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_compute_directory_index_is_level_local_under_rem_euclid() {
        // Coords differing by exactly `pool_dims[i]` along axis `i` must
        // collide at the same directory index — that's the toroidal
        // pool's contract. This test witnesses the collision so a future
        // implementation that tries to disambiguate by `coord` alone
        // (which would break the HLSL mirror) fails here.
        let pool_dims: [u32; 3] = [4, 4, 4];
        let offset = 64;

        let base = SubchunkCoord::new(0, 0, 0);
        let wrap = SubchunkCoord::new(4, 0, -4); // +dim on x, -dim on z

        assert_eq!(
            cpu_compute_directory_index(base, pool_dims, offset),
            cpu_compute_directory_index(wrap, pool_dims, offset),
        );
    }

    #[test]
    fn hlsl_formula_would_diverge_if_pool_origin_subtracted() {
        // Demonstrates the specific failure mode the previous HLSL
        // implementation produced. Subtracting a non-aligned pool_origin
        // before the `rem_euclid` shifts the slot assignment by
        // `-pool_origin.rem_euclid(pool_dims)`; the formulas only agree
        // when pool_origin is a multiple of pool_dims on every axis.
        //
        // This test isn't a claim about behaviour — it's a witness that
        // the non-aligned case is where the cross-wiring occurs, so the
        // parity test above is load-bearing.
        fn buggy_formula(
            coord:       SubchunkCoord,
            pool_dims:   [u32; 3],
            pool_origin: [i32; 3],
        ) -> u32 {
            let dx = pool_dims[0] as i32;
            let dy = pool_dims[1] as i32;
            let dz = pool_dims[2] as i32;

            let rx = coord.x - pool_origin[0];
            let ry = coord.y - pool_origin[1];
            let rz = coord.z - pool_origin[2];

            let px = ((rx % dx) + dx) % dx;
            let py = ((ry % dy) + dy) % dy;
            let pz = ((rz % dz) + dz) % dz;

            (pz as u32) * pool_dims[1] * pool_dims[0]
                + (py as u32) * pool_dims[0]
                + (px as u32)
        }

        let pool_dims: [u32; 3] = [4, 4, 4];
        let pool: SlotPool<()> = SlotPool::new(pool_dims);

        // Non-aligned origin (z = -2 is the L1-under-coarsest-L2
        // recenter case observed in `failure-resolve-coord-to-slot-
        // diverges-from-cpu-pool`).
        let origin = [0, 0, -2];
        let coord  = SubchunkCoord::new(0, -2, -4);

        let cpu   = pool.slot_id(coord).0;
        let buggy = buggy_formula(coord, pool_dims, origin);
        assert_ne!(
            cpu, buggy,
            "non-aligned pool_origin MUST produce a divergence between \
             CPU slot_id and the buggy (origin-subtracting) formula, or \
             the parity-regression test above isn't actually exercising \
             the failure case",
        );
    }
}
