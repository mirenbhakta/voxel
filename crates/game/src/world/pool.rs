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
}
