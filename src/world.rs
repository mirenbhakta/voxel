//! World manager for sparse chunk grids.
//!
//! The [`World`] is a spatial collection of [`Chunk`]s addressed by
//! [`ChunkPos`]. It provides chunk-level operations (insert, remove, lookup)
//! and convenience methods for working in world-space voxel coordinates.

#![allow(dead_code)]

use std::collections::HashMap;

use eden_math::Vector3;

use crate::chunk::{BlockId, Chunk, ChunkPos, CHUNK_SIZE};

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

/// Split a world-space voxel coordinate into a chunk position and local offset.
///
/// Uses Euclidean division so negative coordinates decompose correctly:
/// world position -1 maps to chunk -1, local offset 31.
pub fn decompose(world_pos: Vector3<i32>) -> (ChunkPos, Vector3<u8>) {
    let size = CHUNK_SIZE as i32;

    let chunk = ChunkPos::new(
        world_pos.x.div_euclid(size),
        world_pos.y.div_euclid(size),
        world_pos.z.div_euclid(size),
    );

    let local = Vector3::new(
        world_pos.x.rem_euclid(size) as u8,
        world_pos.y.rem_euclid(size) as u8,
        world_pos.z.rem_euclid(size) as u8,
    );

    (chunk, local)
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// A sparse collection of chunks addressed by [`ChunkPos`].
///
/// The world does not know about rendering or any derived data structures.
/// It owns source data only: occupancy and material per chunk.
pub struct World {
    /// The loaded chunks.
    chunks : HashMap<ChunkPos, Chunk>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Create an empty world with no loaded chunks.
    pub fn new() -> Self {
        World { chunks: HashMap::new() }
    }

    /// Returns a reference to the chunk at `pos`, if loaded.
    pub fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    /// Returns a mutable reference to the chunk at `pos`, if loaded.
    pub fn chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.chunks.get_mut(&pos)
    }

    /// Insert a chunk at `pos`, returning the previous chunk if one existed.
    pub fn insert_chunk(
        &mut self,
        pos   : ChunkPos,
        chunk : Chunk,
    ) -> Option<Chunk>
    {
        self.chunks.insert(pos, chunk)
    }

    /// Remove and return the chunk at `pos`, if loaded.
    pub fn remove_chunk(&mut self, pos: ChunkPos) -> Option<Chunk> {
        self.chunks.remove(&pos)
    }

    /// Returns the number of loaded chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Read the block at a world-space voxel position.
    ///
    /// Returns `None` if the containing chunk is not loaded.
    pub fn get_block(&self, world_pos: Vector3<i32>) -> Option<BlockId> {
        let (chunk_pos, local) = decompose(world_pos);

        self.chunks
            .get(&chunk_pos)
            .map(|chunk| chunk.get_block(&local))
    }

    /// Write a block at a world-space voxel position.
    ///
    /// Returns `false` if the containing chunk is not loaded.
    pub fn set_block(
        &mut self,
        world_pos : Vector3<i32>,
        block     : BlockId,
    ) -> bool
    {
        let (chunk_pos, local) = decompose(world_pos);

        match self.chunks.get_mut(&chunk_pos) {
            Some(chunk) => {
                chunk.set_block(&local, block);
                true
            }
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- decompose_positive --

    #[test]
    fn decompose_positive() {
        let (chunk, local) = decompose(Vector3::new(33, 0, 65));

        assert_eq!(*chunk, Vector3::new(1, 0, 2));
        assert_eq!(local, Vector3::new(1, 0, 1));
    }

    // -- decompose_negative --

    #[test]
    fn decompose_negative() {
        // -1 should map to chunk -1, local 31.
        let (chunk, local) = decompose(Vector3::new(-1, -1, -1));

        assert_eq!(*chunk, Vector3::new(-1, -1, -1));
        assert_eq!(local, Vector3::new(31, 31, 31));
    }

    // -- decompose_boundary --

    #[test]
    fn decompose_boundary() {
        // -32 should map to chunk -1, local 0.
        let (chunk, local) = decompose(Vector3::new(-32, 0, 0));

        assert_eq!(*chunk, Vector3::new(-1, 0, 0));
        assert_eq!(local, Vector3::new(0, 0, 0));
    }

    // -- decompose_origin --

    #[test]
    fn decompose_origin() {
        let (chunk, local) = decompose(Vector3::new(0, 0, 0));

        assert_eq!(*chunk, Vector3::new(0, 0, 0));
        assert_eq!(local, Vector3::new(0, 0, 0));
    }

    // -- world_get_set_block --

    #[test]
    fn world_get_set_block() {
        let mut world = World::new();

        // No chunk loaded: get returns None, set returns false.
        assert_eq!(world.get_block(Vector3::new(5, 5, 5)), None);
        assert!(!world.set_block(Vector3::new(5, 5, 5), BlockId::new(1)));

        // Insert a chunk at the origin.
        world.insert_chunk(ChunkPos::new(0, 0, 0), Chunk::new());

        // Set and read back.
        let stone = BlockId::new(1);
        assert!(world.set_block(Vector3::new(5, 5, 5), stone));
        assert_eq!(world.get_block(Vector3::new(5, 5, 5)), Some(stone));

        // Unset voxel in a loaded chunk returns air.
        assert_eq!(
            world.get_block(Vector3::new(10, 10, 10)),
            Some(BlockId::AIR),
        );
    }

    // -- world_negative_coordinates --

    #[test]
    fn world_negative_coordinates() {
        let mut world = World::new();

        world.insert_chunk(ChunkPos::new(-1, 0, 0), Chunk::new());

        let dirt = BlockId::new(2);
        assert!(world.set_block(Vector3::new(-1, 0, 0), dirt));
        assert_eq!(world.get_block(Vector3::new(-1, 0, 0)), Some(dirt));
    }
}
