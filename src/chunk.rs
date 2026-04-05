//! Chunk types and storage.
//!
//! A chunk is a 32x32x32 unit of voxel data, the fundamental building block
//! of the world. Each chunk stores occupancy (which voxels are solid) and
//! material identity (what block type each voxel is).
//!
//! The chunk owns two parallel data stores:
//! - An occupancy [`Bitmask`] (4 KB) -- one bit per voxel, the source of truth
//!   for geometry.
//! - A material [`Dense`] array (32 KB) -- one palette index per voxel, mapping
//!   through the chunk's local palette to global [`BlockId`]s.

#![allow(dead_code)]

use std::ops::Deref;

use eden_math::Vector3;

use crate::block::BlockId;
use crate::index::Linear3D;
use crate::storage::{Bitmask, Dense};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The side length of a chunk in voxels.
pub const CHUNK_SIZE: usize = 32;

/// Total number of voxels in a chunk.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// The indexing strategy for 32x32x32 chunk storage.
///
/// Linear layout: index = z * 1024 + y * 32 + x. This produces the bitmask
/// word layout `occ[z * 32 + y]` with bit position `x`, matching the data
/// layout required by the rendering pipeline.
pub type ChunkIndexer = Linear3D<CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE>;

// ---------------------------------------------------------------------------
// ChunkPos
// ---------------------------------------------------------------------------

/// A chunk-level position in world space.
///
/// Each component identifies a chunk along its axis. The world-space voxel
/// origin of a chunk is `pos * CHUNK_SIZE`.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct ChunkPos(Vector3<i32>);

impl Eq for ChunkPos {}

impl ChunkPos {
    /// Create a chunk position from integer coordinates.
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        ChunkPos(Vector3::new(x, y, z))
    }
}

impl Deref for ChunkPos {
    type Target = Vector3<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vector3<i32>> for ChunkPos {
    fn from(v: Vector3<i32>) -> Self {
        ChunkPos(v)
    }
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A 32x32x32 volume of voxel data.
///
/// Stores occupancy and material identity. The occupancy bitmask is the
/// source of truth for which voxels are solid. The material array holds
/// palette indices that map through the chunk's local palette to global
/// [`BlockId`]s.
pub struct Chunk {
    /// Per-voxel occupancy: one bit per voxel (4 KB).
    occupancy : Bitmask<ChunkIndexer>,
    /// Per-voxel material: palette index per voxel (32 KB).
    material  : Dense<ChunkIndexer, u8>,
    /// Local palette mapping `u8` indices to global block identifiers.
    palette   : Vec<BlockId>,
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk {
    /// Create an empty chunk with no occupied voxels.
    pub fn new() -> Self {
        Chunk {
            occupancy : Bitmask::new(CHUNK_VOLUME, false),
            material  : Dense::new(CHUNK_VOLUME, 0),
            palette   : Vec::new(),
        }
    }

    /// Returns the block at `pos`, or [`BlockId::AIR`] if the voxel is empty.
    pub fn get_block(&self, pos: &Vector3<u8>) -> BlockId {
        if !self.occupancy.get(pos) {
            return BlockId::AIR;
        }

        let index = *self.material.get(pos) as usize;
        self.palette[index]
    }

    /// Set the block at `pos`.
    ///
    /// Setting [`BlockId::AIR`] clears the voxel. Setting any other value
    /// marks the voxel as occupied and assigns the block type, inserting
    /// into the local palette if needed.
    ///
    /// # Panics
    ///
    /// Panics if the local palette exceeds 256 unique block types.
    pub fn set_block(&mut self, pos: &Vector3<u8>, block: BlockId) {
        if block.is_air() {
            self.occupancy.set(pos, false);
            return;
        }

        self.occupancy.set(pos, true);

        // Find or insert the block in the local palette.
        let index = match self.palette.iter().position(|&b| b == block) {
            Some(idx) => idx,
            None => {
                assert!(
                    self.palette.len() < 256,
                    "palette overflow: more than 256 unique block types in chunk",
                );
                let idx = self.palette.len();
                self.palette.push(block);
                idx
            }
        };

        self.material.set(pos, index as u8);
    }

    /// Returns whether the voxel at `pos` is occupied.
    pub fn is_occupied(&self, pos: &Vector3<u8>) -> bool {
        self.occupancy.get(pos)
    }

    /// Returns the occupancy bitmask.
    pub fn occupancy(&self) -> &Bitmask<ChunkIndexer> {
        &self.occupancy
    }

    /// Returns the material palette index array.
    pub fn material_data(&self) -> &Dense<ChunkIndexer, u8> {
        &self.material
    }

    /// Returns the local palette.
    pub fn palette(&self) -> &[BlockId] {
        &self.palette
    }

    /// Copy the occupancy bitmask into a 1024-word array.
    ///
    /// Word layout: `words[z * 32 + y]`, bit position `x`. This matches
    /// the `ChunkIndexer` linear layout expected by the build shader.
    pub fn occupancy_words(&self) -> [u32; 1024] {
        let raw     = self.occupancy.as_raw();
        let mut out = [0u32; 1024];
        out[..raw.len()].copy_from_slice(raw);
        out
    }

    /// Build a per-voxel block ID array for GPU upload.
    ///
    /// Each byte contains the raw [`BlockId`] value for the voxel at
    /// that position, resolved through the local palette. Unoccupied
    /// voxels are zero (air). Layout: `data[z * 1024 + y * 32 + x]`.
    pub fn material_block_ids(&self) -> [u8; 32768] {
        let palette_data = self.material.as_slice();
        let occ_data     = self.occupancy.as_raw();
        let mut out      = [0u8; 32768];

        for z in 0..32usize {
            for y in 0..32usize {
                let word = occ_data[z * 32 + y];
                if word == 0 {
                    continue;
                }

                let row_base = z * 1024 + y * 32;
                for x in 0..32usize {
                    if word & (1 << x) != 0 {
                        let idx = palette_data[row_base + x] as usize;
                        out[row_base + x] = self.palette[idx].raw() as u8;
                    }
                }
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- new_chunk_is_empty --

    #[test]
    fn new_chunk_is_empty() {
        let chunk = Chunk::new();

        assert_eq!(chunk.occupancy().count_ones(), 0);
        assert!(chunk.palette().is_empty());
    }

    // -- set_get_block_roundtrip --

    #[test]
    fn set_get_block_roundtrip() {
        let mut chunk = Chunk::new();

        let stone = BlockId::new(1);
        let dirt  = BlockId::new(2);
        let grass = BlockId::new(3);

        chunk.set_block(&Vector3::new(0, 0, 0), stone);
        chunk.set_block(&Vector3::new(15, 8, 20), dirt);
        chunk.set_block(&Vector3::new(31, 31, 31), grass);

        assert_eq!(chunk.get_block(&Vector3::new(0, 0, 0)), stone);
        assert_eq!(chunk.get_block(&Vector3::new(15, 8, 20)), dirt);
        assert_eq!(chunk.get_block(&Vector3::new(31, 31, 31)), grass);

        // Unset voxels return air.
        assert_eq!(chunk.get_block(&Vector3::new(1, 1, 1)), BlockId::AIR);
    }

    // -- set_air_clears_occupancy --

    #[test]
    fn set_air_clears_occupancy() {
        let mut chunk = Chunk::new();
        let pos       = Vector3::new(5, 5, 5);
        let stone     = BlockId::new(1);

        chunk.set_block(&pos, stone);
        assert!(chunk.is_occupied(&pos));

        chunk.set_block(&pos, BlockId::AIR);
        assert!(!chunk.is_occupied(&pos));
        assert_eq!(chunk.get_block(&pos), BlockId::AIR);
    }

    // -- palette_deduplication --

    #[test]
    fn palette_deduplication() {
        let mut chunk = Chunk::new();
        let stone     = BlockId::new(1);

        // Setting the same block type at multiple positions should not
        // grow the palette beyond one entry.
        chunk.set_block(&Vector3::new(0, 0, 0), stone);
        chunk.set_block(&Vector3::new(1, 0, 0), stone);
        chunk.set_block(&Vector3::new(2, 0, 0), stone);

        assert_eq!(chunk.palette().len(), 1);
        assert_eq!(chunk.palette()[0], stone);
    }

    // -- chunk_pos_equality --

    #[test]
    fn chunk_pos_equality() {
        let a = ChunkPos::new(1, 2, 3);
        let b = ChunkPos::new(1, 2, 3);
        let c = ChunkPos::new(1, 2, 4);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // -- chunk_pos_deref --

    #[test]
    fn chunk_pos_deref() {
        let pos = ChunkPos::new(10, -5, 3);

        assert_eq!(pos.x, 10);
        assert_eq!(pos.y, -5);
        assert_eq!(pos.z, 3);
    }

    // -- occupancy_words_roundtrip --

    #[test]
    fn occupancy_words_roundtrip() {
        let mut chunk = Chunk::new();
        let pos       = Vector3::new(5, 10, 15);

        chunk.set_block(&pos, BlockId::new(1));

        let words = chunk.occupancy_words();

        // Voxel (5, 10, 15) should be bit 5 in word[15 * 32 + 10].
        assert_ne!(words[15 * 32 + 10] & (1 << 5), 0);

        // All other bits in that word should be zero.
        assert_eq!(words[15 * 32 + 10] & !(1 << 5), 0);
    }

    // -- material_block_ids_resolves_palette --

    #[test]
    fn material_block_ids_resolves_palette() {
        let mut chunk = Chunk::new();
        let stone     = BlockId::new(1);
        let dirt      = BlockId::new(2);

        chunk.set_block(&Vector3::new(0, 0, 0), stone);
        chunk.set_block(&Vector3::new(1, 0, 0), dirt);

        let ids = chunk.material_block_ids();

        // The array should contain raw BlockId values, not palette indices.
        assert_eq!(ids[0], stone.raw() as u8);
        assert_eq!(ids[1], dirt.raw() as u8);

        // Unoccupied voxel should be 0 (air).
        assert_eq!(ids[2], 0);
    }
}
