//! CPU mirror of `shaders/include/worldgen.hlsl`.
//!
//! The shader-side file is the authoritative worldgen signature — a
//! pure `(coord, seed) → density` function per `decision-world-
//! streaming-architecture`, evaluated 512 times per sub-chunk per prep
//! dispatch. This module ports the same formulas bit-for-bit so we can
//! unit-test:
//!
//! 1. **Determinism.** Same `(xz, seed)` yields identical output across
//!    calls. Load-bearing for persistence: player edits are stored as a
//!    delta against `terrain_height`, which relies on the "if I
//!    regenerate the chunk, I get the same base terrain" property.
//! 2. **Seed dependence.** Different seeds produce different output at
//!    (nearly) every sample. A dead seed path would silently turn the
//!    `world_seed` field in `GpuConsts` into a no-op and collapse
//!    future hash-derived-prefab placement onto a single world.
//! 3. **Stable signature.** Hashing a fixed grid of 100 samples
//!    produces a known constant. Any unintended formula drift fires
//!    the test. This guard pattern comes from
//!    `failure-resolve-coord-to-slot-diverges-from-cpu-pool` and
//!    `failure-face-mask-bits-misaligned-and-or-folded`, both prior
//!    cases where the HLSL side drifted from its Rust twin silently.
//!
//! The CPU is not the primary consumer — the shader is. But any future
//! work that needs a height query CPU-side (collision, chunk
//! existence, editor preview, coarse-LOD OR-reduction on the host
//! path) has a guaranteed-in-sync evaluator here.

/// Integer hash: `(uint3 coord, uint seed) → u32` in the full u32 range.
///
/// Must match `hash_u32` in `shaders/include/worldgen.hlsl` operation-
/// for-operation. All arithmetic is `u32` with two's-complement
/// wraparound — Rust's `wrapping_*` explicitly, HLSL's default `uint`
/// semantics implicitly.
#[inline]
pub fn hash_u32(p: [u32; 3], seed: u32) -> u32 {
    let mut h = seed;
    h ^= p[0].wrapping_mul(0x85ebca6b);
    h = h.rotate_left(13);
    h ^= p[1].wrapping_mul(0xc2b2ae35);
    h = h.rotate_left(17);
    h ^= p[2].wrapping_mul(0x27d4eb2f);
    h = h.rotate_left(11);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= h >> 16;
    h
}

/// Scalar value in `[0, 1)` from an integer hash.
///
/// Mirrors `hash_to_unit` in the HLSL header: top 24 bits scaled by
/// `1 / 2^24`.
#[inline]
pub fn hash_to_unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Perlin quintic fade: `6t^5 - 15t^4 + 10t^3`.
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 2D value noise in `[0, 1)`.
///
/// Samples `hash_u32` at the four integer lattice corners surrounding
/// `p` and bilinearly interpolates with quintic fade weights. The
/// `z` axis is pinned to 0 for the 2D field; matches HLSL's
/// `value_noise_2d`.
#[inline]
pub fn value_noise_2d(p: [f32; 2], seed: u32) -> f32 {
    // Mirror HLSL `floor(p)` + `int2(pi)` then cast to uint.
    // Negative coordinates land on negative `i32` values which cast to
    // large `u32` under two's complement — identical bit pattern to
    // HLSL `uint(int_val)`.
    let pix = p[0].floor();
    let piy = p[1].floor();
    let pfx = p[0] - pix;
    let pfy = p[1] - piy;
    let ix = pix as i32;
    let iy = piy as i32;

    let c00 = [ix as u32, iy as u32, 0];
    let c10 = [ix.wrapping_add(1) as u32, iy as u32, 0];
    let c01 = [ix as u32, iy.wrapping_add(1) as u32, 0];
    let c11 = [ix.wrapping_add(1) as u32, iy.wrapping_add(1) as u32, 0];

    let v00 = hash_to_unit(hash_u32(c00, seed));
    let v10 = hash_to_unit(hash_u32(c10, seed));
    let v01 = hash_to_unit(hash_u32(c01, seed));
    let v11 = hash_to_unit(hash_u32(c11, seed));

    let u = fade(pfx);
    let v = fade(pfy);

    let x0 = lerp(v00, v10, u);
    let x1 = lerp(v01, v11, u);
    lerp(x0, x1, v)
}

/// Fractal Brownian motion: sum value_noise_2d across `octaves` with
/// doubling frequency and halving amplitude. Mirrors HLSL `fbm_2d`.
///
/// The per-octave seed rotation matches the HLSL formula exactly:
/// `octave_seed = octave_seed * 0x9E3779B1 + 0x7F4A7C15` (wrapping).
#[inline]
pub fn fbm_2d(p: [f32; 2], seed: u32, octaves: u32) -> f32 {
    let mut total = 0.0_f32;
    let mut amplitude = 1.0_f32;
    let mut norm = 0.0_f32;
    let mut sample_p = p;
    let mut octave_seed = seed;

    for _ in 0..octaves {
        total += value_noise_2d(sample_p, octave_seed) * amplitude;
        norm += amplitude;
        sample_p[0] *= 2.0;
        sample_p[1] *= 2.0;
        amplitude *= 0.5;
        octave_seed = octave_seed
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add(0x7F4A_7C15);
    }

    total / norm
}

/// Worldgen parameters — must match the `const` block inside HLSL
/// `terrain_height`. Grouped here so the Rust mirror reads as a
/// single source of truth for anyone comparing Rust and HLSL.
pub const TERRAIN_FREQ: f32 = 0.015;
pub const TERRAIN_AMPLITUDE: f32 = 32.0;
pub const TERRAIN_BASE_OFFSET: f32 = 0.0;
pub const TERRAIN_OCTAVES: u32 = 5;

/// Terrain height in world meters at world-space `(x, z)`. Mirrors
/// HLSL `terrain_height`.
#[inline]
pub fn terrain_height(xz: [f32; 2], seed: u32) -> f32 {
    let p = [xz[0] * TERRAIN_FREQ, xz[1] * TERRAIN_FREQ];
    let n = fbm_2d(p, seed, TERRAIN_OCTAVES);
    TERRAIN_BASE_OFFSET + (n - 0.5) * TERRAIN_AMPLITUDE
}

/// Per-voxel solidity at world position `wp`. Mirrors the HLSL
/// `terrain_occupied` predicate that `subchunk_prep.cs.hlsl` evaluates
/// once per L_0 cell during prep voxelization.
#[inline]
pub fn terrain_occupied(wp: [f32; 3], seed: u32) -> bool {
    wp[1] < terrain_height([wp[0], wp[2]], seed)
}

/// Hierarchical OR-reduction of `terrain_occupied` at level `lvl` over
/// the 2^(3*lvl) L_0 cells covered by an L-voxel whose world-space lower
/// corner is `coarse_base_wp`.
///
/// Mirrors HLSL `coarse_occupied` in `subchunk_prep.cs.hlsl`. The
/// invariant this implements is the load-bearing one for cross-LOD
/// `here & ~there` (step 3 of `decision-subchunk-visibility-storage-
/// here-and-there`):
///
///   `coarse_occupied(lvl, p) == false` ⇒ every L_0 cell inside
///   `[p, p + 2^lvl)^3` evaluates `terrain_occupied` == false.
///
/// Equivalently: a coarse voxel is solid whenever any covered finer cell
/// is solid. Conservative-monotone reduction; the exposure / cull
/// downstream relies on this never under-marking solidity.
#[inline]
pub fn coarse_occupied(lvl: u32, coarse_base_wp: [f32; 3], seed: u32) -> bool {
    let extent = 1u32 << lvl;
    for dz in 0..extent {
        for dy in 0..extent {
            for dx in 0..extent {
                let wp = [
                    coarse_base_wp[0] + dx as f32 + 0.5,
                    coarse_base_wp[1] + dy as f32 + 0.5,
                    coarse_base_wp[2] + dz as f32 + 0.5,
                ];
                if terrain_occupied(wp, seed) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `hash_u32` must be a pure function — same input, same output,
    /// across every call. If this ever fails, a global was introduced.
    #[test]
    fn hash_u32_is_pure() {
        let inputs: [([u32; 3], u32); 5] = [
            ([0, 0, 0], 0),
            ([1, 2, 3], 0xDEADBEEF),
            ([u32::MAX, u32::MAX, u32::MAX], 42),
            ([0x80000000, 0, 0x7FFFFFFF], 1),
            ([123, 456, 789], 0x12345678),
        ];
        for (coord, seed) in inputs {
            let a = hash_u32(coord, seed);
            let b = hash_u32(coord, seed);
            assert_eq!(a, b, "hash_u32 not deterministic for {coord:?} {seed}");
        }
    }

    /// Different seeds must disperse output across a grid of inputs.
    /// We don't require every single sample to differ (the hash is
    /// stateless, so a rare coincidental collision is possible), but
    /// the overwhelming majority — >90% of a 100-point grid — must.
    #[test]
    fn hash_u32_responds_to_seed_change() {
        let seed_a = 0xDEAD_BEEF_u32;
        let seed_b = 0x1234_5678_u32;
        let mut differ = 0_u32;
        let mut total = 0_u32;
        for x in 0..10_u32 {
            for y in 0..10_u32 {
                total += 1;
                let a = hash_u32([x, y, 0], seed_a);
                let b = hash_u32([x, y, 0], seed_b);
                if a != b {
                    differ += 1;
                }
            }
        }
        assert!(
            differ * 10 >= total * 9,
            "hash barely depends on seed: {differ}/{total} differ"
        );
    }

    /// `hash_to_unit` output lies in `[0, 1)`. Upper bound is strict:
    /// the max 24-bit value 0xFFFFFF scaled by `1/2^24` is `1 - 2^-24`.
    #[test]
    fn hash_to_unit_is_in_half_open_unit_interval() {
        let cases = [
            0x00000000_u32,
            0x000000FF,
            0xFFFFFF00,
            0xFFFFFFFF,
            0x7FFFFFFF,
            0x80000000,
        ];
        for h in cases {
            let u = hash_to_unit(h);
            assert!((0.0..1.0).contains(&u), "hash_to_unit({h:#x}) = {u}");
        }
    }

    /// `terrain_height` is a pure function of its arguments.
    #[test]
    fn terrain_height_is_deterministic() {
        let seed = 0xDEAD_BEEF_u32;
        for x in [-128.0, -1.5, 0.0, 0.25, 17.0, 1024.0] {
            for z in [-512.0, -1.0, 0.0, 2.5, 99.0, 4096.0] {
                let a = terrain_height([x, z], seed);
                let b = terrain_height([x, z], seed);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "terrain_height not deterministic at ({x}, {z})"
                );
            }
        }
    }

    /// Seed must actually influence `terrain_height`. A 100-point grid
    /// must differ in nearly every sample between two seeds; if this
    /// ever collapses, the seed plumbing silently became a no-op.
    #[test]
    fn terrain_height_depends_on_seed() {
        let seed_a = 0xDEAD_BEEF_u32;
        let seed_b = 0x1234_5678_u32;
        let mut differ = 0_u32;
        let mut total = 0_u32;
        for ix in 0..10_i32 {
            for iz in 0..10_i32 {
                total += 1;
                let x = ix as f32 * 3.5;
                let z = iz as f32 * 3.5;
                let a = terrain_height([x, z], seed_a);
                let b = terrain_height([x, z], seed_b);
                if a.to_bits() != b.to_bits() {
                    differ += 1;
                }
            }
        }
        assert!(
            differ * 10 >= total * 9,
            "terrain_height barely depends on seed: {differ}/{total} differ"
        );
    }

    /// `terrain_height` output stays in the expected band given the
    /// amplitude + offset constants. FBM output is in `[0, 1)`, so
    /// `(n - 0.5) * amplitude` is in `[-amplitude/2, +amplitude/2)`.
    #[test]
    fn terrain_height_bounded_by_amplitude() {
        let seed = 42_u32;
        let half = TERRAIN_AMPLITUDE * 0.5;
        let lo = TERRAIN_BASE_OFFSET - half;
        let hi = TERRAIN_BASE_OFFSET + half;
        for ix in 0..20_i32 {
            for iz in 0..20_i32 {
                let x = ix as f32 * 7.0 - 70.0;
                let z = iz as f32 * 7.0 - 70.0;
                let h = terrain_height([x, z], seed);
                assert!(
                    h >= lo && h < hi,
                    "terrain_height({x}, {z}) = {h} out of [{lo}, {hi})"
                );
            }
        }
    }

    /// Stable-signature regression guard.
    ///
    /// Samples `terrain_height` on a fixed 10×10 grid at seed
    /// `0xDEADBEEF`, reduces to a single u64 via a deterministic
    /// accumulator over the f32 bit patterns, and compares to a
    /// recorded value. **Any formula change in Rust or HLSL must
    /// update this constant** — that is the point of the guard:
    /// silent drift between Rust and HLSL is loud here because the
    /// Rust side is the test subject.
    ///
    /// When the formula changes intentionally, run the test, copy the
    /// new printed value into `EXPECTED`, and commit both the formula
    /// change and the signature update atomically.
    #[test]
    fn terrain_height_stable_signature() {
        let seed = 0xDEAD_BEEF_u32;
        // fnv-1a style accumulator, u64 to give the 100 samples room
        // to accumulate without collapsing bit-pattern variety.
        let mut sig: u64 = 0xcbf2_9ce4_8422_2325;
        for ix in 0..10_i32 {
            for iz in 0..10_i32 {
                let x = ix as f32 * 4.0 - 20.0;
                let z = iz as f32 * 4.0 - 20.0;
                let h = terrain_height([x, z], seed);
                sig ^= h.to_bits() as u64;
                sig = sig.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        const EXPECTED: u64 = STABLE_SIGNATURE;
        assert_eq!(
            sig, EXPECTED,
            "terrain_height signature drifted. New value: {sig:#x}. If the \
             formula change was intentional, update EXPECTED and STABLE_SIGNATURE \
             in worldgen.rs, and confirm the HLSL side matches bit-for-bit."
        );
    }

    /// Recorded stable signature for the 10×10 grid sampled at seed
    /// `0xDEADBEEF`. Regenerate by failing the test, reading the
    /// printed `New value: ...` from the panic message, and copying it
    /// here. Formula-change drift shows up as a test failure with the
    /// new value in the panic.
    const STABLE_SIGNATURE: u64 = 0x7423_3640_B3F8_DC29;

    /// At level 0 the OR-reduction collapses to a single density sample
    /// at the L_0 cell centre. The shader's old single-sample path is
    /// `terrain_occupied(corner + 0.5)`; verify the new helper agrees
    /// bit-for-bit with that prior contract for the L=0 case across a
    /// mix of solid/empty positions.
    #[test]
    fn coarse_occupied_at_level_zero_matches_single_sample() {
        let seed = 0xDEAD_BEEF_u32;
        for ix in -4..4_i32 {
            for iy in -16..16_i32 {
                for iz in -4..4_i32 {
                    let corner = [ix as f32, iy as f32, iz as f32];
                    let centre = [corner[0] + 0.5, corner[1] + 0.5, corner[2] + 0.5];
                    let or_reduced = coarse_occupied(0, corner, seed);
                    let single = terrain_occupied(centre, seed);
                    assert_eq!(
                        or_reduced, single,
                        "L=0 OR-reduction must equal single-sample at \
                         corner={corner:?}, centre={centre:?}"
                    );
                }
            }
        }
    }

    /// Load-bearing invariant for step 3 of `decision-subchunk-
    /// visibility-storage-here-and-there`: a coarse voxel evaluates
    /// empty only when every finer cell it covers also evaluates empty.
    /// This is the property that lets cross-LOD sub/super-sample
    /// comparisons stay conservative — they may over-store material
    /// but cannot produce voids.
    ///
    /// Walk a strip of corners that crosses the heightfield surface so
    /// the test exercises both fully-empty, fully-solid, and partially-
    /// solid coarse voxels at L=1 (extent 2) and L=2 (extent 4).
    #[test]
    fn coarse_occupied_empty_implies_all_finer_cells_empty() {
        let seed = 0xDEAD_BEEF_u32;
        for lvl in 1..=2_u32 {
            let extent = 1u32 << lvl;
            for ix in -3..3_i32 {
                // y-strip wide enough to straddle the [-16, +16) m
                // amplitude band the heightfield is bounded to.
                for iy in -20..20_i32 {
                    for iz in -3..3_i32 {
                        let corner = [
                            (ix * extent as i32) as f32,
                            (iy * extent as i32) as f32,
                            (iz * extent as i32) as f32,
                        ];
                        if coarse_occupied(lvl, corner, seed) {
                            continue;
                        }
                        for dz in 0..extent {
                            for dy in 0..extent {
                                for dx in 0..extent {
                                    let wp = [
                                        corner[0] + dx as f32 + 0.5,
                                        corner[1] + dy as f32 + 0.5,
                                        corner[2] + dz as f32 + 0.5,
                                    ];
                                    assert!(
                                        !terrain_occupied(wp, seed),
                                        "lvl={lvl} corner={corner:?}: \
                                         coarse_occupied returned false but \
                                         finer cell at {wp:?} is solid"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The OR-reduction must be conservative-monotone in level: any
    /// coarse cell that is solid at a finer level remains solid when
    /// re-evaluated at a coarser level, because the coarser query
    /// strictly contains the finer cell's footprint.
    ///
    /// Concretely, if the L=0 cell at lower-corner `c` is solid then
    /// the L=1 cell containing it (corner = `floor(c / 2) * 2`) must
    /// also be solid, and the L=2 cell containing that must be too.
    #[test]
    fn coarse_occupied_is_monotone_in_level() {
        let seed = 0xDEAD_BEEF_u32;
        for ix in -10..10_i32 {
            for iy in -20..20_i32 {
                for iz in -10..10_i32 {
                    let corner_l0 = [ix as f32, iy as f32, iz as f32];
                    if !coarse_occupied(0, corner_l0, seed) {
                        continue;
                    }
                    let l1_corner = [
                        ix.div_euclid(2) as f32 * 2.0,
                        iy.div_euclid(2) as f32 * 2.0,
                        iz.div_euclid(2) as f32 * 2.0,
                    ];
                    let l2_corner = [
                        ix.div_euclid(4) as f32 * 4.0,
                        iy.div_euclid(4) as f32 * 4.0,
                        iz.div_euclid(4) as f32 * 4.0,
                    ];
                    assert!(
                        coarse_occupied(1, l1_corner, seed),
                        "L=0 cell at {corner_l0:?} solid but L=1 parent at \
                         {l1_corner:?} reports empty — invariant broken"
                    );
                    assert!(
                        coarse_occupied(2, l2_corner, seed),
                        "L=0 cell at {corner_l0:?} solid but L=2 ancestor at \
                         {l2_corner:?} reports empty — invariant broken"
                    );
                }
            }
        }
    }
}
