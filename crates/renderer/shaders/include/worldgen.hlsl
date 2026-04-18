// worldgen.hlsl — authoritative shader-side worldgen signature.
//
// Pure `(coord, seed) → density` per `decision-world-streaming-
// architecture`. No time-based variation, no per-thread state, no
// cross-invocation reads. Safe to evaluate redundantly from any shader
// at any LOD — the result is identical for identical inputs, which is
// the load-bearing property for persistence (player edits store only
// the delta against this function) and for coarse-level generation
// (the "generate at N-1, OR-reduce once" path samples this function
// at each coarse voxel's footprint corner).
//
// Current scene is a pure 2D heightfield: `y < terrain_height(xz, seed)`
// is solid, everything above is air. No 3D noise, no caves, no
// overhangs — the sub-chunk prep shader's per-column "all above =
// air, all below = solid" shape still holds, keeping the exposure /
// is_solid / sparsity logic in `subchunk_prep.cs.hlsl` unchanged.
//
// # Companion Rust mirror
//
// `renderer::worldgen` on the Rust side ports these formulas bit-for-bit
// so we can (a) unit-test determinism + stable-signature regression,
// (b) evaluate the heightfield CPU-side if a later increment needs to
// (e.g., collision, pathfinding, chunk-existence queries). CPU/GPU
// formula drift has bitten us twice before (see `failure-resolve-coord-
// to-slot-diverges-from-cpu-pool`, `failure-face-mask-bits-misaligned-
// and-or-folded`); the mirror + its signature test is the regression
// guard. Any edit here must match the Rust file in the same commit.

#ifndef RENDERER_WORLDGEN_HLSL
#define RENDERER_WORLDGEN_HLSL

// Integer hash: `(uint3 coord, uint seed) → uint` in the full u32 range,
// used as the per-lattice-point random seed for value noise. Affine
// multiply-mix over each axis + seed, a variant of the "Hash Functions
// for GPU Rendering" (Jarzynski & Olano, 2020) pcg3d-lite mix adapted to
// one output scalar. No mod, no div, no branching — just u32 mul/xor/
// rotate/add. Deterministic across driver/hardware because all ops are
// defined on u32 with two's-complement wraparound.
uint hash_u32(uint3 p, uint seed) {
    uint h = seed;
    h ^= p.x * 0x85ebca6bu;
    h  = (h << 13) | (h >> 19);
    h ^= p.y * 0xc2b2ae35u;
    h  = (h << 17) | (h >> 15);
    h ^= p.z * 0x27d4eb2fu;
    h  = (h << 11) | (h >> 21);
    // Finaliser: xorshift + mul (xxhash32-style avalanching), keeps
    // adjacent lattice points from producing correlated low bits.
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// Scalar value in [0, 1) from the integer hash. Takes the top 24 bits
// (the most-mixed region after the finaliser) so bias from the low-bit
// finaliser doesn't leak into the fractional distribution.
float hash_to_unit(uint h) {
    return float(h >> 8) * (1.0 / 16777216.0);
}

// Smoothstep-style interpolation weight. `6t^5 - 15t^4 + 10t^3` — the
// Perlin quintic fade. Used to interpolate between lattice corners so
// the value-noise field has continuous first and second derivatives,
// which kills the grid-aligned ridges plain linear interpolation leaves
// behind.
float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 2D value noise in [0, 1). Samples the hash at the four lattice corners
// of the unit square containing `p` (integer lattice, coordinates in
// noise units) and bilinearly interpolates with quintic fade weights.
// The `z` lattice axis is pinned to 0 so 2D consumers get a stable
// 2D field; 3D consumers would call `hash_u32` with a varying z.
float value_noise_2d(float2 p, uint seed) {
    float2 pi = floor(p);
    float2 pf = p - pi;
    int2   ii = int2(pi);

    // Cast to uint for the hash: signed coords negative-of-zero and
    // large-magnitude negatives produce distinct u32 bit patterns via
    // two's-complement, which the hash mixer disperses cleanly.
    uint3 c00 = uint3(uint(ii.x    ), uint(ii.y    ), 0u);
    uint3 c10 = uint3(uint(ii.x + 1), uint(ii.y    ), 0u);
    uint3 c01 = uint3(uint(ii.x    ), uint(ii.y + 1), 0u);
    uint3 c11 = uint3(uint(ii.x + 1), uint(ii.y + 1), 0u);

    float v00 = hash_to_unit(hash_u32(c00, seed));
    float v10 = hash_to_unit(hash_u32(c10, seed));
    float v01 = hash_to_unit(hash_u32(c01, seed));
    float v11 = hash_to_unit(hash_u32(c11, seed));

    float u = fade(pf.x);
    float v = fade(pf.y);

    float x0 = lerp(v00, v10, u);
    float x1 = lerp(v01, v11, u);
    return lerp(x0, x1, v);
}

// Fractal Brownian motion: sum value_noise_2d across `octaves`, each
// octave doubling frequency and halving amplitude. Output normalised
// into approximately [0, 1) by the geometric-sum amplitude total
// (`1 + 1/2 + 1/4 + ... = 2 - 2^(1-octaves)`). Each octave gets a
// distinct per-octave seed (base seed xor'd with an octave-dependent
// constant) so the octaves don't share lattice-aligned structure.
float fbm_2d(float2 p, uint seed, uint octaves) {
    float total      = 0.0;
    float amplitude  = 1.0;
    float norm       = 0.0;
    float2 sample_p  = p;
    uint octave_seed = seed;

    [unroll(8)] for (uint i = 0u; i < octaves; ++i) {
        total       += value_noise_2d(sample_p, octave_seed) * amplitude;
        norm        += amplitude;
        sample_p    *= 2.0;
        amplitude   *= 0.5;
        // Per-octave seed rotation: multiply by an odd u32 constant so
        // successive octaves see decorrelated hash inputs. Any odd
        // multiplier works; `0x9E3779B1` is the golden-ratio-derived
        // mix used by many standard PRNGs.
        octave_seed  = octave_seed * 0x9E3779B1u + 0x7F4A7C15u;
    }

    return total / norm;
}

// Terrain height in world meters at world-space (x, z). Combines FBM
// with a scale factor and an offset that together define the current
// scene's "rolling terrain" feel.
//
//   freq          — lattice spacing in world meters per noise unit.
//                   Smaller freq ⇒ broader features. At 0.015, the base
//                   octave has a ~67 m wavelength; 5 octaves carry the
//                   cascade down to ~4 m wavelengths.
//   amplitude     — peak-to-valley vertical range. FBM output is in
//                   ~[0,1); multiplying by 32 and subtracting 16 gives
//                   a centred [-16, +16] m heightfield.
//   base_offset   — y-value at FBM = 0.5. Pins the "sea level" of the
//                   heightfield. Kept at 0 so the origin sub-chunk
//                   straddles the surface at spawn.
//   octaves       — 5. Stays under ~100 ALU per call (hash + lerp is
//                   ~40 ALU per octave; 5 octaves ≈ 200 ALU worst case,
//                   but the hash mul chain is ALU-pipelined and the
//                   actual throughput is closer to 120 ALU cycles).
float terrain_height(float2 xz, uint seed) {
    const float freq        = 0.015;
    const float amplitude   = 32.0;
    const float base_offset = 0.0;
    const uint  octaves     = 5u;
    float n = fbm_2d(xz * freq, seed, octaves);
    return base_offset + (n - 0.5) * amplitude;
}

#endif // RENDERER_WORLDGEN_HLSL
