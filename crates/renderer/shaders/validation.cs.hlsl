// validation.cs.hlsl — first-pass smoke shader.
//
// A 1-thread compute shader that reads the CPU-written `frame_sentinel`
// from `g_consts` (forced to binding 0/0 via `types.hlsl`) and stores it
// to a `RWByteAddressBuffer` bound at slot 1. This exists only to give
// Increment 5 something to compile through the DXC → SPIR-V → wgpu
// passthrough path; the sentinel roundtrip test in Increment 10 will
// actually observe the value on the CPU side.
//
// Plan reference: `.local/renderer_plan.md` §8.3, §9.5.

#include "include/types.hlsl"

[[vk::binding(1, 0)]] RWByteAddressBuffer g_out;

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x == 0) {
        g_out.Store(0, g_consts.frame_sentinel);
    }
}
