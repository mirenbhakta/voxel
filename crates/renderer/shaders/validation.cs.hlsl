// validation.cs.hlsl — first-pass smoke shader.
//
// A compute shader that reads the CPU-written `frame_sentinel` from `g_consts`
// (forced to binding 0/0 via `types.hlsl`) and stores it to a
// `RWByteAddressBuffer` bound at slot 1. Only thread 0 does the write; the
// remaining 63 threads in the workgroup are idle. Semantic behaviour is
// identical to the original 1-thread version from Increment 5.
//
// The workgroup size [numthreads(64, 1, 1)] was chosen to match a typical
// compute workload and is the value that the Increment 6 SPIR-V reflection
// test asserts against (reflect_spirv_reports_workgroup_size_for_validation_cs).
//

#include "include/types.hlsl"

[[vk::binding(1, 0)]] RWByteAddressBuffer g_out;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x == 0) {
        g_out.Store(0, g_consts.frame_sentinel);
    }
}
