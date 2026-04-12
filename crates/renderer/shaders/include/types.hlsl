// types.hlsl — single source of truth for GPU-visible shared type layouts.
//
// Principle 4: every shared struct is defined once here and accessed through
// namespaced helpers that operate on `RWByteAddressBuffer`. Call sites never
// compute raw offsets; every `buf.Store(offset + N * 4, value)` lives inside
// this file.
//
// Shared types are added here when real subsystems land. Currently:
// `UploadMsg` and `WatermarkSnapshot`.

#ifndef RENDERER_TYPES_HLSL
#define RENDERER_TYPES_HLSL

#include "gpu_consts.hlsl"

// -----------------------------------------------------------------------
// UploadMsg — one element of an `UploadRing<UploadMsg>` on the Rust side.
// -----------------------------------------------------------------------
struct UploadMsg {
    uint command_index; // monotonic; shaders may assert strictly increasing
    uint payload;       // opaque u32; validation binary uses it for sentinels
};

namespace UploadMsg_accessor {
    static const uint STRIDE = 8;

    uint load_command_index(ByteAddressBuffer buf, uint i) {
        return buf.Load(i * STRIDE + 0);
    }
    uint load_payload(ByteAddressBuffer buf, uint i) {
        return buf.Load(i * STRIDE + 4);
    }
    void store(RWByteAddressBuffer buf, uint i, UploadMsg m) {
        buf.Store(i * STRIDE + 0, m.command_index);
        buf.Store(i * STRIDE + 4, m.payload);
    }
}

// -----------------------------------------------------------------------
// WatermarkSnapshot — end-of-frame aggregate published via readback.
// One atomic-add + one store per subsystem per frame. See §9.3 of the plan.
// -----------------------------------------------------------------------
struct WatermarkSnapshot {
    uint frame_sentinel;       // echo of g_consts.frame_sentinel
    uint command_watermark;    // highest command index processed this frame
    uint invariant_violations; // shadow-ledger counter; >0 means crash
    uint _pad;
};

namespace WatermarkSnapshot_accessor {
    static const uint STRIDE = 16;

    void store(RWByteAddressBuffer buf, uint slot, WatermarkSnapshot s) {
        uint off = slot * STRIDE;
        buf.Store(off + 0,  s.frame_sentinel);
        buf.Store(off + 4,  s.command_watermark);
        buf.Store(off + 8,  s.invariant_violations);
        buf.Store(off + 12, s._pad);
    }
}

#endif // RENDERER_TYPES_HLSL
