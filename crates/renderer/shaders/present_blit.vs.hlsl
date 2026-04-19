// present_blit.vs.hlsl — fullscreen-triangle vertex shader for the phase-1.5
// present blit pass.
//
// No vertex buffers, no bind-group inputs. The caller dispatches
// `draw(0..3, 0..1)` and this shader generates a covering fullscreen
// triangle procedurally from `SV_VertexID`, plus a UV in `[0, 1]^2` keyed
// so that texel `(0, 0)` of the source texture lands at the top-left of
// the swapchain (matching HLSL / D3D UV convention, which the pixel
// shader's `.Load` side ignores anyway — the UV is carried only for the
// Sample-based alternative and kept here so future overlays can reuse the
// same VS).
//
// The trick: three vertices at `(-1, 1)`, `(-1, -3)`, `(3, 1)` cover the
// entire clip-space [-1, 1]^2 region after rasterisation clips the
// off-screen corners. The UV is `(0, 0)`, `(0, 2)`, `(2, 0)` respectively
// — after interpolation across the visible quad, UVs land in `[0, 1]^2`.
// Encoded via a single bitmask trick on `vid`.

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut main(uint vid : SV_VertexID) {
    VSOut o;
    float2 xy = float2((vid << 1) & 2, vid & 2);
    o.uv  = xy;
    o.pos = float4(xy * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}
