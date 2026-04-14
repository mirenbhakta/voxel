// Fullscreen triangle vertex shader for the isolated sub-chunk ray cast test.
// No vertex buffer: SV_VertexID 0/1/2 generate a triangle covering the viewport.
// No GpuConsts: this shader is isolated from the ring/frame infrastructure.
//
// Coordinate convention (Vulkan NDC, Y-down):
//   vid=0 → uv=(0,0), NDC=(-1,-1)  top-left
//   vid=1 → uv=(0,2), NDC=(-1, 3)  extends beyond bottom edge
//   vid=2 → uv=(2,0), NDC=( 3,-1)  extends beyond right edge
// The resulting triangle covers the full [-1,1]×[-1,1] viewport.

void main(uint vid : SV_VertexID,
          out float4 sv_pos : SV_Position,
          out float2 uv     : TEXCOORD0) {
    uv.x = (vid == 2u) ? 2.0 : 0.0;
    uv.y = (vid == 1u) ? 2.0 : 0.0;
    sv_pos = float4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
}
