// present_blit.ps.hlsl — pixel shader for the phase-1.5 present blit.
//
// Reads the shaded-colour transient written by `subchunk_shade.cs.hlsl`
// and emits it to the swapchain attachment the caller bound. The shaded
// texture is `Rgba8Unorm` (linear); the swapchain is typically
// `Bgra8UnormSrgb`, so the hardware performs the linear→sRGB encode on
// write. No manual gamma in here.
//
// # Load, not Sample
//
// Resolution is 1:1 between the shaded transient and the swapchain — the
// graph allocates both at `(width, height)` — so there is no filtering to
// do. `.Load(int3(pos.xy, 0))` fetches texel `(pos.xy)` directly; this
// avoids binding a sampler entirely, keeping the pipeline layout
// minimal. `SV_Position.xy` is in pixel coordinates for the target, which
// matches the shaded texture one-to-one under the 1:1 assumption.
//
// # Binding
//
//   set 0 binding 0 : `g_shaded` — `Texture2D<float4>` containing the
//                     shade pass's RGBA output.
//
// No sampler binding, no other resources. Fragment output writes the
// fetched RGB with alpha forced to `1.0` — the shade pass already emits
// opaque pixels, but pinning it here keeps the present blit a pure
// overwrite regardless of future shade-stage alpha usage.

[[vk::binding(0, 0)]] Texture2D<float4> g_shaded;

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target0 {
    float4 c = g_shaded.Load(int3(int2(pos.xy), 0));
    return float4(c.rgb, 1.0);
}
