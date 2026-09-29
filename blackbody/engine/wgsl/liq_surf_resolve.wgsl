// Surface builder, pass 2: a signed distance per surface node (negative inside the liquid).
//
// Where particles are dense it is Zhu and Bridson's smooth surface: the distance to the
// kernel-weighted mean particle position, minus the particle radius, which irons out the
// individual particles. Where they are sparse (thin sheets, droplets, the gap between two blobs)
// that mean can fall between separate pieces and web them together, so there the surface may not
// extend beyond the union of the particle spheres. The velocity is the kernel-weighted mean.
// Also resolves the whitewater densities (spray, foam, bubbles) into their own texture, with how
// dense the liquid particles are (for the surface smoothing) in its alpha.

const FX_K: f32 = 16777216.0;
const FX_D: f32 = 4194304.0;
const FX_V: f32 = 262144.0;
const FX_Q: f32 = 65536.0;

struct Params {
  n: vec4<f32>,    // simulation grid dims; w = particle count
  nf: vec4<f32>,   // surface grid dims; w = particle radius r (surface cells)
  z: vec4<f32>,    // slab first z, slab end z (exclusive), kernel radius R (surface cells), kernel sum of the bulk
  e: vec4<f32>,    // reach of the deep test (cells)
};

@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var dst_ww: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> deep: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ni = vec3<i32>(U.nf.xyz);
  let z0 = i32(U.z.x);
  let c = vec3<i32>(i32(id.x), i32(id.y), i32(id.z) + z0);
  if (c.x >= ni.x || c.y >= ni.y || c.z >= min(ni.z, i32(U.z.y))) { return; }
  let base = u32(c.x + ni.x * (c.y + ni.y * (c.z - z0))) * 12u;
  let R = U.z.z;
  let r = U.nf.w;
  let ks = f32(acc[base]) / FX_K;
  var phi = R;
  var vel = vec3<f32>(0.0);
  var dense_out = 0.0;
  if (ks > 1e-6) {
    let off = vec3<f32>(f32(acc[base + 1u]), f32(acc[base + 2u]), f32(acc[base + 3u])) / FX_D / ks;
    let phi_zb = length(off) - r;
    let dmin = R - f32(acc[base + 7u]) / FX_Q;
    let phi_u = dmin - r;
    let dense = smoothstep(0.45 * U.z.w, 0.8 * U.z.w, ks);
    phi = mix(max(phi_zb, phi_u), phi_zb, dense);
    dense_out = smoothstep(0.2 * U.z.w, 0.4 * U.z.w, ks);
    phi = clamp(phi, -R, R);
    vel = vec3<f32>(f32(acc[base + 4u]), f32(acc[base + 5u]), f32(acc[base + 6u])) / FX_V / ks;
  }
  // deep inside the liquid (whose particles were not splatted): inside by more than a kernel radius
  let n = vec3<i32>(U.n.xyz);
  let sc = clamp(vec3<i32>((vec3<f32>(c) + vec3<f32>(0.5)) / U.nf.xyz * U.n.xyz), vec3<i32>(0), n - vec3<i32>(1));
  if (deep[u32(sc.x + n.x * (sc.y + n.y * sc.z))] != 0u) {
    phi = -R;
    dense_out = 1.0;
  }
  textureStore(dst, c, vec4<f32>(phi, vel));
  let ww = vec3<f32>(f32(acc[base + 8u]), f32(acc[base + 9u]), f32(acc[base + 10u])) / FX_K;
  textureStore(dst_ww, c, vec4<f32>(ww, dense_out));
}
