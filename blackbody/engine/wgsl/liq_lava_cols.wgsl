// Light a molten liquid casts on its surroundings, pass 1: per column of the surface grid, the light
// its glowing surface gives off (Lambertian: its mean radiance times its area, over pi, half of it
// toward any one side) and where it comes from on average. Pass 2 (liq_lava_lights.wgsl) gathers the
// columns into a few point lights. The surface is the shell of nodes within half a cell of the zero
// of the signed distance; its underside, lying on the ground, gives nothing.
//
// cols[i] (per column, x fastest): (power rgb, _), (sum w x, sum w y, sum w z, sum w) fire-local metres,
// w = the power's luminance.
//!include common.wgsl

struct Params {
  n: vec4<f32>,      // simulation grid dims, metres per cell
  nf: vec4<f32>,     // surface grid dims, surface cells per grid cell
  org: vec4<f32>,    // grid corner (fire-local m)
  tile: vec4<f32>,   // columns per tile side, tiles along x, tiles along z
  lava: array<vec4<f32>, 8>,
  lava_bb: array<vec4<f32>, 32>,
};

@group(0) @binding(0) var surf_t: texture_3d<f32>;
@group(0) @binding(1) var heat_t: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> cols: array<vec4<f32>>;
@group(0) @binding(3) var crust_t: texture_3d<f32>;   // liq_crust_adv.wgsl: .w the skin's age (s)
@group(1) @binding(0) var<uniform> U: Params;

//!include liq_lava_common.wgsl

fn heat_node(c: vec3<i32>) -> f32 {
  let h = textureLoad(heat_t, c, 0);
  if (h.z > 0.02) { return clamp(h.y / h.z, 0.0, 1.0); }
  return clamp(h.x, 0.0, 1.0);
}

// Mean radiance of the surface: under a skin (liq_lava_shade.wgsl) its thickness varies over the surface from
// about a third to twice its mean, and the thin places give nearly all the light; plates, lava_mean_glow.
// (Here and not in liq_lava_common.wgsl, which the march includes: an edit there recompiles the march.)
fn lava_skin_glow(t: LavaT, tau: f32) -> vec3<f32> {
  if (U.lava[1].w > 0.5) { return lava_mean_glow(t); }
  let tskin = max(U.lava[3].x, 1e-3);
  let Ta = U.lava[0].w;
  let k = sqrt((tau + 0.12 * tskin) / tskin) * 1.5 * mix(0.6, 1.3, clamp(U.lava[1].y, 0.0, 1.0));
  let ms = array<f32, 4>(0.2, 0.45, 0.8, 1.4);
  var e = vec3<f32>(0.0);
  for (var i = 0; i < 4; i++) {
    let veil = 1.0 - exp(-k * ms[i]);
    e += lava_bb(t.Tc - (t.Tc - Ta) * 0.92 * veil) * 0.25;
  }
  return e;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ni = vec3<i32>(U.nf.xyz);
  let x = i32(id.x);
  let z = i32(id.y);
  if (x >= ni.x || z >= ni.z) { return; }
  let ds = U.n.w / U.nf.w;               // metres per surface cell
  let area = ds * ds;
  var power = vec3<f32>(0.0);
  var c = vec4<f32>(0.0);
  for (var y = ni.y - 1; y >= 0; y--) {
    let phi = textureLoad(surf_t, vec3<i32>(x, y, z), 0).x;
    if (abs(phi) >= 0.5) { continue; }
    let up = textureLoad(surf_t, vec3<i32>(x, min(y + 1, ni.y - 1), z), 0).x;
    let dn = textureLoad(surf_t, vec3<i32>(x, max(y - 1, 0), z), 0).x;
    if (up - dn < -0.6) { continue; }    // an underside
    let hs = heat_node(vec3<i32>(x, y, z));
    let hc = max(hs, max(heat_node(vec3<i32>(x, max(y - 2, 0), z)), heat_node(vec3<i32>(x, max(y - 4, 0), z))));
    var tau = lava_tau_from_heat(hs);
    if (U.lava[3].z > 0.5) {
      let nc = vec3<i32>(textureDimensions(crust_t));
      let pc = vec3<i32>((vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5)) * (U.n.xyz / U.nf.xyz));
      tau = textureLoad(crust_t, clamp(pc, vec3<i32>(0), nc - vec3<i32>(1)), 0).w;
    }
    var e = lava_skin_glow(lava_temps(hc, tau), tau);
    if (U.lava[1].w < 0.5) {
      // the foot of an advancing lobe glows in a broken seam (liq_lava_shade.wgsl): about half of it bared
      let gx = textureLoad(surf_t, vec3<i32>(min(x + 1, ni.x - 1), y, z), 0).x - textureLoad(surf_t, vec3<i32>(max(x - 1, 0), y, z), 0).x;
      let gz = textureLoad(surf_t, vec3<i32>(x, y, min(z + 1, ni.z - 1)), 0).x - textureLoad(surf_t, vec3<i32>(x, y, max(z - 1, 0)), 0).x;
      let nr = normalize(vec3<f32>(gx, up - dn, gz) + vec3<f32>(0.0, 1e-5, 0.0));
      let v = textureLoad(surf_t, vec3<i32>(x, y, z), 0).yzw;
      let nh = vec2<f32>(nr.x, nr.z);
      let adv = dot(v.xz, nh / max(length(nh), 1e-4));
      let hy = (f32(y) + 0.5) * ds;
      let foot = (1.0 - smoothstep(0.025, 0.07, hy)) * smoothstep(0.9, 0.5, nr.y) * smoothstep(0.004, 0.035, adv) * 0.45;
      let Ta = U.lava[0].w;
      let Tin = Ta + (U.lava[0].z - Ta) * max(hc, 0.9);
      e = mix(e, lava_bb(Tin - (Tin - Ta) * 0.1), foot);
    }
    e = e * (area * 0.5 / 3.14159265);
    let w = luma(e);
    power += e;
    let pw = U.org.xyz + (vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5)) * (U.n.xyz / U.nf.xyz) * U.n.w;
    c += vec4<f32>(pw * w, w);
  }
  let i = u32(x + ni.x * z) * 2u;
  cols[i] = vec4<f32>(power, 0.0);
  cols[i + 1u] = c;
}
