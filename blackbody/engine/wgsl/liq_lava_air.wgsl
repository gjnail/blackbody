// Heat haze over a molten liquid: the air its surface heats, on the simulation grid, for the heat haze pass
// (haze_opl.wgsl) to trace as it traces a fire's hot air. x: how much hotter than the air around it is,
// as a share of the hottest (the air over bared melt, at the surface).
//
// A hot surface heats the air touching it and the air rises off it in a plume: hottest at the surface,
// falling off with height over a few tens of centimetres, and spreading a little as it rises. Free
// convection carries heat off a surface as its excess temperature to the 4/3 power, so a skinned-over
// surface a few hundred degrees cooler than the melt heats the air far less, and an old crust hardly.
// Inside the liquid there is no air.
//!include common.wgsl

struct Params {
  n: vec4<f32>,      // simulation grid dims, metres per cell
  nf: vec4<f32>,     // surface grid dims, surface cells per grid cell
  air: vec4<f32>,    // the plume's height scale (m), spread with height (m per m), _, _
  lava: array<vec4<f32>, 8>,
  lava_bb: array<vec4<f32>, 32>,
};

@group(0) @binding(0) var surf_t: texture_3d<f32>;
@group(0) @binding(1) var heat_t: texture_3d<f32>;
@group(0) @binding(2) var crust_t: texture_3d<f32>;   // liq_crust_adv.wgsl: .w the skin's age (s)
@group(0) @binding(3) var out_air: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

//!include liq_lava_common.wgsl

fn heat_node(c: vec3<i32>) -> f32 {
  let h = textureLoad(heat_t, c, 0);
  if (h.z > 0.02) { return clamp(h.y / h.z, 0.0, 1.0); }
  return clamp(h.x, 0.0, 1.0);
}

// The top surface of the column at surface-grid (x, z) below height y0 (surface cells): its height (surface
// cells; -1 for none) and how strongly it heats the air over it (0..1).
fn column(x: i32, z: i32, y0: i32) -> vec2<f32> {
  let ni = vec3<i32>(U.nf.xyz);
  if (x < 0 || z < 0 || x >= ni.x || z >= ni.z) { return vec2<f32>(-1.0, 0.0); }
  for (var y = min(y0, ni.y - 1); y >= 0; y--) {
    let phi = textureLoad(surf_t, vec3<i32>(x, y, z), 0).x;
    if (phi >= 0.0) { continue; }
    // the surface: between this node (inside) and the one above
    let up = textureLoad(surf_t, vec3<i32>(x, min(y + 1, ni.y - 1), z), 0).x;
    let ys = f32(y) + clamp(-phi / max(up - phi, 1e-4), 0.0, 1.0);
    let hs = heat_node(vec3<i32>(x, y, z));
    let hc = max(hs, max(heat_node(vec3<i32>(x, max(y - 2, 0), z)), heat_node(vec3<i32>(x, max(y - 4, 0), z))));
    var tau = lava_tau_from_heat(hs);
    if (U.lava[3].z > 0.5) {
      let nc = vec3<i32>(textureDimensions(crust_t));
      let pc = vec3<i32>((vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5)) * (U.n.xyz / U.nf.xyz));
      tau = textureLoad(crust_t, clamp(pc, vec3<i32>(0), nc - vec3<i32>(1)), 0).w;
    }
    // the surface's mean temperature: the melt under a skin of its age (liq_lava_shade.wgsl), or plates
    let t = lava_temps(hc, tau);
    let Ta = U.lava[0].w;
    var Ts = t.Ts;
    if (U.lava[1].w < 0.5) {
      let tskin = max(U.lava[3].x, 1e-3);
      let k = sqrt((tau + 0.12 * tskin) / tskin) * 1.5 * mix(0.6, 1.3, clamp(U.lava[1].y, 0.0, 1.0));
      Ts = t.Tc - (t.Tc - Ta) * 0.92 * (1.0 - exp(-k * 0.8));
    }
    let q = clamp((Ts - Ta) / max(U.lava[0].z - Ta, 1.0), 0.0, 1.0);
    return vec2<f32>(ys, pow(q, 1.3333));
  }
  return vec2<f32>(-1.0, 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  if (i32(id.x) >= n.x || i32(id.y) >= n.y || i32(id.z) >= n.z) { return; }
  let fs = U.nf.w;
  let ds = U.n.w / fs;                                   // metres per surface cell
  let pf = (vec3<f32>(id) + vec3<f32>(0.5)) * fs;        // this cell's centre on the surface grid
  let y0 = i32(floor(pf.y));
  // the columns under it: here and a little way round (the plume spreads as it rises)
  var heat = 0.0;
  var wsum = 0.0;
  var inside = false;
  let c0 = column(i32(pf.x), i32(pf.z), y0);
  let r = 2.0 + max(pf.y - max(c0.x, 0.0), 0.0) * U.air.y;   // (surface cells)
  for (var k = 0; k < 5; k++) {
    var c = c0;
    if (k > 0) {
      let a = f32(k) * 1.5707963 + 0.4;
      c = column(i32(pf.x + cos(a) * r), i32(pf.z + sin(a) * r), y0);
    }
    let w = select(0.6, 1.0, k == 0);
    wsum += w;
    if (c.x < 0.0) { continue; }
    let dy = (pf.y - c.x) * ds;                            // height over the surface (m)
    if (k == 0 && dy < 0.0) { inside = true; }
    heat += w * c.y * exp(-max(dy, 0.0) / max(U.air.x, 1e-3));
  }
  let v = select(heat / wsum, 0.0, inside);
  textureStore(out_air, vec3<i32>(id), vec4<f32>(v, 0.0, 0.0, 0.0));
}
