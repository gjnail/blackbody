// Fire, water and lava in one box: water chills the lava it touches. Water draws heat out of a lava
// surface a hundred times faster than air does, so the skin of lava the water reaches goes black
// and glassy within a second or so (the crust shading reads the particles' heat), while lava a
// little way in keeps glowing: only particles near the lava's own surface lose heat to it, as they
// do to the air (liq_g2p.wgsl).
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity, the lava's rest density (particles per cell), quench rate (1/s) in full contact, the water's rest density
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;   // the lava's
@group(0) @binding(1) var lava_dens: texture_3d<f32>;
@group(0) @binding(2) var water_dens: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn water_share(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return clamp(textureLoad(water_dens, c, 0).x / max(U.k.w, 1e-3), 0.0, 1.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let n = gdim(U.g);
  let c = clamp(vec3<i32>(floor(P.p.xyz)), vec3<i32>(0), n - vec3<i32>(1));
  var wf = water_share(c, n);
  for (var j = 0; j < 6; j++) {
    var d = vec3<i32>(0);
    d[j >> 1] = select(-1, 1, (j & 1) == 1);
    wf = max(wf, water_share(c + d, n));
  }
  if (wf <= 0.0) { return; }
  let rho = interp_cell(lava_dens, P.p.xyz, n).x / max(U.k.y, 1e-3);
  let expose = 0.1 + 0.9 * (1.0 - smoothstep(0.55, 0.95, rho));
  set_heat(&P, heat_of(P) * exp(-U.k.z * wf * expose * U.g.bc.w));
  parts[i] = P;
}
