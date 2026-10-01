// Fire and liquid in one box: the liquid carries the air with it. Water is 800 times denser than
// air, so gas caught in it or in the spray and sheets just off it moves as the water does: a hose
// stream drags a jet of air along and punches through the flames and smoke, a falling sheet pushes
// the gas aside, and the gas over a still pool stays still. Each face of the fire's velocity grid is
// pulled toward the liquid's velocity there, as fast as the share of liquid around it. With lava in
// the box too, gas caught in the lava moves with the lava the same way.
//!include common.wgsl

struct Params {
  g: Grid,
  nl: vec4<f32>,  // liquid grid dims; w = rest density (particles per cell)
  k: vec4<f32>,   // drag rate in a face full of liquid (1/s), a second liquid (lava) on (1/0), its rest density
};

@group(0) @binding(0) var vel: texture_3d<f32>;    // the fire's face velocities (n + 1)
@group(0) @binding(1) var lvel: texture_3d<f32>;   // the liquid's face velocities (liquid grid + 1)
@group(0) @binding(2) var dens: texture_3d<f32>;   // liquid particle density (liquid grid)
@group(0) @binding(3) var dst: texture_storage_3d<${VELFMT}, write>;
@group(0) @binding(4) var lvel2: texture_3d<f32>;  // the lava's (same grid as the water's)
@group(0) @binding(5) var dens2: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn to_liquid(c: vec3<i32>, n: vec3<i32>) -> vec3<i32> {
  return min(vec3<i32>(floor((vec3<f32>(c) + vec3<f32>(0.5)) * U.nl.xyz / vec3<f32>(n))), vec3<i32>(U.nl.xyz) - vec3<i32>(1));
}

fn share(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return clamp(textureLoad(dens, to_liquid(c, n), 0).x / max(U.nl.w, 1e-3), 0.0, 1.0);
}

fn share2(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (U.k.y < 0.5 || !in_grid(c, n)) { return 0.0; }
  return clamp(textureLoad(dens2, to_liquid(c, n), 0).x / max(U.k.z, 1e-3), 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  var v = textureLoad(vel, c, 0);
  let dt = U.g.bc.w;
  // (the same face of the liquid's grid: the two grids share their layout in a fire-and-liquid box)
  let lc = min(vec3<i32>(floor(vec3<f32>(c) * U.nl.xyz / vec3<f32>(n) + vec3<f32>(0.5))), vec3<i32>(U.nl.xyz));
  let lv = textureLoad(lvel, lc, 0).xyz;
  var lv2 = vec3<f32>(0.0);
  if (U.k.y > 0.5) { lv2 = textureLoad(lvel2, lc, 0).xyz; }
  for (var k = 0; k < 3; k++) {
    var e = vec3<i32>(0);
    e[k] = 1;
    // (water: by the square of its mean share over the face, as a dispersed phase drags, so a thin film
    // on the ground or a sheet of spray barely moves the air, while the gas inside a stream moves with it;
    // lava: gas anywhere in it moves with it)
    let fm = 0.5 * (share(c - e, n) + share(c, n));
    let f = fm * fm;
    let f2 = max(share2(c - e, n), share2(c, n));
    if (f <= 0.0 && f2 <= 0.0) { continue; }
    let a = 1.0 - exp(-U.k.x * max(f, f2) * dt);
    v[k] = mix(v[k], select(lv[k], lv2[k], f2 > f), a);
  }
  textureStore(dst, c, v);
}
