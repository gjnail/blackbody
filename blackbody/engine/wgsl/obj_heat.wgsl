// Objects' heat (objheat.py): what each point on an object's surface meets, once a frame. Just off the surface: the air
// (the gas's real temperature in a fire box, and how fast it moves past), the radiation falling on it from the shared
// radiant sources (all but its own object's), whether the liquid is against it, and lava (its temperature, how much of
// the cell it fills).
//!include common.wgsl

struct Params {
  g: vec4<f32>,    // the gas's grid corner (fire-local m), its cell size (m)
  gn: vec4<f32>,   // its cells (x, y, z), gas on (1/0)
  l: vec4<f32>,    // the liquid's grid corner, its cell size (m)
  ln: vec4<f32>,   // its cells, liquid on (1/0)
  v: vec4<f32>,    // lava's grid (the gas's) on (1/0), _, _, _
  t: vec4<f32>,    // ambient (K), the flames' real temperature (K), the hottest gas's (K), points
  w: vec4<f32>,    // the wind past the box (m/s), _
};

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<storage, read> PT: array<vec4<f32>>;   // per point: (position, owner), (outward normal, area)
@group(0) @binding(4) var<storage, read> RL: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> RLC: array<u32>;
@group(0) @binding(6) var LTYPE: texture_3d<f32>;   // the liquid's cells: 0 air, 1 liquid, 2 solid
@group(0) @binding(7) var LAVA: texture_3d<f32>;    // lava on the gas's grid: x its temperature (K), y the share of the cell
@group(0) @binding(8) var<storage, read_write> OUT: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

//!include rad_common.wgsl

fn kelvin(T: f32) -> f32 {
  let amb = U.t.x;
  return amb + (U.t.y - amb) * min(T, 1.0) + (U.t.z - U.t.y) * (1.0 - exp(-max(T - 1.0, 0.0)));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * 64u;
  if (i >= u32(U.t.w)) { return; }
  let a = PT[2u * i];
  let b = PT[2u * i + 1u];
  let p = a.xyz;
  let nrm = b.xyz;
  // the air a cell off the surface (inside the object the gas's cells are solid)
  var tk = U.t.x;
  var speed = length(U.w.xyz);
  var lava = vec2<f32>(0.0);
  if (U.gn.w > 0.5) {
    let n = U.gn.xyz;
    let q = (p + nrm * U.g.w - U.g.xyz) / U.g.w;
    if (all(q >= vec3<f32>(0.0)) && all(q <= n)) {
      tk = kelvin(max(samp_c(scal, lin, q, n).x, 0.0));
      speed = length(vel_at(vel, lin, q, n));
      if (U.v.x > 0.5) {
        let c = clamp(vec3<i32>(floor(q)), vec3<i32>(0), vec3<i32>(n) - vec3<i32>(1));
        let lv = textureLoad(LAVA, c, 0);
        lava = vec2<f32>(lv.x, lv.y);
      }
    }
  }
  // the liquid against it
  var wet = 0.0;
  if (U.ln.w > 0.5) {
    let n = vec3<i32>(U.ln.xyz);
    let c = vec3<i32>(floor((p + nrm * 0.6 * U.l.w - U.l.xyz) / U.l.w));
    if (all(c >= vec3<i32>(0)) && all(c < n)) {
      let t = textureLoad(LTYPE, c, 0).x;
      if (t > 0.5 && t < 1.5) { wet = 1.0; }
    }
  }
  let e = rad_irradiance(p, nrm, a.w, false);
  OUT[2u * i] = vec4<f32>(tk, speed, e, wet);
  OUT[2u * i + 1u] = vec4<f32>(lava, 0.0, 0.0);
}
