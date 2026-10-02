// Grass (engine/strands.py), once as it is set out: where each blade grows and its shape at rest.
//
// On the ground, a blade grows at its patch's height, unless something is there (no grass grows through a crate). A
// patch that grows on objects too drops each blade onto whatever is under it from high above, the ground or the
// objects that stay put (a hillside, a mound, the top of a wall), and stands it up from the slope, between the
// slope's normal and straight up.
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include strand_common.wgsl

struct Params {
  sim: vec4<f32>,      // blades, ground (1/0), ground height (fire-local m), the colliders that stay put (a bit each)
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
  patches: array<Patch, MAX_PATCHES>,
};

@group(0) @binding(0) var<storage, read> BL: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> RT: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> RN: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> X: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> XP: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> ST: array<vec4<f32>>;
@group(0) @binding(6) var atlas: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

// The objects that stay put, at p (signed distance, m).
fn fixed_sdf(p: vec3<f32>) -> f32 {
  var d = 1.0e9;
  let mask = u32(U.sim.w + 0.5);
  for (var c = 0; c < i32(U.ccnt.x); c++) {
    if (((mask >> u32(c)) & 1u) == 0u) { continue; }
    d = min(d, col_sdf(U.col[c], p));
  }
  return d;
}

fn any_sdf(p: vec3<f32>) -> f32 {
  var d = 1.0e9;
  for (var c = 0; c < i32(U.ccnt.x); c++) {
    d = min(d, col_sdf(U.col[c], p));
  }
  return d;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.sim.x)) { return; }
  let b0 = BL[2u * i];
  let b1 = BL[2u * i + 1u];
  let pa = U.patches[u32(b0.z + 0.5)];
  var root = vec3<f32>(b0.x, pa.base.x, b0.y);
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if (pa.base.y > 0.5) {
    // from high above, down to the first thing that stays put (or the patch's own height: the ground)
    var p = vec3<f32>(b0.x, pa.base.x + 60.0, b0.y);
    var hit = false;
    for (var s = 0; s < 200; s++) {
      let d = fixed_sdf(p);
      if (d < 0.002) { hit = true; break; }
      if (p.y - d <= pa.base.x) { break; }
      p.y -= d;
    }
    if (hit && p.y > pa.base.x) {
      let e = 0.01;
      let g = vec3<f32>(fixed_sdf(p + vec3<f32>(e, 0.0, 0.0)) - fixed_sdf(p - vec3<f32>(e, 0.0, 0.0)),
                        fixed_sdf(p + vec3<f32>(0.0, e, 0.0)) - fixed_sdf(p - vec3<f32>(0.0, e, 0.0)),
                        fixed_sdf(p + vec3<f32>(0.0, 0.0, e)) - fixed_sdf(p - vec3<f32>(0.0, 0.0, e)));
      let gl = length(g);
      let n = select(vec3<f32>(0.0, 1.0, 0.0), g / max(gl, 1e-9), gl > 1e-9);
      if (n.y > 0.2) {   // (not on a wall's side or under an overhang)
        root = p;
        up = normalize(mix(n, vec3<f32>(0.0, 1.0, 0.0), 0.5));
      }
    }
  }
  var alive = 1.0;
  if (any_sdf(root + up * 0.01) < 0.0) { alive = 0.0; }               // inside something
  if (U.sim.y > 0.5 && root.y < U.sim.z - 1e-4) { alive = 0.0; }       // under the ground
  RT[i] = vec4<f32>(root, alive);
  RN[i] = vec4<f32>(up, 0.0);
  // its shape at rest
  let seg = b1.x / f32(POINTS - 1u);
  let lean = pa.phys.y * (0.3 + 1.5 * b0.w * b0.w);
  var x = root;
  X[i * POINTS] = vec4<f32>(x, 0.0);
  XP[i * POINTS] = vec4<f32>(x, 0.0);
  for (var k = 0u; k < POINTS - 1u; k++) {
    x += rest_dir(up, b1.w, lean, k) * seg;
    X[i * POINTS + k + 1u] = vec4<f32>(x, 0.0);
    XP[i * POINTS + k + 1u] = vec4<f32>(x, 0.0);
  }
  ST[i] = vec4<f32>(0.0);
}
