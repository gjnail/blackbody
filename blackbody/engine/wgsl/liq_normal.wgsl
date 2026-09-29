// Surface tension, pass 1: outward surface normal from the particle density, with a 3D Sobel
// gradient (a derivative across one axis, smoothed along the other two) to keep particle noise out
// of it. Solid cells mirror the cell itself, which meets walls at a right angle; next to a solid,
// the ground or a closed wall the normal is then turned to the contact angle (below 90 degrees the
// liquid wets and climbs the solid, above it beads up and pulls away).
// out: xyz = outward unit normal, w = gradient strength (1/cell).
//!include common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // rest density (particles), cos and sin of the contact angle, contact angle on (1/0)
};

@group(0) @binding(0) var dens: texture_3d<f32>;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let s0 = min(textureLoad(dens, c, 0).x / U.k.x, 1.2);
  var g = vec3<f32>(0.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3<i32>(x, y, z);
        let q = c + o;
        var s = 0.0;  // outside the domain: air
        if (in_grid(q, n)) {
          s = min(textureLoad(dens, q, 0).x / U.k.x, 1.2);
          if (textureLoad(sdf, q, 0).x < 0.0) { s = s0; }
        } else if (q.y < 0 && U.g.bc.z < 0.5) {
          s = s0;  // the ground
        }
        let w = vec3<f32>(f32(2 - abs(y)) * f32(2 - abs(z)), f32(2 - abs(x)) * f32(2 - abs(z)), f32(2 - abs(x)) * f32(2 - abs(y)));
        g += vec3<f32>(f32(x), f32(y), f32(z)) * w * s;
      }
    }
  }
  g /= 32.0;
  let l = length(g);
  var nrm = vec3<f32>(0.0);
  if (l > 1e-4) { nrm = -g / l; }
  if (U.k.w > 0.5 && l > 1e-4) {
    // nearest solid surface: colliders, then the ground and closed walls
    var dw = textureLoad(sdf, c, 0).x;
    var nw = vec3<f32>(0.0, 1.0, 0.0);
    if (dw < 1.5) {
      let ex = vec3<i32>(1, 0, 0);
      let ey = vec3<i32>(0, 1, 0);
      let ez = vec3<i32>(0, 0, 1);
      let lim = n - vec3<i32>(1);
      let gs = vec3<f32>(
        textureLoad(sdf, min(c + ex, lim), 0).x - textureLoad(sdf, max(c - ex, vec3<i32>(0)), 0).x,
        textureLoad(sdf, min(c + ey, lim), 0).x - textureLoad(sdf, max(c - ey, vec3<i32>(0)), 0).x,
        textureLoad(sdf, min(c + ez, lim), 0).x - textureLoad(sdf, max(c - ez, vec3<i32>(0)), 0).x);
      if (dot(gs, gs) > 1e-8) { nw = normalize(gs); }
    }
    let cf = vec3<f32>(c) + vec3<f32>(0.5);
    let nf = vec3<f32>(n);
    if (U.g.bc.z < 0.5 && cf.y < dw) { dw = cf.y; nw = vec3<f32>(0.0, 1.0, 0.0); }
    if (U.g.bc.y < 0.5 && nf.y - cf.y < dw) { dw = nf.y - cf.y; nw = vec3<f32>(0.0, -1.0, 0.0); }
    if (U.g.bc.x < 0.5) {
      if (cf.x < dw) { dw = cf.x; nw = vec3<f32>(1.0, 0.0, 0.0); }
      if (nf.x - cf.x < dw) { dw = nf.x - cf.x; nw = vec3<f32>(-1.0, 0.0, 0.0); }
      if (cf.z < dw) { dw = cf.z; nw = vec3<f32>(0.0, 0.0, 1.0); }
      if (nf.z - cf.z < dw) { dw = nf.z - cf.z; nw = vec3<f32>(0.0, 0.0, -1.0); }
    }
    let wgt = clamp(1.5 - dw, 0.0, 1.0);
    if (wgt > 0.0) {
      var tn = nrm - dot(nrm, nw) * nw;
      if (dot(tn, tn) > 1e-6) {
        tn = normalize(tn);
        let want = nw * U.k.y + tn * U.k.z;
        nrm = normalize(mix(nrm, want, wgt));
      }
    }
  }
  textureStore(dst, c, vec4<f32>(nrm, l));
}
