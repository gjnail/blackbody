// Viscosity: one Jacobi sweep of the implicit diffusion (1 - a lap) u = u0 on the face velocities,
// a = nu dt / h^2. Only faces touching liquid diffuse; a neighbouring solid face lends its own
// velocity and the ground and closed walls hold it still (the liquid sticks to them), an air face
// lends nothing (the free surface is left stress-free), so thick liquid stays together as a lump,
// sags and coils instead of splashing.
//!include common.wgsl
//!include liq_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // a (nu dt / h^2), surface density (particles), ln(stiffening when cold), cooling on (1/0)
};

@group(0) @binding(0) var u0: texture_3d<f32>;
@group(0) @binding(1) var cur: texture_3d<f32>;
@group(0) @binding(2) var dens: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var heat: texture_3d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

fn heat_at(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 1.0; }
  return textureLoad(heat, c, 0).x;
}

fn liquid(c: vec3<i32>, n: vec3<i32>) -> bool {
  if (!in_grid(c, n)) { return false; }
  return textureLoad(dens, c, 0).x >= U.k.y;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  let a0 = textureLoad(u0, c, 0);
  let mask = u32(a0.w);
  var v = a0.xyz;
  for (var k = 0u; k < 3u; k++) {
    if (flag_fixed(mask, k) || any(c > face_lim(k, n))) { continue; }
    var e = vec3<i32>(0);
    e[k] = 1;
    if (!liquid(c - e, n) && !liquid(c, n)) { continue; }
    // a cooling liquid stiffens: its viscosity rises exponentially as its heat falls
    var a = U.k.x;
    var th = 1.0;
    if (U.k.w > 0.5) {
      th = 0.5 * (heat_at(c - e, n) + heat_at(c, n));
      a *= exp(U.k.z * (1.0 - th));
    }
    let lim = face_lim(k, n);
    var s = 0.0;
    var m = 0.0;
    for (var j = 0; j < 6; j++) {
      var d = vec3<i32>(0);
      d[j >> 1] = select(-1, 1, (j & 1) == 1);
      let q = c + d;
      if (any(q < vec3<i32>(0)) || any(q > lim)) {
        // past the edge of the box: a closed wall or the ground holds the liquid still (no slip), an
        // open side lets it slide
        let ax = j >> 1;
        var closed = U.g.bc.x < 0.5;
        if (ax == 1) { closed = select(U.g.bc.y < 0.5, U.g.bc.z < 0.5, d.y < 0); }
        if (closed) { m += 1.0; }
        continue;
      }
      let b = textureLoad(cur, q, 0);
      if (flag_fixed(u32(b.w), k)) {
        s += b[k];
        m += 1.0;
        continue;
      }
      if (!liquid(q - e, n) && !liquid(q, n)) { continue; }
      s += b[k];
      m += 1.0;
    }
    v[k] = (a0[k] + a * s) / (1.0 + a * m);
    if (U.k.w > 0.5) {
      // cold enough, it has set: it holds still
      v[k] *= exp(-25.0 * clamp(1.0 - th / 0.12, 0.0, 1.0) * U.g.bc.w);
    }
  }
  textureStore(dst, c, vec4<f32>(v, a0.w));
}
