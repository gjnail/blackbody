// Sampling a cell-centred field next to moving colliders. The cells inside a solid hold nothing (the
// reaction empties them), so plain trilinear filtering next to a surface mixes that emptiness into the
// gas beside it. Behind a collider moving through smoke, and along its sides, the cells it has just
// left (or slides past) took in empty air from inside it every step, so a car driven through a bank of
// smoke cleared a tunnel through it. Here the cells inside a collider that was moving a step ago (the
// air's velocity is the collider's in them) are left out of the filter where its surface moves away from
// the gas or along it, and the others weighed up to make up for them: the gas there fills from beside.
// Where its surface moves into the gas they are kept (empty): the gas it pushes is carried on ahead of
// it, and what is left in a cell when the collider covers it is pushed out to its surface (sweep.wgsl).
// (Still walls are filtered as before, and so are things that only jitter where they lie: broken pieces
// and sand at rest are baked again every step and move a millimetre a second or so, which is no motion.)
// Needs common.wgsl. The includer declares `sdf_was`: the colliders' distance (cells; negative inside)
// on the field's grid when the field was last made, `fn solid_vel_in(c: vec3<i32>) -> vec3<f32>`: the
// air's velocity in cell c (inside a collider, the collider's), and `fn moving_speed() -> f32`: the
// slowest a collider counts as moving at (m/s; MOVING of a cell a step, solver.py), above 1e29 when
// nothing moves this step.

// Whether cell c is inside a collider that was moving.
fn in_moving(c: vec3<i32>) -> bool {
  if (textureLoad(sdf_was, c, 0).x >= 0.0) { return false; }
  let v = solid_vel_in(c);
  let m = moving_speed();
  return dot(v, v) > m * m;
}

// Whether cell c is left out: inside a collider that was moving, where its surface was not moving into the gas.
fn left_out(c: vec3<i32>) -> bool {
  if (!in_moving(c)) { return false; }
  let hi = vec3<i32>(textureDimensions(sdf_was)) - vec3<i32>(1);
  let o = vec3<i32>(0);
  let g = vec3<f32>(textureLoad(sdf_was, min(c + vec3<i32>(1, 0, 0), hi), 0).x - textureLoad(sdf_was, max(c - vec3<i32>(1, 0, 0), o), 0).x,
                    textureLoad(sdf_was, min(c + vec3<i32>(0, 1, 0), hi), 0).x - textureLoad(sdf_was, max(c - vec3<i32>(0, 1, 0), o), 0).x,
                    textureLoad(sdf_was, min(c + vec3<i32>(0, 0, 1), hi), 0).x - textureLoad(sdf_was, max(c - vec3<i32>(0, 0, 1), o), 0).x);
  let v = solid_vel_in(c);
  // (g points out of the collider: the surface nearest c moves out into the gas where v follows it)
  return dot(v, g) < 0.5 * length(v) * length(g);
}

// The cells a sample at p (cells, clamped to the domain as samp_c is) filters between: the first, and the
// fractions toward the others.
fn stencil(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  return clamp(p, vec3<f32>(0.5), n - vec3<f32>(0.5)) - vec3<f32>(0.5);
}

// Whether any of the cells a sample at p filters between is inside a collider that was moving. (The
// distance changes by at most a cell per cell, and they all lie within sqrt(3) cells of the first, so away
// from solids one load says.)
fn near_moving(p: vec3<f32>, n: vec3<f32>) -> bool {
  if (moving_speed() > 1e29) { return false; }
  let q = stencil(p, n);
  let i0 = vec3<i32>(floor(q));
  if (textureLoad(sdf_was, i0, 0).x >= 2.0) { return false; }
  let hi = vec3<i32>(n) - vec3<i32>(1);
  for (var j = 0; j < 8; j++) {
    if (in_moving(min(i0 + vec3<i32>(j & 1, (j >> 1) & 1, j >> 2), hi))) { return true; }
  }
  return false;
}

// Field t at p (cells, clamped to the domain as samp_c is), filtered between the cells not left out:
// plainly where none is, and as plainly where all of them are.
fn samp_free(t: texture_3d<f32>, s: sampler, p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
  let q = stencil(p, n);
  let plain = textureSampleLevel(t, s, (q + vec3<f32>(0.5)) / n, 0.0);
  if (!near_moving(p, n)) { return plain; }
  let i0 = vec3<i32>(floor(q));
  let f = q - floor(q);
  let hi = vec3<i32>(n) - vec3<i32>(1);
  var acc = vec4<f32>(0.0);
  var wsum = 0.0;
  var all_w = 0.0;
  for (var j = 0; j < 8; j++) {
    let o = vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    let c = min(i0 + o, hi);
    let w3 = select(vec3<f32>(1.0) - f, f, o == vec3<i32>(1));
    let w = w3.x * w3.y * w3.z;
    all_w += w;
    if (left_out(c)) { continue; }
    acc += w * textureLoad(t, c, 0);
    wsum += w;
  }
  if (wsum < 1e-4 || wsum > all_w - 1e-6) { return plain; }
  return acc / wsum;
}
