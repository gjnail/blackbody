// Emitter shapes and emission masks, shared by the reaction, force, surface and particle kernels.
// Needs noise.wgsl and meshsdf.wgsl.

const MAX_EMITTERS: u32 = 16u;  // matches solver.MAX_EMITTERS

struct Emitter {
  a: vec4<f32>,    // position (world m), shape id
  b: vec4<f32>,    // size (m): radius / half extents / (ring radius, tube radius), or the mesh scale; edge softness (m)
  c: vec4<f32>,    // second point for capsules (world m), emission noise amount (0..1)
  d: vec4<f32>,    // fuel rate (1/s), temperature, smoke rate (1/s), velocity blend (0..1)
  e: vec4<f32>,    // velocity (m/s, including the emitter's own motion), radial speed (m/s)
  f: vec4<f32>,    // noise frequency (1/m), noise rise speed (m/s), seed, noise contrast
  g: vec4<f32>,    // swirl: speed at the core edge (m/s), axis x, axis z (world m), core radius (m)
  k: vec4<f32>,    // swirl reach (m), swirl base height (m), rotation about y (radians), douse rate (1/s)
  col: vec4<f32>,  // colourant released per second (rgb), water vapour in the released gas (g/m^3)
  m0: vec4<f32>,   // mesh or volume: bounding box min, atlas z offset (negative: none)
  m1: vec4<f32>,   // mesh: bounding box max; w = surface depth (m, 0 = solid). Volume: w = has a temperature layer
  m2: vec4<f32>,   // mesh or volume: grid dims (cells); w = weight when picking an emitter for embers
  s: vec4<f32>,    // deforming mesh: next frame's atlas z offset (negative: none), blend, frames per second;
                   // w = volume mode (1 releases at its rates, 2 fills the box once, 3 keeps it topped up)
};

// Shape ids: 0 ellipsoid, 1 box, 2 cylinder (vertical), 3 capsule, 4 ring (horizontal torus), 5 cone, 6 mesh,
// 7 volume (the density of a VDB, io/volume.py, in the mesh atlas: layer 0 density, layer 1 temperature).

// A volume's field at q (its own frame, metres before scaling), trilinear; zero outside it.
fn field_at(q: vec3<f32>, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, layer: i32) -> f32 {
  if (m0.w < 0.0) { return 0.0; }
  let cell = (m1.xyz - m0.xyz) / max(m2.xyz, vec3<f32>(1.0));
  let t = (q - m0.xyz) / cell - vec3<f32>(0.5);
  if (any(t < vec3<f32>(-0.5)) || any(t > m2.xyz - vec3<f32>(0.5))) { return 0.0; }
  let dims = vec3<i32>(m2.xyz);
  let i0 = vec3<i32>(floor(t));
  let f = t - floor(t);
  let z = i32(m0.w) + layer * dims.z;
  let c00 = mix(atlas_at(i0, dims, z), atlas_at(i0 + vec3<i32>(1, 0, 0), dims, z), f.x);
  let c10 = mix(atlas_at(i0 + vec3<i32>(0, 1, 0), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 0), dims, z), f.x);
  let c01 = mix(atlas_at(i0 + vec3<i32>(0, 0, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 0, 1), dims, z), f.x);
  let c11 = mix(atlas_at(i0 + vec3<i32>(0, 1, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 1), dims, z), f.x);
  return max(mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z), 0.0);
}

fn volume_local(e: Emitter, w: vec3<f32>) -> vec3<f32> {
  return yaw_to_local(w - e.a.xyz, e.k.z) / max(e.b.xyz, vec3<f32>(1e-4));
}

// A Volume emitter's density at world point w (about 1 where the volume is densest).
fn volume_density(e: Emitter, w: vec3<f32>) -> f32 {
  return field_at(volume_local(e, w), e.m0, e.m1, e.m2, 0);
}

// Its temperature (scaled the same way), or its density when it has none.
fn volume_heat(e: Emitter, w: vec3<f32>) -> f32 {
  return field_at(volume_local(e, w), e.m0, e.m1, e.m2, select(0, 1, e.m1.w > 0.5));
}
fn emitter_sdf(e: Emitter, w: vec3<f32>) -> f32 {
  let shape = i32(e.a.w + 0.5);
  if (shape == 3) {
    let q = w - e.a.xyz;
    let ba = e.c.xyz - e.a.xyz;
    let t = clamp(dot(q, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return length(q - ba * t) - e.b.x;
  }
  if (shape == 6) {
    let d = placed_mesh_sdf(w, e.a.xyz, e.b.xyz, e.k.z, e.m0, e.m1, e.m2, e.s);
    if (e.m1.w > 0.0) { return abs(d) - e.m1.w; }
    return d;
  }
  if (shape == 7) {
    // inside where the volume is at least a quarter as dense as its densest; about a cell's worth of
    // distance per unit of density
    let cell = (e.m1.xyz - e.m0.xyz) / max(e.m2.xyz, vec3<f32>(1.0)) * max(e.b.xyz, vec3<f32>(1e-4));
    return (0.25 - min(volume_density(e, w), 1.0)) * 4.0 * min(cell.x, min(cell.y, cell.z));
  }
  let q = yaw_to_local(w - e.a.xyz, e.k.z);
  if (shape == 0) {
    let r = max(e.b.xyz, vec3<f32>(1e-4));
    let k0 = length(q / r);
    let k1 = length(q / (r * r));
    if (k1 < 1e-8) { return -min(r.x, min(r.y, r.z)); }
    return k0 * (k0 - 1.0) / k1;
  }
  if (shape == 1) {
    let d = abs(q) - e.b.xyz;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
  }
  if (shape == 2) {
    let d = vec2<f32>(length(q.xz) - e.b.x, abs(q.y) - e.b.y);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
  }
  if (shape == 4) {
    let t = vec2<f32>(length(q.xz) - e.b.x, q.y);
    return length(t) - e.b.y;
  }
  if (shape == 5) {
    // cone standing on its base: radius b.x at y = -b.y, apex at y = +b.y
    let hgt = max(2.0 * e.b.y, 1e-4);
    let r = e.b.x * clamp(1.0 - (q.y + e.b.y) / hgt, 0.0, 1.0);
    let slope = hgt / sqrt(hgt * hgt + e.b.x * e.b.x);
    return max((length(q.xz) - r) * slope, abs(q.y) - e.b.y);
  }
  return 1.0e9;
}

// Soft shape mask in [0, 1].
fn emitter_mask(e: Emitter, w: vec3<f32>, h: f32) -> f32 {
  if (i32(e.a.w + 0.5) == 7) { return min(volume_density(e, w), 1.0); }
  let soft = max(e.b.w, 0.75 * h);
  let d = emitter_sdf(e, w);
  if (d >= soft) { return 0.0; }
  return 1.0 - smoothstep(-soft, soft, d);
}

// Shape mask times animated, rising patchy noise: fuel comes out in clumps, not as a smooth block.
fn emitter_weight(e: Emitter, w: vec3<f32>, h: f32, time: f32) -> f32 {
  var wgt = emitter_mask(e, w, h);
  if (wgt <= 0.0 || e.c.w <= 0.0) { return wgt; }
  let p = w * e.f.x + vec3<f32>(e.f.z * 17.13, -time * e.f.y * e.f.x, e.f.z * 5.71);
  let nz = fbm3(p, 3) * 2.0;
  let clump = clamp(0.5 + nz * e.f.w, 0.0, 1.0) * 2.0;
  return wgt * mix(1.0, clump, e.c.w);
}

fn emitter_velocity(e: Emitter, w: vec3<f32>) -> vec3<f32> {
  var v = e.e.xyz;
  if (e.e.w != 0.0) {
    var r = w - e.a.xyz;
    if (i32(e.a.w + 0.5) == 3) {
      let ba = e.c.xyz - e.a.xyz;
      let t = clamp(dot(r, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
      r = r - ba * t;
    }
    let l = length(r);
    if (l > 1e-5) { v += r / l * e.e.w; }
  }
  return v;
}

// Swirl around the emitter's vertical axis, as a Rankine vortex: solid-body rotation inside the
// core, falling off as 1/r outside it. It is driven only in a squat cylinder at the base (as wide
// and as tall as the reach), like the ambient spin a real fire whirl or dust devil draws in; the
// rising gas carries the spin up and tightens it. (Driving the whole column would suck a jet in
// through the top of the box.) Returns the tangent (x, z), the target tangential speed and a weight.
fn swirl_at(e: Emitter, w: vec3<f32>) -> vec4<f32> {
  if (e.g.x == 0.0) { return vec4<f32>(0.0); }
  let r = vec2<f32>(w.x - e.g.y, w.z - e.g.z);
  let d = length(r);
  let core = max(e.g.w, 1e-3);
  let reach = max(e.k.x, core * 1.01);
  let up = w.y - e.k.y;
  if (d >= reach || d < 1e-5 || up >= reach) { return vec4<f32>(0.0); }
  // anticlockwise seen from above is a positive rotation about +y: tangent (r.z, -r.x)
  let t = vec2<f32>(r.y, -r.x) / d;
  let sp = e.g.x * select(core / d, d / core, d < core);
  let wgt = (1.0 - smoothstep(0.5 * reach, reach, d)) * smoothstep(0.0, core, up) * (1.0 - smoothstep(0.4 * reach, reach, up));
  return vec4<f32>(t, sp, wgt);
}
