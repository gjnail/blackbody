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
  m0: vec4<f32>,   // mesh or volume: bounding box min, atlas code (meshsdf.wgsl atlas_org; negative: none)
  m1: vec4<f32>,   // mesh: bounding box max; w = surface depth (m, 0 = solid). Volume: w = its layers past the
                   // density (1: a temperature, 2: a velocity, 3: both)
  m2: vec4<f32>,   // mesh or volume: grid dims (cells); w = weight when picking an emitter for embers
  s: vec4<f32>,    // deforming mesh or changing volume: next frame's atlas code (negative: none), blend; z: a mesh's
                   // frames per second, a volume's velocity strength (0..1); w = volume mode (1 releases at its
                   // rates, 2 fills the box once, 3 keeps it topped up)
  r: vec4<f32>,    // orientation quaternion (x, y, z, w), applied after the yaw: an object it rides on tipped or
                   // tumbled (identity: none)
};

// Shape ids: 0 ellipsoid, 1 box, 2 cylinder (vertical), 3 capsule, 4 ring (horizontal torus), 5 cone, 6 mesh,
// 7 volume (the field of a VDB or of USD points, io/volume.py, in the mesh atlas: layer 0 density, then the
// temperature and the velocity's x, y and z when it has them, as m1.w says).

// Layer `layer` of a volume's field at q (its own frame, metres before scaling) from the grid at atlas code `code`,
// trilinear; zero outside it.
fn field_layer(q: vec3<f32>, code: f32, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, layer: i32) -> f32 {
  let cell = (m1.xyz - m0.xyz) / max(m2.xyz, vec3<f32>(1.0));
  let t = (q - m0.xyz) / cell - vec3<f32>(0.5);
  if (any(t < vec3<f32>(-0.5)) || any(t > m2.xyz - vec3<f32>(0.5))) { return 0.0; }
  let dims = vec3<i32>(m2.xyz);
  let i0 = vec3<i32>(floor(t));
  let f = t - floor(t);
  let z = atlas_org(code) + vec3<i32>(0, 0, layer * dims.z);
  let c00 = mix(atlas_at(i0, dims, z), atlas_at(i0 + vec3<i32>(1, 0, 0), dims, z), f.x);
  let c10 = mix(atlas_at(i0 + vec3<i32>(0, 1, 0), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 0), dims, z), f.x);
  let c01 = mix(atlas_at(i0 + vec3<i32>(0, 0, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 0, 1), dims, z), f.x);
  let c11 = mix(atlas_at(i0 + vec3<i32>(0, 1, 1), dims, z), atlas_at(i0 + vec3<i32>(1, 1, 1), dims, z), f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

// A volume's field at q, blended toward its next frame when it changes (an = (next frame's atlas code or -1, blend)).
fn field_at(q: vec3<f32>, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, an: vec4<f32>, layer: i32) -> f32 {
  if (m0.w < 0.0) { return 0.0; }
  let a = field_layer(q, m0.w, m0, m1, m2, layer);
  if (an.x < 0.0 || an.y <= 0.0) { return a; }
  return mix(a, field_layer(q, an.x, m0, m1, m2, layer), an.y);
}

fn volume_has_heat(e: Emitter) -> bool {
  return (i32(e.m1.w + 0.5) & 1) != 0;
}

fn volume_has_velocity(e: Emitter) -> bool {
  return (i32(e.m1.w + 0.5) & 2) != 0;
}

// v turned by the unit quaternion q (colliders.wgsl's quat_rotate: not every kernel with emitters includes it)
fn em_turn(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

// World point w in the emitter's own frame (metres from its position): undo its orientation, then its yaw.
fn em_to_local(e: Emitter, w: vec3<f32>) -> vec3<f32> {
  return yaw_to_local(em_turn(vec4<f32>(-e.r.xyz, e.r.w), w - e.a.xyz), e.k.z);
}

// A point in the emitter's own frame, in the world.
fn em_to_world(e: Emitter, q: vec3<f32>) -> vec3<f32> {
  return e.a.xyz + em_turn(e.r, yaw_to_world(q, e.k.z));
}

fn volume_local(e: Emitter, w: vec3<f32>) -> vec3<f32> {
  return em_to_local(e, w) / max(e.b.xyz, vec3<f32>(1e-4));
}

// A Volume emitter's density at world point w (about 1 where the volume is densest).
fn volume_density(e: Emitter, w: vec3<f32>) -> f32 {
  return max(field_at(volume_local(e, w), e.m0, e.m1, e.m2, e.s, 0), 0.0);
}

// Its temperature (scaled the same way), or its density when it has none.
fn volume_heat(e: Emitter, w: vec3<f32>) -> f32 {
  return max(field_at(volume_local(e, w), e.m0, e.m1, e.m2, e.s, select(0, 1, volume_has_heat(e))), 0.0);
}

// Its own velocity at world point w (m/s, world axes; scaled and turned with it); zero when it has none.
fn volume_velocity(e: Emitter, w: vec3<f32>) -> vec3<f32> {
  if (!volume_has_velocity(e)) { return vec3<f32>(0.0); }
  let q = volume_local(e, w);
  let first = select(1, 2, volume_has_heat(e));
  let v = vec3<f32>(field_at(q, e.m0, e.m1, e.m2, e.s, first), field_at(q, e.m0, e.m1, e.m2, e.s, first + 1),
                    field_at(q, e.m0, e.m1, e.m2, e.s, first + 2));
  return em_turn(e.r, yaw_to_world(v * e.b.xyz, e.k.z));
}

// How strongly a Volume emitter's own velocity counts (Velocity from the volume, s.z); 0 for any other emitter.
fn volume_vel_strength(e: Emitter) -> f32 {
  if (i32(e.a.w + 0.5) != 7 || !volume_has_velocity(e)) { return 0.0; }
  return clamp(e.s.z, 0.0, 1.0);
}

// Whether an emitter sets the velocity where it is at all: its Velocity strength, or its velocity from the volume.
fn emitter_vel_blend(e: Emitter) -> f32 {
  return max(e.d.w, volume_vel_strength(e));
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
    let d = placed_mesh_sdf_local(em_to_local(e, w), e.b.xyz, e.m0, e.m1, e.m2, e.s);
    if (e.m1.w > 0.0) { return abs(d) - e.m1.w; }
    return d;
  }
  if (shape == 7) {
    // inside where the volume is at least a quarter as dense as its densest; about a cell's worth of
    // distance per unit of density
    let cell = (e.m1.xyz - e.m0.xyz) / max(e.m2.xyz, vec3<f32>(1.0)) * max(e.b.xyz, vec3<f32>(1e-4));
    return (0.25 - min(volume_density(e, w), 1.0)) * 4.0 * min(cell.x, min(cell.y, cell.z));
  }
  let q = em_to_local(e, w);
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

// The velocity something an emitter releases starts with (embers, a liquid source's particles): its own Velocity,
// plus a volume's own velocity at its strength.
fn emitter_velocity(e: Emitter, w: vec3<f32>) -> vec3<f32> {
  var v = emitter_own_velocity(e, w);
  let s = volume_vel_strength(e);
  if (s > 0.0) { v += s * volume_velocity(e, w); }
  return v;
}

// The velocity an emitter sets the flow to where its mask is m (xyz), and how firmly (w): its own Velocity counts at
// its Velocity strength (d.w) and a volume's own velocity at its strength (s.z), each on its own, so mixed in at w
// the flow keeps 1 - w of its velocity and takes m d.w of the one and m s.z of the other.
fn emitter_vel_target(e: Emitter, w: vec3<f32>, m: f32) -> vec4<f32> {
  let a = clamp(m * e.d.w, 0.0, 1.0);
  let b = clamp(m * volume_vel_strength(e), 0.0, 1.0);
  let k = max(a, b);
  if (k <= 0.0) { return vec4<f32>(0.0); }
  // (no velocity from a volume: exactly as before, since (a own) / a is not always own in float32)
  if (b <= 0.0) { return vec4<f32>(emitter_own_velocity(e, w), a); }
  var t = vec3<f32>(0.0);
  if (a > 0.0) { t += a * emitter_own_velocity(e, w); }
  if (b > 0.0) { t += b * volume_velocity(e, w); }
  return vec4<f32>(t / k, k);
}

// Its own Velocity at w: the set velocity plus the radial push (a volume's own velocity aside).
fn emitter_own_velocity(e: Emitter, w: vec3<f32>) -> vec3<f32> {
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
