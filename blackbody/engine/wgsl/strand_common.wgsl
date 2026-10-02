// Grass and plants (engine/strands.py): what the strand kernels share.
//
// Each blade is a chain of POINTS points from its root (fixed where it grows) to its tip. At rest it stands up
// from its root along `up` and arches over toward its lean's azimuth, more toward the tip (rest_dir). It is bent
// back toward that shape segment by segment from the root (follow the leader: each segment's rest direction is
// turned with its parent), every segment keeping its length.
//
// Buffers, per blade: BL two vec4s (root x, z (fire-local m), patch, a random 0..1; height (m), width (m), the
// azimuth its flat face looks along, the azimuth it leans toward); RT its root (xyz) and whether it grows (1/0);
// RN its up (unit); ST (heat toward catching 0..1, burnt share (0..1 burning; 1..2 the stubble cooling),
// burning (1/0), _). Per point: X position (fire-local m), XP where it was a substep ago.

const POINTS: u32 = 5u;          // matches strands.POINTS
const MAX_PATCHES: u32 = 8u;     // matches strands.MAX_PATCHES
const STUBBLE: f32 = 0.15;       // matches strands.STUBBLE: what is left of a blade once it has burnt

struct Patch {
  fresh: vec4<f32>,   // its colour fresh (linear), translucency
  dry: vec4<f32>,     // its colour dry, dryness (0..1)
  phys: vec4<f32>,    // bending (share of the way back to its rest shape a substep, at the root), lean (rad), drag (1/m), gravity (share of g)
  burn: vec4<f32>,    // burns (1/0), catches at (field temperature), time to catch in the flame (s), burn time (s)
  form: vec4<f32>,    // blade thickness (m), ear (1/0), canopy extinction (1/m), height (m)
  base: vec4<f32>,    // its foot's height (fire-local m), grows on objects too (1/0), smoke per fuel, _
};

// A blade's rest direction for segment k (0 .. POINTS - 2).
fn rest_dir(up: vec3<f32>, az: f32, lean: f32, k: u32) -> vec3<f32> {
  var s = vec3<f32>(cos(az), 0.0, sin(az));
  s = s - up * dot(up, s);
  let l = length(s);
  s = select(vec3<f32>(1.0, 0.0, 0.0), s / max(l, 1e-6), l > 1e-6);
  let f = (f32(k) + 0.5) / f32(POINTS - 1u);
  let a = lean * f * f;
  return up * cos(a) + s * sin(a);
}

// v turned by the shortest turn that takes unit vector a onto unit vector b.
fn turn_by(a: vec3<f32>, b: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  let c = dot(a, b);
  if (c < -0.9999) { return -v; }
  let k = cross(a, b);
  return v * c + cross(k, v) + k * (dot(k, v) / (1.0 + c));
}
