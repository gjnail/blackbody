// Shared by the fabric kernels (engine/cloth.py).

const MAX_FABRICS: u32 = 16u;   // matches cloth.MAX_FABRICS

// A fabric's placement at this substep (its pins follow it).
struct Fab {
  pos: vec4<f32>,   // position (fire-local m), rotation about y (radians)
  scl: vec4<f32>,   // scale on each axis, released (1: its pins have let go)
};

// A fabric's material (engine/cloth.py MATERIALS), in SI units.
struct Mat {
  a: vec4<f32>,  // areal density (kg/m^2), damping (1/s), friction, collision thickness (m)
  b: vec4<f32>,  // pressure drag coefficient, skin friction coefficient, ignition temperature (K), burn time (s)
  c: vec4<f32>,  // temperature it burns at (K), fuel it gives off (share of its mass), smoke per fuel, heating time constant (s)
  d: vec4<f32>,  // softening temperature (K), melting temperature (K; 0 = does not melt), shrink when soft (0..1), burnable (1/0)
  e: vec4<f32>,  // heat spread along the cloth (m^2/s), shrink when charred (0..1), flammability, _
  f: vec4<f32>,  // in water: fibre density (kg/m^3), water it holds soaked (kg per kg), time to soak (s), _
  g: vec4<f32>,  // water in it: wicking (m^2/s), share of a soaking it keeps once drained, _, _
};

// Water the cloth holds is kept in P.w (0 dry .. 1 soaked): cloth_predict.wgsl wets it under a liquid,
// cloth_finish.wgsl dries it with heat.
const LATENT: f32 = 2.26e6;     // J/kg, water boiling off
const C_WATER: f32 = 4186.0;    // J/(kg K)
const C_FIBRE: f32 = 1300.0;    // J/(kg K), dry textile fibres (cloth.C_P)

fn fabric_to_world(f: Fab, q: vec3<f32>) -> vec3<f32> {
  let s = q * f.scl.xyz;
  let cs = cos(f.pos.w);
  let sn = sin(f.pos.w);
  return f.pos.xyz + vec3<f32>(cs * s.x + sn * s.z, s.y, -sn * s.x + cs * s.z);
}

fn gone(st: vec4<f32>) -> bool { return st.y >= 1.0; }

// Spatial hash of a cell (self-collision).
fn cell_hash(c: vec3<i32>, size: u32) -> u32 {
  let h = (u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u);
  return h % size;
}
