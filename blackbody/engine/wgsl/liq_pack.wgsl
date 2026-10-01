// Pack live particles for the frame cache and the surface builder: position as 16-bit fractions
// of the grid, velocity (m/s) and age (s) as half floats. 16 bytes per particle. With dye, its
// absorption (rgb) and scattering per metre as half floats in a parallel buffer, 8 bytes each. With
// heat (liquid_thermal.py), its frozen share and how cloudy its ice is as 16-bit fractions in a
// third, 4 bytes each.
//!include common.wgsl
//!include liq_common.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,  // slot capacity, dye on (1/0), _, ice on (1/0)
};

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> packed: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read> attr: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> pattr: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> therm: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> pice: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  let P = parts[i];
  if (!alive(P)) { return; }
  let q = clamp(P.p.xyz / vec3<f32>(gdim(U.g)), vec3<f32>(0.0), vec3<f32>(1.0));
  let slot = u32(atomicAdd(&ctr[2], 1));
  packed[slot] = vec4<u32>(pack2x16unorm(q.xy), pack2x16unorm(vec2<f32>(q.z, clamp(heat_of(P), 0.0, 1.0))),
                           pack2x16float(P.v.xy), pack2x16float(vec2<f32>(P.v.z, min(P.p.w, 60000.0))));
  if (U.k.y > 0.5) { pattr[slot] = attr[i].xy; }
  if (U.k.w > 0.5) {
    let s = therm[i];
    // drawn as ice only where it is mostly frozen: a particle part-way through melting, drifting off in the
    // meltwater, is slush at 0 C and looks like the water it is melting into
    pice[slot] = pack2x16unorm(vec2<f32>(smoothstep(0.5, 0.8, ice_of(s.x, s.y)), clamp(s.y, 0.0, 1.0)));
  }
}
