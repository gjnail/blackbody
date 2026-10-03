// Adding to the frame's shared radiant sources (radiant.py): what a glowing surface radiates, onto the coarse cell it is
// in, in fixed point. The shader that includes this binds the accumulator as `RA: array<atomic<i32>>` (seven words a
// coarse cell: power, power x its place in the cell (x, y, z, 0..1), power x its outward normal (x, y, z)) and passes
// the coarse grid as a RadGrid (radiant.Radiant.uniform_block).

struct RadGrid {
  o: vec4<f32>,   // corner (fire-local m), coarse cell size (m)
  n: vec4<f32>,   // coarse cells (x, y, z), on (1/0)
};

const RAD_FX: f32 = 100.0;   // radiant.FX_P

// A surface at p radiating `power` W out along nrm (a unit vector).
fn rad_add(g: RadGrid, p: vec3<f32>, power: f32, nrm: vec3<f32>) {
  if (g.n.w < 0.5 || power <= 0.0) { return; }
  let dims = vec3<i32>(g.n.xyz);
  let q = (p - g.o.xyz) / g.o.w;
  let c = clamp(vec3<i32>(floor(q)), vec3<i32>(0), dims - vec3<i32>(1));
  let f = clamp(q - vec3<f32>(c), vec3<f32>(0.0), vec3<f32>(1.0));
  let pw = min(power * RAD_FX, 2.0e8);
  let k = 7u * u32((c.z * dims.y + c.y) * dims.x + c.x);
  atomicAdd(&RA[k], i32(round(pw)));
  atomicAdd(&RA[k + 1u], i32(round(pw * f.x)));
  atomicAdd(&RA[k + 2u], i32(round(pw * f.y)));
  atomicAdd(&RA[k + 3u], i32(round(pw * f.z)));
  atomicAdd(&RA[k + 4u], i32(round(pw * nrm.x)));
  atomicAdd(&RA[k + 5u], i32(round(pw * nrm.y)));
  atomicAdd(&RA[k + 6u], i32(round(pw * nrm.z)));
}
