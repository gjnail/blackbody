// Fire and liquid in one box: droplets boil away in the flames. Hot gas holds very little heat for
// its volume, so it barely warms a body of water (it is the burning fuel under the water that
// boils it, both_wet.wgsl), but drops of spray and the frayed edges of a stream have so much surface
// for their water that flames boil them away as they fly through. So only sparse particles (drops
// on their own, thin spray) are freed here, at a rate that grows with the gas's heat above boiling.
// Drops that land on hot lava sizzle away too (skittering on their own steam first, as on a hot
// plate), at up to the lava rate.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // slot capacity, boiling point (fire temperature scale), rate at full heat (1/s), step seed
  nf: vec4<f32>,  // fire grid dims; w = rest density (particles per cell)
  lv: vec4<f32>,  // lava on (1/0), rate on lava at full heat (1/s), temperature of fresh lava (K)
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(3) var scal: texture_3d<f32>;
@group(0) @binding(4) var dens: texture_3d<f32>;
@group(0) @binding(5) var lava_f: texture_3d<f32>;   // the lava field (both_lava.wgsl), or a 1-cell zero
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.k.x)) { return; }
  var P = parts[i];
  if (!alive(P)) { return; }
  let nl = gdim(U.g);
  let lc = clamp(vec3<i32>(floor(P.p.xyz)), vec3<i32>(0), nl - vec3<i32>(1));
  // a drop on its own (or the edge of thin spray): the bulk of a stream or a pool does not boil off
  let rho = textureLoad(dens, lc, 0).x / max(U.nf.w, 1e-3);
  let sparse = 1.0 - smoothstep(0.15, 0.45, rho);
  if (sparse <= 0.0) { return; }
  let c = min(vec3<i32>(floor(P.p.xyz * U.nf.xyz / vec3<f32>(nl))), vec3<i32>(U.nf.xyz) - vec3<i32>(1));
  let T = textureLoad(scal, max(c, vec3<i32>(0)), 0).x;
  let boil = U.k.y;
  var rate = U.k.z * clamp((T - boil) / max(0.6 - boil, 1e-3), 0.0, 1.0);
  if (U.lv.x > 0.5) {
    // on or next to hot lava (this cell or the one under it)
    var tk = textureLoad(lava_f, max(c, vec3<i32>(0)), 0).x;
    tk = max(tk, textureLoad(lava_f, max(c - vec3<i32>(0, 1, 0), vec3<i32>(0)), 0).x);
    rate = max(rate, U.lv.y * clamp((tk - 373.15) / max(U.lv.z - 373.15, 1.0), 0.0, 1.0));
  }
  if (rate <= 0.0) { return; }
  if (rand1(i * 2654435761u + u32(U.k.w) * 97u) >= rate * sparse * U.g.bc.w) { return; }
  P.p.w = -1.0;
  parts[i] = P;
  let k = atomicAdd(&ctr[C_FREE], 1);
  freelist[u32(k)] = i;
}
