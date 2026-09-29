// Surface builder, pass 1: every particle adds a smooth kernel k = (1 - (d/R)^2)^3 to the surface
// grid nodes within radius R, with its offset from the node and its velocity, and records the
// nearest-particle distance. The surface grid is finer than the simulation grid and is built in
// slabs along z so its accumulators stay small.
//
// accumulator slots per node: sum k, sum k*dx, sum k*dy, sum k*dz, sum k*vx, sum k*vy, sum k*vz,
// max(R - d) (fixed point); slots 8-10 hold whitewater (liq_surf_ww.wgsl).

const FX_K: f32 = 16777216.0;   // 2^24
const FX_D: f32 = 4194304.0;    // 2^22, offsets (surface cells)
const FX_V: f32 = 262144.0;     // 2^18, velocity (m/s)
const FX_Q: f32 = 65536.0;      // 2^16, distance (surface cells)

fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

struct Params {
  n: vec4<f32>,    // simulation grid dims; w = particle count
  nf: vec4<f32>,   // surface grid dims; w = particle radius r (surface cells)
  z: vec4<f32>,    // slab first z, slab end z (exclusive), kernel radius R (surface cells), _
  e: vec4<f32>,    // reach k of the deep test (cells): skip particles whose cells within k are all deep
};

@group(0) @binding(0) var<storage, read> packed: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> deep: array<u32>;

// A particle whose every cell within k is deep can only reach nodes the resolve marks inside anyway.
fn buried(q: vec3<f32>) -> bool {
  let n = vec3<i32>(U.n.xyz);
  let c = clamp(vec3<i32>(q * U.n.xyz), vec3<i32>(0), n - vec3<i32>(1));
  let k = i32(U.e.x);
  for (var z = -k; z <= k; z++) {
    for (var y = -k; y <= k; y++) {
      for (var x = -k; x <= k; x++) {
        let a = c + vec3<i32>(x, y, z);
        let ac = clamp(a, vec3<i32>(0), n - vec3<i32>(1));
        if (deep[u32(ac.x + n.x * (ac.y + n.y * ac.z))] == 0u) { return false; }
      }
    }
  }
  return true;
}
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let pk = packed[i];
  let q = vec3<f32>(unpack2x16unorm(pk.x), unpack2x16unorm(pk.y).x);
  let vxy = unpack2x16float(pk.z);
  let vel = vec3<f32>(vxy, unpack2x16float(pk.w).x);
  let nf = U.nf.xyz;
  let pf = q * nf;                 // surface-grid units (node centres at i + 0.5)
  let R = U.z.z;
  let z0 = i32(U.z.x);
  let z1 = i32(U.z.y);
  if (pf.z + R < f32(z0) || pf.z - R > f32(z1)) { return; }
  if (buried(q)) { return; }
  let ni = vec3<i32>(nf);
  let lo = max(vec3<i32>(ceil(pf - vec3<f32>(R + 0.5))), vec3<i32>(0, 0, z0));
  let hi = min(vec3<i32>(floor(pf + vec3<f32>(R - 0.5))), vec3<i32>(ni.x - 1, ni.y - 1, z1 - 1));
  let r2 = R * R;
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let d = pf - (vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5));
        let s2 = dot(d, d) / r2;
        if (s2 >= 1.0) { continue; }
        let t = 1.0 - s2;
        let k = t * t * t;
        let base = u32(x + ni.x * (y + ni.y * (z - z0))) * 12u;
        atomicAdd(&acc[base], i32(round(k * FX_K)));
        atomicAdd(&acc[base + 1u], i32(round(k * d.x * FX_D)));
        atomicAdd(&acc[base + 2u], i32(round(k * d.y * FX_D)));
        atomicAdd(&acc[base + 3u], i32(round(k * d.z * FX_D)));
        atomicAdd(&acc[base + 4u], i32(round(k * vel.x * FX_V)));
        atomicAdd(&acc[base + 5u], i32(round(k * vel.y * FX_V)));
        atomicAdd(&acc[base + 6u], i32(round(k * vel.z * FX_V)));
        atomicMax(&acc[base + 7u], i32(round((R - sqrt(dot(d, d))) * FX_Q)));
      }
    }
  }
}
