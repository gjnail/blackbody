// Surface builder, whitewater: every spray, foam and bubble particle adds a smooth kernel to its
// class's density on the surface grid (slots 8, 9, 10 of each node's accumulator). Foam fades out
// over the last quarter of its life.

const FX_K: f32 = 1048576.0;     // 2^20: room for thousands of overlapping kernels in a node without overflowing
const ACC_STRIDE: u32 = 12u;

fn lin_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

struct Params {
  n: vec4<f32>,    // simulation grid dims; w = particle count
  nf: vec4<f32>,   // surface grid dims
  z: vec4<f32>,    // slab first z, slab end z (exclusive), kernel radius (surface cells), _
};

@group(0) @binding(0) var<storage, read> packed: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> acc: array<atomic<i32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = lin_id(id, nwg);
  if (i >= u32(U.n.w)) { return; }
  let pk = packed[i];
  let b = unpack2x16unorm(pk.y);
  let q = vec3<f32>(unpack2x16unorm(pk.x), b.x);
  let code = b.y * 3.0;
  let cls = min(u32(floor(code + 1e-4)), 2u);
  let life = fract(code + 1e-4);
  var wgt = 1.0;
  if (cls == 1u) { wgt = clamp(life * 4.0, 0.0, 1.0); }
  let nf = U.nf.xyz;
  let pf = q * nf;
  let R = U.z.z;
  let z0 = i32(U.z.x);
  let z1 = i32(U.z.y);
  if (pf.z + R < f32(z0) || pf.z - R > f32(z1)) { return; }
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
        let base = u32(x + ni.x * (y + ni.y * (z - z0))) * ACC_STRIDE;
        atomicAdd(&acc[base + 8u + cls], i32(round(t * t * t * wgt * FX_K)));
      }
    }
  }
}
