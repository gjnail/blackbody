// Fast mesh baking for dense meshes, in three passes over one voxel grid:
//   band:  each triangle writes its distance into the voxels near it (atomic min, a few cells deep)
//   cross: each row of voxels along x counts the triangles its ray crosses, signed by facing
//   final: a running sum along each row gives inside/outside; far voxels get the band depth
// Rays are nudged off the voxel centres so they do not graze shared edges and vertices.

struct Params {
  n: vec4<f32>,   // grid dims (cells), band depth (cells)
  o: vec4<f32>,   // grid min corner (mesh units), cell size
  r: vec4<f32>,   // first triangle of this slice, triangles in this slice, workgroups along x, _
};

@group(0) @binding(0) var<storage, read> tri: array<vec4<f32>>;                 // three corners per triangle
@group(0) @binding(1) var<storage, read_write> dist: array<atomic<u32>>;       // distance (float bits), per voxel
@group(0) @binding(2) var<storage, read_write> hits_buf: array<atomic<i32>>;   // signed crossings, per voxel
@group(0) @binding(3) var<storage, read_write> sdf: array<f32>;                // result, per voxel
@group(1) @binding(0) var<uniform> U: Params;

fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }

fn tri_dist2(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let ba = b - a; let pa = p - a;
  let cb = c - b; let pb = p - b;
  let ac = a - c; let pc = p - c;
  let nor = cross(ba, ac);
  let inside = sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) + sign(dot(cross(ac, nor), pc));
  if (inside < 2.0 || dot2(nor) < 1e-20) {
    let e0 = dot2(ba * clamp(dot(ba, pa) / max(dot2(ba), 1e-20), 0.0, 1.0) - pa);
    let e1 = dot2(cb * clamp(dot(cb, pb) / max(dot2(cb), 1e-20), 0.0, 1.0) - pb);
    let e2 = dot2(ac * clamp(dot(ac, pc) / max(dot2(ac), 1e-20), 0.0, 1.0) - pc);
    return min(min(e0, e1), e2);
  }
  let k = dot(nor, pa);
  return k * k / dot2(nor);
}

fn idx(c: vec3<u32>) -> u32 {
  let d = vec3<u32>(U.n.xyz);
  return c.x + c.y * d.x + c.z * d.x * d.y;
}

var<workgroup> box_lo: vec3<i32>;
var<workgroup> box_n: vec3<i32>;

// One workgroup per triangle: every voxel within the band of it.
@compute @workgroup_size(64, 1, 1)
fn band(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x + wg.y * u32(U.r.z);
  if (t >= u32(U.r.y)) { return; }
  let j = (u32(U.r.x) + t) * 3u;
  let a = tri[j].xyz;
  let b = tri[j + 1u].xyz;
  let c = tri[j + 2u].xyz;
  let dims = vec3<i32>(U.n.xyz);
  let cell = U.o.w;
  let band_cells = U.n.w;
  let lo = vec3<i32>(floor((min(a, min(b, c)) - U.o.xyz) / cell - band_cells));
  let hi = vec3<i32>(ceil((max(a, max(b, c)) - U.o.xyz) / cell + band_cells));
  let l = clamp(lo, vec3<i32>(0), dims - vec3<i32>(1));
  let h = clamp(hi, vec3<i32>(0), dims - vec3<i32>(1));
  let cnt = h - l + vec3<i32>(1);
  let total = u32(cnt.x * cnt.y * cnt.z);
  let limit = band_cells * cell;
  for (var k = li; k < total; k += 64u) {
    let v = vec3<i32>(i32(k) % cnt.x, (i32(k) / cnt.x) % cnt.y, i32(k) / (cnt.x * cnt.y)) + l;
    let p = U.o.xyz + (vec3<f32>(v) + 0.5) * cell;
    let d = sqrt(tri_dist2(p, a, b, c));
    if (d <= limit) { atomicMin(&dist[idx(vec3<u32>(v))], bitcast<u32>(d)); }
  }
}

// One thread per row of voxels along x (y, z): signed crossings of the row's ray, at the voxel just past each hit.
@compute @workgroup_size(8, 8, 1)
fn crossings(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = vec3<u32>(U.n.xyz);
  if (id.x >= dims.y || id.y >= dims.z) { return; }
  let cell = U.o.w;
  // nudged off the voxel centre so the ray misses shared edges and vertices
  let py = U.o.y + (f32(id.x) + 0.5 + 0.0137) * cell;
  let pz = U.o.z + (f32(id.y) + 0.5 - 0.0071) * cell;
  let count = u32(U.r.y);
  for (var t = 0u; t < count; t++) {
    let j = (u32(U.r.x) + t) * 3u;
    let a = tri[j].xyz;
    let b = tri[j + 1u].xyz;
    let c = tri[j + 2u].xyz;
    if (py < min(a.y, min(b.y, c.y)) || py > max(a.y, max(b.y, c.y)) || pz < min(a.z, min(b.z, c.z)) || pz > max(a.z, max(b.z, c.z))) {
      continue;
    }
    // barycentric test in the yz plane
    let d = (b.y - a.y) * (c.z - a.z) - (c.y - a.y) * (b.z - a.z);
    if (abs(d) < 1e-20) { continue; }
    let u = ((py - a.y) * (c.z - a.z) - (c.y - a.y) * (pz - a.z)) / d;
    let v = ((b.y - a.y) * (pz - a.z) - (py - a.y) * (b.z - a.z)) / d;
    if (u < 0.0 || v < 0.0 || u + v > 1.0) { continue; }
    let x = a.x + u * (b.x - a.x) + v * (c.x - a.x);
    let xi = i32(ceil((x - U.o.x) / cell - 0.5));  // first voxel centre past the hit
    if (xi >= i32(dims.x)) { continue; }
    let s = select(-1, 1, d > 0.0);
    atomicAdd(&hits_buf[idx(vec3<u32>(u32(max(xi, 0)), id.x, id.y))], s);
  }
}

// One thread per row: running sums of the signed crossings (winding along the ray) and their count
// (parity) decide inside; either one says inside, so both overlapping parts and flipped faces work.
@compute @workgroup_size(8, 8, 1)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = vec3<u32>(U.n.xyz);
  if (id.x >= dims.y || id.y >= dims.z) { return; }
  let far = U.n.w * U.o.w;
  var wind = 0;
  var hits = 0;
  for (var x = 0u; x < dims.x; x++) {
    let i = idx(vec3<u32>(x, id.x, id.y));
    let s = atomicLoad(&hits_buf[i]);
    wind += s;
    hits += abs(s);
    let d = min(bitcast<f32>(atomicLoad(&dist[i])), far);
    let inside = wind != 0 || (hits & 1) == 1;
    sdf[i] = select(d, -d, inside);
  }
}

@compute @workgroup_size(64, 1, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = vec3<u32>(U.n.xyz);
  let total = dims.x * dims.y * dims.z;
  let i = id.x + id.y * 65535u * 64u;
  if (i >= total) { return; }
  atomicStore(&dist[i], 0x7f7fffffu);
  atomicStore(&hits_buf[i], 0);
}
