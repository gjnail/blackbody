// Bakes a triangle mesh into a signed distance grid, one slice of triangles per dispatch. Each voxel
// keeps the nearest (unsigned) distance found so far and sums the solid angles the triangles
// subtend: the generalised winding number, about 1 inside a closed mesh and 0 outside, which stays
// sensible for meshes with holes or overlapping parts.

struct Params {
  n: vec4<f32>,   // grid dims (cells), triangles in this slice
  o: vec4<f32>,   // grid min corner (mesh units), cell size
  r: vec4<f32>,   // first triangle of the slice, initialise (1/0)
};

@group(0) @binding(0) var<storage, read> tri: array<vec4<f32>>;          // three corners per triangle
@group(0) @binding(1) var<storage, read_write> acc: array<vec2<f32>>;   // per voxel: nearest distance^2, winding
@group(1) @binding(0) var<uniform> U: Params;

fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }

// Squared distance from p to the triangle abc (after Inigo Quilez), safe for degenerate triangles.
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

// Solid angle of triangle abc seen from p (Van Oosterom and Strackee), in steradians.
fn solid_angle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let A = a - p;
  let B = b - p;
  let C = c - p;
  let la = length(A);
  let lb = length(B);
  let lc = length(C);
  let num = dot(A, cross(B, C));
  let den = la * lb * lc + dot(A, B) * lc + dot(A, C) * lb + dot(B, C) * la;
  if (abs(num) < 1e-30 && abs(den) < 1e-30) { return 0.0; }
  return 2.0 * atan2(num, den);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = vec3<u32>(U.n.xyz);
  if (any(id >= dims)) { return; }
  let i = id.x + id.y * dims.x + id.z * dims.x * dims.y;
  let p = U.o.xyz + (vec3<f32>(id) + 0.5) * U.o.w;
  var best = 1.0e30;
  var wind = 0.0;
  if (U.r.y < 0.5) {
    let a = acc[i];
    best = a.x;
    wind = a.y;
  }
  let first = u32(U.r.x);
  let count = u32(U.n.w);
  for (var t = 0u; t < count; t++) {
    let j = (first + t) * 3u;
    let a = tri[j].xyz;
    let b = tri[j + 1u].xyz;
    let c = tri[j + 2u].xyz;
    best = min(best, tri_dist2(p, a, b, c));
    wind += solid_angle(p, a, b, c);
  }
  acc[i] = vec2<f32>(best, wind);
}
