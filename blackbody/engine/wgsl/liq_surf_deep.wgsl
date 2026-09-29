// Surface builder, pass 0b: a cell is deep inside the liquid when every cell within k of it holds
// nearly its full share of particles. Deep cells are inside by more than a kernel radius, so the
// builder marks them inside directly and never splats particles that could only reach deep cells.

struct Params {
  n: vec4<f32>,    // simulation grid dims; w = neighbourhood reach k (cells)
  k: vec4<f32>,    // particles that count as full; yzw = sides, top, bottom closed (a solid boundary counts as full)
};

@group(0) @binding(0) var<storage, read> occ: array<u32>;
@group(0) @binding(1) var<storage, read_write> deep: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let k = i32(U.n.w);
  let full = u32(U.k.x);
  var d = 1u;
  for (var z = -k; z <= k && d == 1u; z++) {
    for (var y = -k; y <= k && d == 1u; y++) {
      for (var x = -k; x <= k; x++) {
        let q = c + vec3<i32>(x, y, z);
        var ok = false;
        if (any(q < vec3<i32>(0)) || any(q >= n)) {
          // beyond the box: full if that side is solid (the liquid presses against it there)
          var closed = U.k.y;
          if (q.y < 0) { closed = U.k.w; } else if (q.y >= n.y) { closed = U.k.z; }
          ok = closed > 0.5;
        } else {
          ok = occ[u32(q.x + n.x * (q.y + n.y * q.z))] >= full;
        }
        if (!ok) {
          d = 0u;
          break;
        }
      }
    }
  }
  deep[u32(c.x + n.x * (c.y + n.y * c.z))] = d;
}
