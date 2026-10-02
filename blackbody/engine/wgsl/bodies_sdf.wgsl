// The pieces of broken objects (engine/fracture.py) in a simulation grid, one substep: the signed distance to the
// nearest one is folded into the colliders' (sdf, cells, read and written in place), and the cells inside one (or
// within half a cell of it) get its velocity there (svel; w = 1 + which piece it is in this substep's list: the
// matter tells it what it pushed), which the boundary passes give the faces round it.
// Pieces are convex: the distance is the largest of the distances to their planes (exact inside, a little short
// outside near an edge). Each tile of the grid lists the pieces near it (bodyfield.py).
//!include common.wgsl

struct PieceG { a: vec4<f32>, q: vec4<f32>, v: vec4<f32>, o: vec4<f32>, r: vec4<f32> };   // as stage.wgsl

struct Params {
  g: Grid,
  t: vec4<f32>,      // tile size (cells), tiles along x, y, z
  s: vec4<f32>,      // this substep's first piece, first tile entry, pieces (count), _
};

@group(0) @binding(0) var<storage, read> PC: array<PieceG>;
@group(0) @binding(1) var<storage, read> PL: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> TC: array<vec2<u32>>;   // per tile: first, count in TL (this substep's)
@group(0) @binding(3) var<storage, read> TL: array<u32>;
@group(0) @binding(4) var sdf: texture_storage_3d<r32float, read_write>;
@group(0) @binding(5) var svel: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn rot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let h = U.g.n.w;
  let w = world_of(U.g, vec3<f32>(c) + 0.5);
  let ts = i32(U.t.x);
  let tc = vec3<u32>(c / vec3<i32>(ts));
  let ti = tc.x + u32(U.t.y) * (tc.y + u32(U.t.z) * tc.z);
  let r = TC[u32(U.s.y) + ti];
  var best = 1.0e9;
  var vel = vec3<f32>(0.0);
  var owner = 0u;
  for (var j = r.x; j < r.x + r.y; j++) {
    let k = u32(U.s.x) + TL[j];
    let P = PC[k];
    let qi = vec4<f32>(-P.q.xyz, P.q.w);
    let o = rot(qi, w - P.a.xyz);
    if (dot(o, o) > (P.r.w + h) * (P.r.w + h) && best < 0.0) { continue; }
    var dist = -1.0e9;
    let first = u32(P.v.w);
    for (var i = 0u; i < u32(P.a.w); i++) {
      let pl = PL[first + i];
      dist = max(dist, dot(normalize(pl.xyz), o) - pl.w);
    }
    if (dist < best) {
      best = dist;
      vel = P.v.xyz + cross(P.o.xyz, w - P.a.xyz);
      owner = TL[j];
    }
  }
  let base = textureLoad(sdf, c).x;
  let s = best / h;
  textureStore(sdf, c, vec4<f32>(min(base, s), 0.0, 0.0, 0.0));
  textureStore(svel, c, select(vec4<f32>(0.0), vec4<f32>(vel, 1.0 + f32(owner)), s < 0.5));
}
