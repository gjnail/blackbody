// Fabric self-collision, vertex against triangle: each vertex is kept at least its fabric's collision
// radius from every triangle of cloth nearby (found through the hashed grid of vertices, cloth_hash.wgsl,
// and the triangles around them), except the cloth right around itself. A vertex remembers which side of
// a triangle it was on when the substep began: one that has crossed in this step is put back on that
// side, so a fast fold cannot slip through itself. Two fabrics collide at the larger of their radii.
// Layers rub: half their sliding past each other is taken away. The pushes are gathered here and applied
// in cloth_collide.wgsl.
//!include cloth_common.wgsl

struct Params {
  h: vec4<f32>,                                 // vertex count, cell size (m), bucket count, _
  rad: array<vec4<f32>, 4>,                     // collision radius (m) per fabric, four to a vec4
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> P: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> UV: array<vec4<f32>>;     // weave coordinates (m), fabric
@group(0) @binding(4) var<storage, read> HN: array<u32>;
@group(0) @binding(5) var<storage, read_write> D: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> T: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> VT: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

fn radius(f: u32) -> f32 {
  let q = U.rad[f / 4u];
  let k = f % 4u;
  return select(select(q.w, q.z, k == 2u), select(q.y, q.x, k == 0u), k < 2u);
}

// closest point on triangle (a, b, c) to p, and whether it lies inside the triangle (not on an edge)
fn closest(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
  let ab = b - a;
  let ac = c - a;
  let ap = p - a;
  let d1 = dot(ab, ap);
  let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return vec4<f32>(a, 0.0); }
  let bp = p - b;
  let d3 = dot(ab, bp);
  let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return vec4<f32>(b, 0.0); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return vec4<f32>(a + ab * (d1 / (d1 - d3)), 0.0); }
  let cp = p - c;
  let d5 = dot(ab, cp);
  let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return vec4<f32>(c, 0.0); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return vec4<f32>(a + ac * (d2 / (d2 - d6)), 0.0); }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    return vec4<f32>(b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))), 0.0);
  }
  let den = 1.0 / (va + vb + vc);
  return vec4<f32>(a + ab * (vb * den) + ac * (vc * den), 1.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.h.x)) { return; }
  D[i] = vec4<f32>(0.0);
  let xi4 = X[i];
  if (gone(S[i]) || xi4.w <= 0.0) { return; }
  let xi = xi4.xyz;
  let pi = P[i].xyz;
  let ui = UV[i];
  let fi = u32(ui.z + 0.5);
  let ri = radius(fi);
  let nb = u32(U.h.z);
  let cell = U.h.y;
  let c0 = vec3<i32>(floor(xi / cell));
  var acc = vec3<f32>(0.0);
  var cnt = 0.0;
  for (var k = 0; k < 27; k++) {
    let c = c0 + vec3<i32>(k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1);
    var j = HN[cell_hash(c, nb)];
    var guard = 0;
    loop {
      if (j == 0xffffffffu || guard >= 64) { break; }
      guard++;
      let nj = HN[nb + j];
      if (j != i && all(vec3<i32>(floor(X[j].xyz / cell)) == c)) {
        // the triangles around j (one met from several of its vertices counts once each time: the pushes
        // are averaged)
        let first = VT[2u * j];
        let tc = VT[2u * j + 1u];
        for (var q = 0u; q < tc; q++) {
          let t = T[VT[first + q]];
          if (t.x == i || t.y == i || t.z == i) { continue; }
          if (gone(S[t.x]) || gone(S[t.y]) || gone(S[t.z])) { continue; }
          let ft = t.w;
          let r = max(ri, radius(ft));
          if (ft == fi) {
            // the cloth right around this vertex holds it apart by itself
            let du = min(min(distance(UV[t.x].xy, ui.xy), distance(UV[t.y].xy, ui.xy)), distance(UV[t.z].xy, ui.xy));
            if (du < 2.5 * r) { continue; }
          }
          let a = X[t.x].xyz;
          let b = X[t.y].xyz;
          let cc = X[t.z].xyz;
          let cp = closest(xi, a, b, cc);
          let dv = xi - cp.xyz;
          let dist = length(dv);
          if (dist > 2.0 * r) { continue; }
          // which side it was on when the substep began
          let pa = P[t.x].xyz;
          let np = cross(P[t.y].xyz - pa, P[t.z].xyz - pa);
          let n = cross(b - a, cc - a);
          let ln = length(n);
          if (ln < 1e-12) { continue; }
          let nn = n / ln;
          let was = dot(pi - pa, np);
          let now = dot(xi - a, nn);
          var push = vec3<f32>(0.0);
          if (cp.w > 0.5 && was * now < 0.0) {
            // crossed through: back to its own side, a radius off the cloth
            push = nn * (select(-r, r, was > 0.0) - now);
          } else if (dist < r) {
            let dir = select(nn * sign(was + 1e-30), dv / max(dist, 1e-12), dist > 1e-6);
            push = dir * (r - dist);
          } else {
            continue;
          }
          // rub: take away half the sliding against the triangle
          let rel = (xi - pi) - ((a + b + cc) - (pa + P[t.y].xyz + P[t.z].xyz)) / 3.0;
          push -= 0.5 * (rel - dot(rel, nn) * nn);
          acc += push;
          cnt += 1.0;
        }
      }
      j = nj;
    }
  }
  if (cnt > 0.0) { D[i] = vec4<f32>(acc / cnt, cnt); }
}
