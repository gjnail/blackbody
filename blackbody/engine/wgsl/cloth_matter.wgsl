// Fabric and matter (cloth.py, matter.py): the cloth as a thin sheet on the matter's grid, which the matter cannot pass
// through, and the push the matter gives it back.
//  - splat: each vertex onto the nodes within a radius (by (1 - d^2/R^2)^2): its weight, velocity, normal, how far
//    each node is in front of it along its normal (the side the node is on), and its weight per area (how hard it
//    pushes what it moves into: mpm_common.wgsl SHEET_N). Fixed-point atomics. It also reaches on along its normal
//    as far as it moves through the frame (the distance still measured from where it is), so the matter in its way
//    knows which side of it it is on before the sheet gets there: a cloth falling a few nodes a frame onto a heap comes
//    to rest on its smoothed surface, under the tip's grains, and those would otherwise first meet it from its far
//    side and be lifted onto it.
//  - gather: each vertex's share of the momentum the matter's grid took from its nodes over the frame
//    (mpm_grid.wgsl, mpm_g2p.wgsl), as a force through the next one (the matter's weight on a sling, sand landing on a
//    sheet), and that force's worth of mass (its weight's: kg), which rides on the vertex through the next frame
//    (cloth_predict.wgsl): a sling holding sand moves as sand, not as cloth kicked by it. Half last frame's and half
//    this one's (the push comes a frame late; unsmoothed, sheet and matter can kick each other in turn).

struct Params {
  m: vec4<f32>,      // the matter's grid: node 0 (fire-local m), node spacing (m)
  mn: vec4<f32>,     // its nodes (x, y, z), cloth vertices (count)
  k: vec4<f32>,      // the reach (m); splat: the frame (s), gather: 1 / the frame (1/s: momentum to force), gravity
                     // (m/s^2), _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;    // cloth positions (fire-local m)
@group(0) @binding(1) var<storage, read> V: array<vec4<f32>>;    // velocities (m/s)
@group(0) @binding(2) var<storage, read> N: array<vec4<f32>>;    // normals
@group(0) @binding(3) var<storage, read> S: array<vec4<f32>>;    // state: y 1 = burnt away
@group(0) @binding(4) var<storage, read_write> CF: array<atomic<i32>>;   // per node: weight, w velocity (3), w normal (3), w side,
                                                                          // w weight per area
@group(0) @binding(5) var<storage, read_write> CT: array<atomic<i32>>;   // per node: momentum taken (3)
@group(0) @binding(6) var<storage, read_write> MP: array<vec4<f32>>;     // per vertex: the matter's push (N), the mass
                                                                          // riding on it (kg)
@group(0) @binding(7) var<storage, read> REST: array<vec4<f32>>;    // rest positions; w: inverse mass (negative: pinned)
@group(1) @binding(0) var<uniform> U: Params;

const FX_W: f32 = 4096.0;
const FX_V: f32 = 4096.0;
const FX_T: f32 = 16384.0;  // CT's (mpm_common.wgsl SHEET_FX_T)
const CF_N: u32 = 9u;
const PINNED: f32 = 10000.0;   // kg/m^2: a pinned vertex's weight per area (it pushes whatever it meets)

fn nidx3(c: vec3<i32>, n: vec3<i32>) -> u32 { return u32(c.x + n.x * (c.y + n.y * c.z)); }

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.mn.w)) { return; }
  if (S[i].y >= 1.0) { return; }
  let x = X[i].xyz;
  let v = V[i].xyz;
  let nl = length(N[i].xyz);
  if (nl < 1.0e-6) { return; }
  let nrm = N[i].xyz / nl;
  // its weight per area (kg/m^2; X.w is 0 while pinned)
  let sigma = select(PINNED, 1.0 / max(abs(REST[i].w), 1.0e-9) / max(N[i].w, 1.0e-8), X[i].w > 0.0);
  let R = U.k.x;
  let n = vec3<i32>(U.mn.xyz);
  let g = (x - U.m.xyz) / U.m.w;
  let r = R / U.m.w;
  // (the way it goes along its normal this frame, at most four reaches)
  var way = nrm * (dot(v, nrm) * U.k.y);
  let wl = length(way);
  if (wl > 4.0 * R) { way *= 4.0 * R / wl; }
  let g1 = g + way / U.m.w;
  let lo = max(vec3<i32>(ceil(min(g, g1) - vec3<f32>(r))), vec3<i32>(0));
  let hi = min(vec3<i32>(floor(max(g, g1) + vec3<f32>(r))), n - vec3<i32>(1));
  let ww = max(dot(way, way), 1.0e-12);
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var xx = lo.x; xx <= hi.x; xx++) {
        let c = vec3<i32>(xx, y, z);
        let off = (vec3<f32>(c) - g) * U.m.w;
        var q = dot(off, off) / (R * R);
        if (q >= 1.0) {
          // out of its reach where it is: in reach of its way?
          let e = off - way * clamp(dot(off, way) / ww, 0.0, 1.0);
          q = dot(e, e) / (R * R);
          if (q >= 1.0) { continue; }
        }
        let w = (1.0 - q) * (1.0 - q);
        let k = nidx3(c, n) * CF_N;
        atomicAdd(&CF[k], i32(round(w * FX_W)));
        atomicAdd(&CF[k + 1u], i32(round(w * v.x * FX_V)));
        atomicAdd(&CF[k + 2u], i32(round(w * v.y * FX_V)));
        atomicAdd(&CF[k + 3u], i32(round(w * v.z * FX_V)));
        atomicAdd(&CF[k + 4u], i32(round(w * nrm.x * FX_W)));
        atomicAdd(&CF[k + 5u], i32(round(w * nrm.y * FX_W)));
        atomicAdd(&CF[k + 6u], i32(round(w * nrm.z * FX_W)));
        atomicAdd(&CF[k + 7u], i32(round(w * dot(off, nrm) / U.m.w * FX_W)));
        atomicAdd(&CF[k + 8u], i32(round(w * min(sigma, PINNED) * FX_W)));
      }
    }
  }
}

@compute @workgroup_size(64, 1, 1)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.mn.w)) { return; }
  var f = vec3<f32>(0.0);
  if (S[i].y < 1.0) {
    let x = X[i].xyz;
    let R = U.k.x;
    let n = vec3<i32>(U.mn.xyz);
    let g = (x - U.m.xyz) / U.m.w;
    let r = R / U.m.w;
    let lo = max(vec3<i32>(ceil(g - vec3<f32>(r))), vec3<i32>(0));
    let hi = min(vec3<i32>(floor(g + vec3<f32>(r))), n - vec3<i32>(1));
    for (var z = lo.z; z <= hi.z; z++) {
      for (var y = lo.y; y <= hi.y; y++) {
        for (var xx = lo.x; xx <= hi.x; xx++) {
          let c = vec3<i32>(xx, y, z);
          let off = (vec3<f32>(c) - g) * U.m.w;
          let q = dot(off, off) / (R * R);
          if (q >= 1.0) { continue; }
          let w = (1.0 - q) * (1.0 - q);
          let k = nidx3(c, n);
          let W = f32(atomicLoad(&CF[k * CF_N])) / FX_W;
          if (W <= 1.0e-6) { continue; }
          let t = vec3<f32>(f32(atomicLoad(&CT[3u * k])), f32(atomicLoad(&CT[3u * k + 1u])),
                            f32(atomicLoad(&CT[3u * k + 2u]))) / FX_T;
          f += t * (w / W);
        }
      }
    }
  }
  let push = f * U.k.y;
  MP[i] = mix(MP[i], vec4<f32>(push, length(push) / max(U.k.z, 1.0)), 0.5);
}
