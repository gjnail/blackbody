// Water in fabric (engine/cloth.py), after each substep's heat (cloth_finish.wgsl): how it moves through
// the cloth. P.w is how soaked a spot is (0 dry .. 1 holding all the water it can: Mat.f.y kg per kg of
// fibre, dripping wet). The water is of two kinds:
// - held by capillarity, in the fibres and between them: it spreads from wet to dry (wicking; the wet
//   front creeps as the square root of time, Lucas-Washburn, its diffusivity rising steeply with how wet
//   the cloth is, so the front stays sharp), and gravity pulls on it a little: water climbs a hanging
//   cloth dipped in a basin a few centimetres in a minute, wetter low down than high up;
// - free, beyond what the fabric keeps once it has drained (Mat.g.y of a soaking): it fills the gaps
//   between the yarns and runs down the cloth, faster the more there is of it (Darcy flow, its
//   permeability falling as the gaps empty), collecting along the hem and at the bottoms of folds and
//   sags, where it drips off (cloth_drip.wgsl). A towel lifted out of the water streams for a few
//   seconds, then drips more and more slowly, and stays damp.
// Also how much drier each spot is than it was after the last substep: the water the heat boiled or
// dried off it (cloth_finish.wgsl), which becomes steam in the gas (cloth_drip.wgsl, cloth_steam.wgsl).
//
// Per vertex, W[2i] = (its new wetness, wetness boiled off this substep, wetness dripped this substep,
// its wetness after the last substep); W[2i + 1] = (water waiting to drip (kg), _, _, _).
//!include cloth_common.wgsl

const H_CAP: f32 = 0.15;     // m: water held climbs this far up a cloth for its wetness to fall by a soaking
const V_DRAIN: f32 = 0.06;   // m/s: free water running straight down a cloth that is full of it

struct Params {
  a: vec4<f32>,   // dt, vertex count, ground (1/0), _
};

@group(0) @binding(0) var<storage, read> P: array<vec4<f32>>;    // w: wetness (after the heat)
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> R: array<vec4<f32>>;    // rest positions (the lengths along the cloth)
@group(0) @binding(3) var<storage, read> NB: array<u32>;         // per vertex: first neighbour, count, then the lists
@group(0) @binding(4) var<storage, read> M: array<Mat>;
@group(0) @binding(5) var<storage, read> V: array<vec4<f32>>;    // w: fabric
@group(0) @binding(6) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> W: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

// how fast free water runs down a cloth holding f of it, its keep share k (m/s)
fn drain_speed(f: f32, k: f32) -> f32 {
  let phi = clamp(f / max(1.0 - k, 0.05), 0.0, 1.0);
  return V_DRAIN * phi * phi;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.a.y)) { return; }
  let dt = U.a.x;
  let base = 2u * i;
  let w0 = W[base];
  let si = clamp(P[i].w, 0.0, 1.0);
  if (gone(S[i])) {
    // burnt away: its water went with it
    W[base] = vec4<f32>(0.0);
    return;
  }
  let boiled = max(w0.w - si, 0.0);
  let mat = M[u32(V[i].w + 0.5)];
  let keep = clamp(mat.g.y, 0.05, 0.95);
  let D0 = max(mat.g.x, 0.0);
  let first = NB[i * 2u];
  let cnt = NB[i * 2u + 1u];
  let yi = X[i].y;
  let ri = R[i].xyz;
  let fi = max(si - keep, 0.0);
  let vi = drain_speed(fi, keep);
  var ds = 0.0;
  var lower = false;
  var lsum = 0.0;
  for (var k = 0u; k < cnt; k++) {
    let j = NB[first + k];
    if (gone(S[j])) { continue; }
    let sj = clamp(P[j].w, 0.0, 1.0);
    let dr = R[j].xyz - ri;
    let l2 = max(dot(dr, dr), 1e-8);
    lsum += sqrt(l2);
    let cj = NB[j * 2u + 1u];
    let dy = X[j].y - yi;                       // > 0: j is above
    if (dy < -1e-4) { lower = true; }
    // capillarity, with gravity on the water that would move: the same exchange seen from either end
    let smax = max(si, sj);
    if (smax > 0.0) {
      let c = min(3.0 * D0 * smax * smax * dt / l2, 0.25 / f32(max(max(cnt, cj), 1u)));
      let up = select(si, sj, dy > 0.0);          // the water above, which gravity moves
      ds += c * ((sj - si) + up * abs(dy) / H_CAP * sign(dy));
    }
    // free water runs downhill: in from the neighbours above, out to those below (upwind, so the same
    // amount leaves one as reaches the other)
    if (dy > 0.0) {
      let fj = max(sj - keep, 0.0);
      if (fj > 0.0) { ds += fj * min(drain_speed(fj, keep) * dy / l2 * dt, 0.5 / f32(max(cj, 1u))); }
    } else if (dy < 0.0 && fi > 0.0) {
      ds -= fi * min(vi * (-dy) / l2 * dt, 0.5 / f32(max(cnt, 1u)));
    }
  }
  var s = max(si + ds, 0.0);
  // it drips where the water has nowhere lower to run to (along the hem, at the bottom of a fold or a
  // sag), as fast as it would run on down; and whatever has run in beyond a soaking drips at once
  var drip = 0.0;
  let on_ground = U.a.z > 0.5 && yi < 3.0 * mat.a.w + 1e-3;
  if (!lower && !on_ground && cnt > 0u) {
    let f = max(s - keep, 0.0);
    drip = f * min(drain_speed(f, keep) * f32(cnt) / max(lsum, 1e-6) * dt, 0.5);
    s -= drip;
  }
  if (s > 1.0) {
    drip += s - 1.0;
    s = 1.0;
  }
  W[base] = vec4<f32>(s, boiled, drip, s);
}
