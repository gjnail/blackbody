// Caustics, pass 1: photons from the key light, one per sample of a regular grid over the ground,
// traced through the liquid surface (Fresnel, refraction, absorption and murk inside, refraction
// out again) to where they land on the ground, and added there. Without liquid every photon lands
// on its own sample, so dividing by the photons per texel later gives the light relative to the
// unobstructed sunlight: above 1 where the surface focuses it, below 1 in its shadow.
//!include liq_caustic_common.wgsl

@group(0) @binding(0) var surf_t: texture_3d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: CParams;

const FX_E: f32 = 4096.0;

fn splat(x: f32, z: f32, e: f32) {
  let cm = vec2<i32>(U.cm.xy);
  let q = vec2<f32>(x / U.n.x, z / U.n.z) * U.cm.xy - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(q));
  let f = q - floor(q);
  for (var j = 0; j < 4; j++) {
    let o = vec2<i32>(j & 1, j >> 1);
    var c = i0 + o;
    // a photon's footprint hanging a texel over the map's edge goes into the edge texel, so the
    // edge is not darker than the rest
    if (any(c < vec2<i32>(-1)) || any(c > cm)) { continue; }
    c = clamp(c, vec2<i32>(0), cm - vec2<i32>(1));
    let w = mix(vec2<f32>(1.0) - f, f, vec2<f32>(o));
    atomicAdd(&acc[u32(c.x + cm.x * c.y)], u32(round(e * w.x * w.y * FX_E)));
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = U.cm.z;
  let np = U.cm.xy * k;
  if (f32(id.x) >= np.x || f32(id.y) >= np.y) { return; }
  let n = U.n.xyz;
  let lg = U.sun.xyz;
  // the ground point this photon lands on when nothing is in the way
  let gp = vec3<f32>((f32(id.x) + 0.5) / np.x * n.x, 0.0, (f32(id.y) + 0.5) / np.y * n.z);
  let d0 = -lg;
  let t_top = (n.y + 1.0) / max(lg.y, 0.05);
  var o = gp + lg * t_top;
  if (U.lvl.y > 0.5) {
    // open water: launch along the path that still, flat water would bend onto this ground point,
    // so the map is covered right to its edges
    let tr = refract(d0, vec3<f32>(0.0, 1.0, 0.0), 1.0 / U.sun.w);
    let pl = gp - tr * (U.lvl.x / max(-tr.y, 1e-3));
    o = pl + lg * ((n.y + 1.0 - U.lvl.x) / max(lg.y, 0.05));
  }
  var rb = box_range_c(o, d0, n);
  // open water: the liquid carries on past the box, so the whole path down to the ground counts
  if (U.lvl.y > 0.5) { rb = vec2<f32>(0.0, o.y / max(-d0.y, 1e-4)); }
  if (rb.y <= rb.x) { splat(gp.x, gp.z, 1.0); return; }
  // down through the air to the liquid (or the ground)
  let minstep = 0.4 / U.nf.w;
  var t = rb.x;
  var d = phi_c(o + d0 * t);
  var hit = false;
  for (var i = 0; i < 400; i++) {
    if (d < 0.0) { hit = true; break; }
    t += max(d * 0.85, minstep);
    if (t >= rb.y) { break; }
    d = phi_c(o + d0 * t);
  }
  if (!hit) { splat(gp.x, gp.z, 1.0); return; }
  var pos = o + d0 * t;
  var nrm = normal_c(pos);
  let ior = U.sun.w;
  var e = 1.0 - fresnel_c(-dot(d0, nrm), 1.0, ior);
  var dir = refract(d0, nrm, 1.0 / ior);
  if (dot(dir, dir) < 1e-6) { return; }
  pos -= nrm * (0.35 / U.nf.w);
  // through the body: to the ground, or out the far side and on to the ground
  let ext = dot(U.absorb.rgb, vec3<f32>(0.3333)) + U.absorb.w;
  for (var ev = 0; ev < 4; ev++) {
    var s = 0.0;
    var exited = false;
    d = phi_c(pos);
    for (var i = 0; i < 400; i++) {
      let st = max(-d * 0.85, 0.5 / U.nf.w);
      s += st;
      let p = pos + dir * s;
      if (p.y <= 0.0) {
        // landed under the liquid
        let back = p.y / min(dir.y, -1e-4);
        let l = s - back;
        let q = pos + dir * l;
        splat(q.x, q.z, e * exp(-ext * l * U.n.w));
        return;
      }
      if (U.lvl.y < 0.5 && (any(p < vec3<f32>(0.0)) || any(p > n))) {
        // left the box inside the liquid (open water): straight on to the ground
        if (dir.y < -1e-4) {
          let l = s + p.y / -dir.y;
          let q = pos + dir * l;
          splat(q.x, q.z, e * exp(-ext * l * U.n.w));
        }
        return;
      }
      d = phi_c(p);
      if (d > 0.0) { exited = true; break; }
    }
    if (!exited) { return; }
    e *= exp(-ext * s * U.n.w);
    pos = pos + dir * s;
    let n2 = normal_c(pos);
    let od = refract(dir, -n2, ior);
    if (dot(od, od) < 1e-6) { return; }  // total internal reflection: this light stays in the liquid
    e *= 1.0 - fresnel_c(dot(dir, n2), ior, 1.0);
    dir = od;
    pos += n2 * (0.35 / U.nf.w);
    // through the air below a drop or a sheet: to the ground, unless it meets more liquid
    var t2 = 0.0;
    var hit2 = false;
    for (var i = 0; i < 200; i++) {
      let p = pos + dir * t2;
      if (p.y <= 0.0) { break; }
      let dd = phi_c(p);
      if (dd < 0.0) { hit2 = true; break; }
      t2 += max(dd * 0.85, minstep);
    }
    if (!hit2) {
      if (dir.y < -1e-4) {
        let q = pos + dir * (-pos.y / dir.y);
        splat(q.x, q.z, e);
      }
      return;
    }
    pos = pos + dir * t2;
    nrm = normal_c(pos);
    e *= 1.0 - fresnel_c(-dot(dir, nrm), 1.0, ior);
    dir = refract(dir, nrm, 1.0 / ior);
    if (dot(dir, dir) < 1e-6) { return; }
    pos -= nrm * (0.35 / U.nf.w);
  }
}
