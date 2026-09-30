// Volume ray march of the fire domain, and the surfaces the fire sits among.
// beauty: premultiplied radiance (emission + scattered light), alpha = 1 - transmittance
// emit:   emission only (premultiplied), alpha = flame coverage
// aux:    x = heat (integrated temperature, m), y = depth (m, opacity-weighted), z = max Kelvin / 1000
// surf:   rgb = fire light arriving at the surface seen in this pixel (the ground, or a collider in
//         the shot), a = holdout coverage (a collider in the shot hides what is behind it)
// mask:   x = scorch (0 untouched .. 1 burnt out), y = holdout depth (clip w) * coverage, z = coverage
//
// Colliders marked "hides fire" are found by sphere tracing their exact shapes (so they work outside
// the simulation box too); the volume march stops at them. Fire light on surfaces is gathered from a
// short list of point lights built from the fire's emission (lights.wgsl): brightest facing the
// flames, falling off with the square of the distance.
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl
//!include burn_common.wgsl
//!include burnobj.wgsl

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var L0: texture_3d<f32>;
@group(0) @binding(3) var L1: texture_3d<f32>;
@group(0) @binding(4) var noise: texture_3d<f32>;
@group(0) @binding(5) var bb: texture_2d<f32>;
@group(0) @binding(6) var lin: sampler;
@group(0) @binding(7) var rep: sampler;
@group(0) @binding(8) var aux: texture_3d<f32>;
@group(0) @binding(9) var chem: texture_3d<f32>;
@group(0) @binding(10) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var out_aux: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var out_surf: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var out_mask: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var<storage, read> lights: array<vec4<f32>>;   // per light: position (fire-local m), softening (m); power (rgb)
@group(0) @binding(16) var<storage, read> light_count: array<u32>;
@group(0) @binding(17) var burn: texture_3d<f32>;                     // burnable floor (simulation grid)
@group(0) @binding(18) var burn_obj: texture_3d<f32>;
@group(0) @binding(19) var<storage, read> slots: array<BurnSlot>;
@group(0) @binding(20) var atlas: texture_3d<f32>;
@group(0) @binding(21) var limit: texture_2d<f32>;                    // liquid in front: y = its depth (m), w = coverage

//!include shade.wgsl

struct Params {
  inv_vp: mat4x4<f32>,   // clip -> world
  w2g: mat4x4<f32>,      // world -> grid (cells)
  vp: mat4x4<f32>,       // world -> clip
  n: vec4<f32>,          // grid dims, metres per cell
  ln: vec4<f32>,         // light-volume dims, ground (1 = no fade at the floor)
  res: vec4<f32>,        // width, height, sub-pixel jitter (px)
  frame: vec4<f32>,      // noise seed, shutter (s), motion blur on, max steps
  look: Look,
  org: vec4<f32>,        // grid corner (fire-local m); w = grid cells per velocity cell (upres)
  surf: vec4<f32>,       // holdouts on (1/0), ground in the shot (1/0), surfaces lit (1/0), scorch on (1/0)
  bgrid: vec4<f32>,      // burnable floor grid dims (simulation grid); w = its cell size (m)
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
  lim: vec4<f32>,        // x = stop the march at the liquid surface in `limit` (fire and liquid in one box)
};
@group(1) @binding(0) var<uniform> U: Params;

fn ign(p: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Nearest collider in the shot at fire-local point p: (distance (m), index).
fn holdout_sdf(p: vec3<f32>) -> vec2<f32> {
  var best = vec2<f32>(1.0e9, -1.0);
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    if (U.col[i].y.w < 0.5) { continue; }
    let d = col_sdf(U.col[i], p);
    if (d < best.x) { best = vec2<f32>(d, f32(i)); }
  }
  return best;
}

fn col_normal(k: Collider, p: vec3<f32>, e: f32) -> vec3<f32> {
  let g = vec3<f32>(col_sdf(k, p + vec3<f32>(e, 0.0, 0.0)) - col_sdf(k, p - vec3<f32>(e, 0.0, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, e, 0.0)) - col_sdf(k, p - vec3<f32>(0.0, e, 0.0)),
                    col_sdf(k, p + vec3<f32>(0.0, 0.0, e)) - col_sdf(k, p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  return select(vec3<f32>(0.0, 1.0, 0.0), g / l, l > 1e-8);
}

// Fire light arriving at a surface point with normal nrm (fire-local metres).
fn surface_light(p: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
  var e = vec3<f32>(0.0);
  let count = light_count[0];
  for (var i = 0u; i < count; i++) {
    let a = lights[2u * i];
    let d = a.xyz - p;
    let r2 = dot(d, d);
    let cosine = dot(nrm, d) * inverseSqrt(max(r2, 1e-8));
    // a light the size of its block: wrap the cosine a little and soften the falloff near it
    let facing = clamp((cosine + 0.2) / 1.2, 0.0, 1.0);
    e += lights[2u * i + 1u].rgb * (facing / (r2 + a.w * a.w));
  }
  return e;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let L = U.look;
  let n = U.n.xyz;
  let h = U.n.w;

  let uv = (vec2<f32>(px) + vec2<f32>(0.5) + U.res.zw) / U.res.xy;
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let pn = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
  let pf = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
  let ro_w = pn.xyz / pn.w;
  let rd_w = normalize(pf.xyz / pf.w - ro_w);
  let ro = (U.w2g * vec4<f32>(ro_w, 1.0)).xyz;
  let rdg = (U.w2g * vec4<f32>(rd_w, 0.0)).xyz;
  let cells_per_m = length(rdg);
  let rd = rdg / cells_per_m;

  let safe = select(rd, vec3<f32>(1e-8), abs(rd) < vec3<f32>(1e-8));
  let inv = 1.0 / safe;
  let ta = (vec3<f32>(0.0) - ro) * inv;
  let tb = (n - ro) * inv;
  let t_in = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  var t_out = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));

  // -- surfaces: colliders in the shot and the ground --------------------------------------------
  let far = max(t_out, 0.0) + 2.0 * max(n.x, max(n.y, n.z));   // cells: the box and as far again beyond
  var t_obj = 1.0e9;
  var hit = -1;
  if (U.surf.x > 0.5 && U.ccnt.x > 0.5) {
    var t = 0.0;
    for (var i = 0; i < 160; i++) {
      let pl = U.org.xyz + (ro + rd * t) * h;
      let dh = holdout_sdf(pl);
      if (dh.x < 0.3 * h) {
        t_obj = t;
        hit = i32(dh.y);
        break;
      }
      t += max(dh.x / h, 0.25);
      if (t > far) { break; }
    }
  }
  var t_gnd = 1.0e9;
  if (U.surf.y > 0.5 && rd.y < -1e-6 && ro.y > 0.0) {
    t_gnd = -ro.y / rd.y;
  }
  var surf = vec4<f32>(0.0);
  var mask = vec4<f32>(0.0);
  let t_s = min(t_obj, t_gnd);
  if (t_s < 1.0e8) {
    let ps = U.org.xyz + (ro + rd * t_s) * h;
    var nrm = vec3<f32>(0.0, 1.0, 0.0);
    if (hit >= 0 && t_obj <= t_gnd) {
      let k = U.col[hit];
      nrm = col_normal(k, ps, 0.5 * h);
      surf.a = 1.0;
      let clip = U.vp * vec4<f32>(ro_w + rd_w * (t_obj / cells_per_m), 1.0);
      mask = vec4<f32>(mask.x, clip.w, 1.0, 0.0);
      if (U.surf.w > 0.5 && i32(k.m2.w) >= 0) {
        mask.x = burn_char(obj_burn_at(k, slots[i32(k.m2.w)], ps + nrm * (0.75 * h)));
      }
    } else if (U.surf.w > 0.5) {
      // the burnable floor is the bottom layer of the simulation grid
      let bh = U.bgrid.w;
      let cg = vec3<i32>(floor((ps - U.org.xyz) / bh));
      if (cg.x >= 0 && cg.z >= 0 && cg.x < i32(U.bgrid.x) && cg.z < i32(U.bgrid.z)) {
        mask.x = burn_char(textureLoad(burn, vec3<i32>(cg.x, 0, cg.z), 0));
      }
    }
    if (U.surf.z > 0.5) { surf = vec4<f32>(surface_light(ps + nrm * (0.5 * h), nrm), surf.a); }
  }
  textureStore(out_surf, px, surf);
  textureStore(out_mask, px, mask);

  t_out = min(t_out, t_obj);
  if (U.lim.x > 0.5) {
    // only the fire in front of the liquid: the liquid layer already shows the fire behind it
    let lq = textureLoad(limit, px, 0);
    if (lq.w > 0.5 && lq.y > 0.0) { t_out = min(t_out, lq.y * cells_per_m); }
  }
  if (t_out <= t_in) {
    textureStore(out_beauty, px, vec4<f32>(0.0));
    textureStore(out_emit, px, vec4<f32>(0.0));
    textureStore(out_aux, px, vec4<f32>(0.0));
    return;
  }

  // -- the volume ---------------------------------------------------------------------------------
  let step = L.misc.z;
  let jit = fract(ign(vec2<f32>(px)) + U.frame.x * 0.61803398875);
  var t = t_in + step * jit;
  let ds = step * h;
  let cos_sun = dot(rd_w, L.sundir.xyz);
  let phase = hg(cos_sun, L.amb.w);
  let lscale = U.ln.xyz / n;
  let fade_cells = max(L.misc.y, 1e-3);
  let shutter = U.frame.y * (fract(ign(vec2<f32>(px) + vec2<f32>(17.0, 59.0)) + U.frame.x * 0.7548776662) - 0.5);
  let rise = vec3<f32>(0.0, -L.misc.w * L.detail.w * L.detail.y, 0.0);
  let max_steps = i32(U.frame.w);
  let vk = max(U.org.w, 1.0);

  var tr = 1.0;   // transmittance for light transport
  var tra = 1.0;  // transmittance of the background (alpha): flames occlude only partly
  var col = vec3<f32>(0.0);
  var emit = vec3<f32>(0.0);
  var flame_a = 0.0;
  var heat = 0.0;
  var dsum = 0.0;
  var dw = 0.0;
  var kmax = 0.0;

  for (var i = 0; i < max_steps; i++) {
    if (t >= t_out || (tr < 0.002 && tra < 0.002)) { break; }
    let p = ro + rd * t;
    let pl = p * lscale;
    let l1 = samp_c(L1, lin, pl, U.ln.xyz);
    if (l1.y < 1e-5) {
      t += step * 3.0;
      continue;
    }
    var q = p;
    if (U.frame.z > 0.5) {
      q -= vel_at(vel, lin, q / vk, n / vk) * (shutter / h);
    }
    var dmod = 1.0;
    if (L.detail.x > 0.0) {
      let nz = textureSampleLevel(noise, rep, q * h * L.detail.y + rise, 0.0);
      q += (nz.yzw - vec3<f32>(0.5)) * L.detail.z;
      dmod = max(1.0 + L.detail.x * (nz.x - 0.5) * 3.0, 0.0);
    }
    let s = samp_c(scal, lin, q, n);

    var edge = min(min(p.x, n.x - p.x), min(p.z, n.z - p.z));
    edge = min(edge, n.y - p.y);
    if (U.ln.w < 0.5) { edge = min(edge, p.y); }
    let fade = smoothstep(0.0, fade_cells, edge);

    var ax = vec4<f32>(0.0);
    if (L.air.z > 0.5) { ax = samp_c(aux, lin, q, n); }
    var ch = vec4<f32>(0.0);
    if (L.air.w > 0.5) { ch = samp_c(chem, lin, q, n); }

    let dens = smoke_extinction(s, L) * dmod * fade;
    let wet = steam_extinction(s, ax, L) * dmod * fade;
    let sf = flame_sigma(s, L) * fade;
    let sig = dens + wet + sf;
    let e = emission_k(s, ch, sf, dens, L);
    let l0 = samp_c(L0, lin, pl, U.ln.xyz);
    var lin_in = L.amb.rgb * l1.x + L.sun.rgb * (l0.a * phase) + l0.rgb * L.sun.w;
    let scat = dens * L.smoke.rgb + wet * L.steam.rgb;
    if (L.ms.x > 0.0 && dens + wet > 1e-5) {
      // multiple scattering (after Wrenninge): each further bounce keeps the albedo's share of the
      // light, sees the medium thinner and scatters more evenly, so pale smoke and steam stay bright inside
      let albedo = scat / (dens + wet);
      var a_o = albedo * L.ms.x;
      var b_o = 0.5;
      var c_o = 0.5;
      for (var o = 0; o < 3; o++) {
        lin_in += a_o * (L.amb.rgb * pow(l1.x, b_o) + L.sun.rgb * (pow(l0.a, b_o) * hg(cos_sun, L.amb.w * c_o))
                         + l0.rgb * L.sun.w);
        a_o *= albedo * L.ms.x;
        b_o *= 0.5;
        c_o *= 0.5;
      }
    }
    let a = exp(-sig * ds);
    let w = select(ds, (1.0 - a) / sig, sig > 1e-6);
    col += tr * (e + scat * lin_in) * w;
    emit += tr * e * w;
    let fa = 1.0 - exp(-sf * ds);
    flame_a += tr * fa * (1.0 - flame_a);
    heat += max(s.x, 0.0) * fade * ds;
    if (luma(e) > 1e-4) { kmax = max(kmax, kelvin(s.x, L)); }
    let dop = tr * (1.0 - a);
    dsum += dop * t;
    dw += dop;
    tr *= a;
    tra *= exp(-(dens + wet + sf * L.fire2.w) * ds);
    t += step;
  }

  let depth = select(0.0, dsum / max(dw, 1e-6) / cells_per_m, dw > 1e-4);
  textureStore(out_beauty, px, vec4<f32>(col, 1.0 - tra));
  textureStore(out_emit, px, vec4<f32>(emit, clamp(flame_a, 0.0, 1.0)));
  textureStore(out_aux, px, vec4<f32>(heat, depth, kmax * 0.001, 1.0 - tra));
}
