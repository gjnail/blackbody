// Clouds: the sky drawn (engine/cloud_render.py), one thread per pixel: the eye ray traced through the
// cloud box, gathering the light the cloud scatters toward the eye.
//  - sunlight: what is left of it after the cloud between the point and the sun (the light volume), times
//    the phase function: droplets throw most of it forward (the silver lining round a cloud in front of the
//    sun), ice crystals less sharply; light scattered many times inside a thick cloud is approximated by
//    octaves of weaker extinction and broader phase (Wrenninge et al.), which keeps thick clouds' lit sides
//    white instead of grey;
//  - skylight, from above and the sides, through the cloud round the point: so bases are dark, tops
//    bright and the shadowed sides blue-grey;
//  - light thrown back up from the ground under it;
//  - rain shafts (grey) and snow curtains (white) where the precipitation falls below the cloud;
//  - the air between: the further away, the more it fades into the haze of the horizon (aerial perspective);
//  - detail finer than the grid: a cloud's edge is sharp (where the air is saturated there is cloud,
//    where it is not there is none), so a cell part-filled with cloud is drawn as cloud where billowing
//    noise puts it and clear air elsewhere: rounded lobes with creases between (as eddies the grid cannot
//    hold heap a cumulus' surface), drifting with the wind; softer for ice cloud, whose crystals fall and
//    evaporate slowly into fibres; the cores stay solid, and rain and hail fall in soft shafts streaked
//    by falling. The
//    sunlight to each point is traced through this detail for the first cell, so the lobes shade each other.
// Behind the box: the sky (brighter toward the horizon and round the sun) and the ground, with the clouds'
// shadows on it. With footage behind, only the clouds are drawn (premultiplied, over it).
//!include common.wgsl
//!include noise.wgsl

struct Params {
  inv_vp: mat4x4<f32>,   // clip -> world (render frame)
  w2l: mat4x4<f32>,      // world -> the scene's own frame
  eye: vec4<f32>,        // camera (scene frame), metres of sky per scene metre
  res: vec4<f32>,        // width, height, step (cells), frame seed
  box: vec4<f32>,        // nx, ny, nz, cell size (m of sky)
  org: vec4<f32>,        // box corner (m of sky), visibility of the air (m)
  sun: vec4<f32>,        // toward the sun (scene frame), sun disc on (1/0)
  sun_col: vec4<f32>,    // sunlight (rgb), brightness of the clouds
  sky: vec4<f32>,        // zenith sky (rgb), skylight on the clouds
  horizon: vec4<f32>,    // horizon haze (rgb), sky drawn (1/0)
  ground: vec4<f32>,     // ground colour (rgb), ground drawn (1/0)
  look: vec4<f32>,       // silver lining (forward scattering), multiple scattering, precipitation shafts, edge detail
  wind: vec4<f32>,       // the wind the detail drifts with (m/s, sky frame), time (s)
  lume: vec4<f32>,       // Lume's light (cloud_lume.wgsl) on (1/0), the cloud's cells per cell of its grid, its cells (yz)
  lume2: vec4<f32>,      // its cells (x), _, _, _
};

@group(0) @binding(0) var light: texture_3d<f32>;
@group(0) @binding(1) var precip: texture_2d<f32>;
@group(0) @binding(2) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var out_aux: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var out_mask: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var mat: texture_3d<f32>;
@group(0) @binding(7) var lume_t: texture_3d<f32>;   // Lume: the light scattered more than once (cloud_lume.wgsl), its own grid
@group(0) @binding(8) var lume_d: texture_3d<f32>;   // ... the way it flows: its first moment (luminance, xyz), its mean (w)
@group(1) @binding(0) var<uniform> U: Params;

const PI: f32 = 3.14159265;
const DENSE: f32 = 0.05;   // 1/m: the densest a cloud's surface is drawn

fn hg(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// droplets: a strong forward peak with a little backscatter (the glory side); ice: broader
fn phase(c: f32, ice: f32, k: f32) -> f32 {
  let water = mix(hg(c, -0.25 * k), hg(c, 0.85 * k), 0.8);
  let icep = mix(hg(c, -0.2 * k), hg(c, 0.7 * k), 0.75);
  return mix(water, icep, ice);
}

fn light_at(p: vec3<f32>) -> vec4<f32> { return trilinear(light, p); }

// Lume's light at p (cells of its own grid, nl of them) scattered toward the eye looking along d by a phase of mean cosine
// g: its mean, and its flow (first moment m) weighed by the phase, L0 + 3 g (m . d), in the light's colour
fn lume_toward(p: vec3<f32>, nl: vec3<i32>, d: vec3<f32>, g: f32) -> vec3<f32> {
  let l0 = lume_at(p, nl);
  let q = clamp(p - vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(nl - vec3<i32>(1)));
  let i = vec3<i32>(floor(q));
  let f = q - vec3<f32>(i);
  let i1 = min(i + vec3<i32>(1), nl - vec3<i32>(1));
  let m00 = mix(textureLoad(lume_d, i, 0), textureLoad(lume_d, vec3<i32>(i1.x, i.y, i.z), 0), f.x);
  let m10 = mix(textureLoad(lume_d, vec3<i32>(i.x, i1.y, i.z), 0), textureLoad(lume_d, vec3<i32>(i1.x, i1.y, i.z), 0), f.x);
  let m01 = mix(textureLoad(lume_d, vec3<i32>(i.x, i.y, i1.z), 0), textureLoad(lume_d, vec3<i32>(i1.x, i.y, i1.z), 0), f.x);
  let m11 = mix(textureLoad(lume_d, vec3<i32>(i.x, i1.y, i1.z), 0), textureLoad(lume_d, i1, 0), f.x);
  let m = mix(mix(m00, m10, f.y), mix(m01, m11, f.y), f.z);
  let k = 1.0 + 3.0 * g * dot(m.xyz, d) / max(m.w, 1e-6);
  return l0 * clamp(k, 0.0, 4.0);
}

// Lume's light at p (cells of its own grid, nl of them)
fn lume_at(p: vec3<f32>, nl: vec3<i32>) -> vec3<f32> {
  let q = clamp(p - vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(nl - vec3<i32>(1)));
  let i = vec3<i32>(floor(q));
  let f = q - vec3<f32>(i);
  let i1 = min(i + vec3<i32>(1), nl - vec3<i32>(1));
  let c000 = textureLoad(lume_t, i, 0).rgb;
  let c100 = textureLoad(lume_t, vec3<i32>(i1.x, i.y, i.z), 0).rgb;
  let c010 = textureLoad(lume_t, vec3<i32>(i.x, i1.y, i.z), 0).rgb;
  let c110 = textureLoad(lume_t, vec3<i32>(i1.x, i1.y, i.z), 0).rgb;
  let c001 = textureLoad(lume_t, vec3<i32>(i.x, i.y, i1.z), 0).rgb;
  let c101 = textureLoad(lume_t, vec3<i32>(i1.x, i.y, i1.z), 0).rgb;
  let c011 = textureLoad(lume_t, vec3<i32>(i.x, i1.y, i1.z), 0).rgb;
  let c111 = textureLoad(lume_t, i1, 0).rgb;
  return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
             mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}
fn mat_at(p: vec3<f32>) -> vec4<f32> { return trilinear(mat, p); }

fn trilinear(tx: texture_3d<f32>, p: vec3<f32>) -> vec4<f32> {
  let n = vec3<i32>(U.box.xyz);
  let q = clamp(p - vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(n - vec3<i32>(1)));
  let i = vec3<i32>(floor(q));
  let f = q - vec3<f32>(i);
  let i1 = min(i + vec3<i32>(1), n - vec3<i32>(1));
  let c000 = textureLoad(tx, i, 0);
  let c100 = textureLoad(tx, vec3<i32>(i1.x, i.y, i.z), 0);
  let c010 = textureLoad(tx, vec3<i32>(i.x, i1.y, i.z), 0);
  let c110 = textureLoad(tx, vec3<i32>(i1.x, i1.y, i.z), 0);
  let c001 = textureLoad(tx, vec3<i32>(i.x, i.y, i1.z), 0);
  let c101 = textureLoad(tx, vec3<i32>(i1.x, i.y, i1.z), 0);
  let c011 = textureLoad(tx, vec3<i32>(i.x, i1.y, i1.z), 0);
  let c111 = textureLoad(tx, i1, 0);
  return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
             mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}

fn sky_col(d: vec3<f32>) -> vec3<f32> {
  let up = max(d.y, 0.0);
  var s = mix(U.horizon.rgb, U.sky.rgb, pow(up, 0.45));
  let c = dot(d, U.sun.xyz);
  // the bright glow round the sun (haze droplets scattering forward), and its disc
  s += U.sun_col.rgb * (0.04 * hg(c, 0.76) + 0.02 * hg(c, 0.3));
  if (U.sun.w > 0.5 && c > 0.99996) { s += U.sun_col.rgb * 400.0; }
  return s;
}

// billowing noise in 0..1 at p (m of sky), about 0.5 on average: rounded lobes with sharp creases between
// (the sum of |noise|), three octaves from 900 m down to 220 m, drifting with the wind
fn billow(p: vec3<f32>) -> f32 {
  var q = (p - U.wind.xyz * U.wind.w) / 900.0;
  var s = 0.0;
  var a = 0.55;
  for (var i = 0; i < 3; i++) {
    s += a * abs(gnoise(q));
    q = q * 2.03 + vec3<f32>(3.1, -1.7, 2.3);
    a *= 0.5;
  }
  return clamp(s * 2.7, 0.0, 1.0);
}

// streaks in 0..1 (about 0.5 on average) for the precipitation: noise drawn out tall, as shafts of rain
// and hail and fallstreaks of snow are by falling
fn streak(p: vec3<f32>) -> f32 {
  let q = (p - U.wind.xyz * U.wind.w) * vec3<f32>(1.0 / 500.0, 1.0 / 4000.0, 1.0 / 500.0);
  return clamp(0.5 + 1.3 * (0.65 * gnoise(q) + 0.35 * gnoise(q * 2.3 + vec3<f32>(5.1, 1.3, -2.7))), 0.0, 1.0);
}

// the extinction drawn (x: the cloud's, y: its precipitation's) with detail finer than the grid. A cloud's
// edge is where the space round it, cov (0 outside, 1 inside, falling over about a cell either side of the
// edge the grid has: light volume, out_m.z), crosses billowing noise: cloud on one side, clear air on the
// other, so its surface is heaped into lobes up to about half a cell out and in, whatever its density. A
// thin cloud (less than a full one, droplet cloud 30 per km, ice cloud 2) is drawn as patches of full
// cloud, as much on average as it has; softer for ice. The rain, snow and hail inside the cloud are inside
// its edge; below and outside it they fall in soft shafts, streaked
fn detail(sc: f32, sp_in: f32, ice: f32, cov: f32, flat: f32, p: vec3<f32>) -> vec2<f32> {
  let er = clamp(U.look.w, 0.0, 1.0);
  if (er <= 0.0) { return vec2<f32>(sc, sp_in); }
  let inside = smoothstep(0.0, 0.3, cov);
  let sp = select(sp_in, sp_in * mix(0.4 + 1.2 * streak(p) * er + 0.6 * (1.0 - er), 1.0, inside), sp_in > 0.0 && inside < 1.0);
  if (cov <= 0.0) { return vec2<f32>(sc, sp); }
  if (cov >= 0.995) { return vec2<f32>(sc, sp); }
  let full = mix(0.03, 0.002, ice);
  let w = mix(0.45, mix(0.06, 0.2, ice), er);
  // (at a cloud's base the edge stays flat, where the rising air reaches its condensation level)
  let e = min(0.55 + 0.6 * (0.5 - billow(p)) * (1.0 - 0.85 * flat), 1.0 - w);
  let m = smoothstep(e - w, e + w, cov);
  return vec2<f32>(max(sc, full) * m, sp * mix(1.0, m, inside));
}

// near the box's open sides the cloud thins away (rather than ending in a wall where the box does)
fn side_fade(q: vec3<f32>) -> f32 {
  let d = min(min(q.x, U.box.x - q.x), min(q.z, U.box.z - q.z));
  return smoothstep(0.0, 6.0, d);
}

fn hash(p: vec2<f32>) -> f32 {
  let q = fract(p * vec2<f32>(0.1031, 0.1030));
  let r = q + dot(q, q.yx + 33.33);
  return fract((r.x + r.y) * r.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let w = u32(U.res.x);
  let h = u32(U.res.y);
  if (id.x >= w || id.y >= h) { return; }
  let px = vec2<i32>(id.xy);
  let ndc = vec2<f32>((f32(id.x) + 0.5) / U.res.x * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / U.res.y * 2.0);
  let fw = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
  let nw = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
  let fl = (U.w2l * vec4<f32>(fw.xyz / fw.w, 1.0)).xyz;
  let nl = (U.w2l * vec4<f32>(nw.xyz / nw.w, 1.0)).xyz;
  let dir = normalize(fl - nl);
  let scale = U.eye.w;
  let o = U.eye.xyz * scale;                 // the eye in the sky's metres
  let cell = U.box.w;
  let n = U.box.xyz;
  let bmin = U.org.xyz;
  let bmax = U.org.xyz + n * cell;
  let vis = max(U.org.w, 1.0);
  // what is behind: the ground (if the ray goes down to it) or the sky
  var t_far = 1.0e9;
  if (dir.y < -1e-5 && U.ground.w > 0.5) { t_far = -o.y / dir.y; }
  // the box
  let inv = 1.0 / select(dir, vec3<f32>(1e-9), abs(dir) < vec3<f32>(1e-9));
  let ta = (bmin - o) * inv;
  let tb = (bmax - o) * inv;
  var t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
  let t1 = min(min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z)), t_far);
  var L = vec3<f32>(0.0);
  var T = 1.0;
  var first = -1.0;
  let cos_sun = dot(dir, U.sun.xyz);
  let ms = clamp(U.look.y, 0.0, 1.0);
  if (t1 > t0) {
    let base_step = cell * max(U.res.z, 0.25);
    let fine = base_step * 0.35;
    var t = t0 + base_step * hash(vec2<f32>(id.xy) + U.res.w);
    var iters = 0;
    var coarse = true;
    var empty = 0;
    var last_dt = 0.0;   // the step just taken through clear air (0: none)
    loop {
      if (t > t1 || T < 0.004 || iters > 1600) { break; }
      iters++;
      let pw = o + dir * t;
      let pc = (pw - bmin) / cell;
      let lv = light_at(pc);
      if (lv.z + lv.w <= 1e-7) {
        // clear air: stride, or (just out of cloud) keep the fine step a little longer
        empty++;
        if (empty > 4) { coarse = true; }
        t += select(fine, base_step, coarse);
        last_dt = fine;
        continue;
      }
      if (coarse) {
        // found cloud while striding: back up and come in finely (every pixel meets the edge alike)
        coarse = false;
        empty = 0;
        let jit = fine * hash(vec2<f32>(id.yx) + U.res.w * 1.7);
        t = max(t - base_step, t0) + jit;
        last_dt = jit;
        continue;
      }
      empty = 0;
      let mt = mat_at(pc);
      let fd = side_fade(pc);
      let sd = detail(lv.z, fd * lv.w, mt.x, fd * mt.z, mt.w, pw);
      // (a dense cloud's surface drawn as if no denser than 50 per km: the eye sees some 20 m into it
      // instead of 2, which no shot of a sky can tell, and its sunlit skin is sampled smoothly)
      let sig = min(sd.x, DENSE) + sd.y;
      if (sig <= 1e-7) {
        t += fine;
        last_dt = fine;
        continue;
      }
      if (last_dt > 0.0) {
        // just come into cloud from clear air: find its surface between there and here, so every pixel
        // starts the cloud at its surface, not somewhere in the step under it (a dense cloud's sunlit skin
        // is metres deep)
        var lo = t - last_dt;
        var hi = t;
        for (var r = 0; r < 5; r++) {
          let tm = 0.5 * (lo + hi);
          let pm = o + dir * tm;
          let qm = (pm - bmin) / cell;
          let lm = light_at(qm);
          let mm = mat_at(qm);
          let fm = side_fade(qm);
          let dm = detail(lm.z, fm * lm.w, mm.x, fm * mm.z, mm.w, pm);
          if (dm.x + dm.y > 1e-7) { hi = tm; } else { lo = tm; }
        }
        last_dt = 0.0;
        t = hi;
        continue;
      }
      if (first < 0.0) { first = t; }
      // (finer steps in dense cloud, no more than about half an optical depth each: its sunlit skin, where
      // the light changes fastest, is sampled alike whatever depth the step lands at)
      let dt = clamp(0.5 / sig, fine * 0.05, fine);
      // the sunlight's optical depth: the light volume's; near a cloud's edge, through the detailed cloud
      // for the first two cells toward the sun (so the lobes finer than the light volume shade one another:
      // the sunlit side's relief), then the light volume's beyond. Six samples, closer together near the
      // point
      var tau_s = lv.x;
      if (mt.z > 0.0 && mt.z < 0.995) {
        tau_s = light_at((pw + U.sun.xyz * 2.0 * cell - bmin) / cell).x;
        var d0 = 0.0;
        for (var j = 0; j < 6; j++) {
          let d1 = cell * min(0.02 * pow(2.75, f32(j)), 2.0);   // 0.02, 0.06, 0.15, 0.42, 1.1, 2 cells
          let pj = pw + U.sun.xyz * (0.5 * (d0 + d1));
          let qj = (pj - bmin) / cell;
          let lj = light_at(qj);
          let mj = mat_at(qj);
          let fj = side_fade(qj);
          let dj = detail(lj.z, fj * lj.w, mj.x, fj * mj.z, mj.w, pj);
          tau_s += (min(dj.x, DENSE) + dj.y) * (d1 - d0);
          d0 = d1;
        }
      }
      // how much of what scatters here is ice
      let ice = (min(sd.x, DENSE) * mt.x + sd.y * mt.y) / sig;
      // sunlight, in octaves for the light scattered many times
      // (each octave: light that has scattered more, so less attenuated, less forward, and weaker; with
      // five of them a thick cloud's sunlit side reflects about as a white wall does)
      var sun = 0.0;
      var skyl = vec3<f32>(0.0);
      var bounce = vec3<f32>(0.0);
      if (U.lume.x > 0.5) {
        // Lume: the sun's light scattered here once, exactly (its phase, through the cloud toward the sun), and all the
        // light that has scattered before, the sky's and the ground's, traced (cloud_lume.wgsl), scattered toward the eye
        // by the phase's mean cosine (its flow: lume_toward)
        sun = phase(cos_sun, ice, U.look.x) * exp(-tau_s);
        let nl = vec3<i32>(i32(U.lume2.x), i32(U.lume.z), i32(U.lume.w));
        skyl = lume_toward(pc / U.lume.y, nl, dir, mix(0.63, 0.475, ice) * U.look.x);
      } else {
        var a = 1.0;
        var b = 1.0;
        var k = 1.0;
        for (var oc = 0; oc < 5; oc++) {
          sun += b * exp(-tau_s * a) * phase(cos_sun, ice, k * U.look.x);
          a *= 0.35;
          b *= 0.65 * ms + 0.001;
          k *= 0.45;
        }
        // (the sky seen from here, lv.y: mostly the blue overhead, some of the paler sky toward the horizon)
        skyl = mix(U.sky.rgb, U.horizon.rgb, 0.4) * U.sky.w * lv.y * (0.6 + 0.4 * (1.0 - ms));
        bounce = U.ground.rgb * U.sun_col.rgb * max(U.sun.y, 0.0) * 0.12 * exp(-0.3 * tau_s);
      }
      let src = (U.sun_col.rgb * sun * 4.0 * PI * 0.25 + skyl + bounce) * U.sun_col.w;
      // the air between the eye and here fades it toward the haze
      let air = exp(-t / vis);
      let st = exp(-sig * dt);
      L += T * (1.0 - st) * (src * air + U.horizon.rgb * (1.0 - air));
      T *= st;
      t += dt;
    }
  }
  // behind the clouds
  var back = vec3<f32>(0.0);
  var back_a = 0.0;
  var depth = 0.0;
  if (U.horizon.w > 0.5) {
    if (t_far < 1.0e8) {
      // the ground, in the clouds' shadows, faded by the air
      let pg = o + dir * t_far;
      let pc = (pg - bmin) / cell;
      var shade = 1.0;
      var wet = 0.0;
      if (all(pc.xz >= vec2<f32>(0.0)) && all(pc.xz <= n.xz)) {
        shade = exp(-light_at(vec3<f32>(pc.x, 0.5, pc.z)).x);
        let pr = textureLoad(precip, vec2<i32>(clamp(pc.xz, vec2<f32>(0.0), n.xz - vec2<f32>(1.0))), 0);
        wet = clamp((pr.x + pr.y + pr.z) * 3600.0 / 5.0, 0.0, 1.0);
      }
      let gl = U.ground.rgb * mix(1.0, 0.55, wet) * (U.sun_col.rgb * max(U.sun.y, 0.0) * shade + U.sky.rgb * 0.8);
      let air = exp(-t_far / vis);
      back = gl * air + U.horizon.rgb * (1.0 - air);
      depth = t_far;
    } else {
      back = sky_col(dir);
    }
    back_a = 1.0;
  }
  let col = L + T * back;
  let alpha = select(1.0 - T, 1.0, U.horizon.w > 0.5);
  textureStore(out_beauty, px, vec4<f32>(col, alpha));
  let d = select(depth, first, first > 0.0) / scale;   // in the scene's units, like the rest of the renderer
  textureStore(out_aux, px, vec4<f32>(1.0, d, 0.0, select(0.0, 1.0, d > 0.0)));
  textureStore(out_emit, px, vec4<f32>(0.0));
  textureStore(out_mask, px, vec4<f32>(0.0, 0.0, 0.0, 1.0 - T));
}
