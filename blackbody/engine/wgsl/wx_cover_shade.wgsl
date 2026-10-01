// Weather: what lies on the ground drawn into the render (engine/weather_render.py), one thread per pixel,
// over a copy of what the march drew. The snow cover is a height field (the surface it lies on, from
// wx_surf.wgsl, plus its depth, from wx_cover.wgsl), traced along each eye ray; where it is nearer than
// what the march found it is shaded and its distance written for what is drawn after it:
//  - fresh snow is bright and slightly blue in its shaded hollows (light travels some way into it before
//    it scatters back out); old, wet snow is greyer and glossier;
//  - its crystals catch the sun: scattered glints, each a facet turned just right;
//  - a dusting thinner than its grains covers in patches, so the ground shows through;
//  - glaze from freezing rain, and water on bare ground, are a clear coat: they reflect the sky and the
//    sun over what is under them.
// The cover fades out over the last stretch of the weather area, which has no edge in the scene.
// Past the area, where no falling pieces are drawn, the fall is seen as a haze: the light of the rays'
// length outside the area is scattered out of them and replaced by the light the snow or rain scatters
// (extinction from the pieces' number and size: heavy snow closes the view to a few hundred metres).

struct Params {
  inv_vp: mat4x4<f32>,  // clip -> world
  w2l: mat4x4<f32>,     // world -> simulation
  eye: vec4<f32>,       // camera (simulation frame), _
  res: vec4<f32>,       // width, height, map nx, nz
  area: vec4<f32>,      // x0, z0, x1, z1 (m)
  yr: vec4<f32>,        // lowest and highest the cover reaches (m), march step (m), edge fade (m)
  sky: vec4<f32>,       // sky light (rgb), snow brightness
  sun: vec4<f32>,       // key light (rgb), sparkle
  sundir: vec4<f32>,    // toward the key light (simulation frame), glaze and wet gloss
  haze: vec4<f32>,      // the fall's haze colour (rgb), its extinction (1/m)
  hz: vec4<f32>,        // top of the falling air (m), _, _, _
};

@group(0) @binding(0) var cover_t: texture_2d<f32>;
@group(0) @binding(1) var surf_t: texture_2d<f32>;
@group(0) @binding(2) var beauty_in: texture_2d<f32>;
@group(0) @binding(3) var emit_in: texture_2d<f32>;
@group(0) @binding(4) var aux_in: texture_2d<f32>;
@group(0) @binding(5) var out_beauty: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_aux: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn hash2(p: vec2<f32>) -> f32 {
  let q = fract(p * vec2<f32>(0.1031, 0.1030));
  let r = q + dot(q, q.yx + 33.33);
  return fract((r.x + r.y) * r.x);
}

fn vnoise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x), mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// bilinear fetch of a map at (x, z) metres
fn map_at(t: texture_2d<f32>, x: f32, z: f32) -> vec4<f32> {
  let n = vec2<f32>(U.res.zw);
  let u = (x - U.area.x) / (U.area.z - U.area.x) * n.x - 0.5;
  let v = (z - U.area.y) / (U.area.w - U.area.y) * n.y - 0.5;
  let i = vec2<i32>(floor(vec2<f32>(u, v)));
  let f = vec2<f32>(u, v) - floor(vec2<f32>(u, v));
  let hi = vec2<i32>(n) - vec2<i32>(1);
  let a = textureLoad(t, clamp(i, vec2<i32>(0), hi), 0);
  let b = textureLoad(t, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), hi), 0);
  let c = textureLoad(t, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), hi), 0);
  let d = textureLoad(t, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), hi), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn edge_fade(x: f32, z: f32) -> f32 {
  let e = min(min(x - U.area.x, U.area.z - x), min(z - U.area.y, U.area.w - z));
  return smoothstep(0.0, max(U.yr.w, 1e-3), e);
}

// snow depth that shows at (x, z): thin dustings in patches
fn shown_depth(x: f32, z: f32, depth: f32) -> f32 {
  let patchy = vnoise(vec2<f32>(x, z) * 40.0) * 0.6 + vnoise(vec2<f32>(x, z) * 9.0) * 0.4;
  let cover = smoothstep(0.0, 0.004, depth * (0.4 + 1.2 * patchy));
  return depth * cover * edge_fade(x, z);
}

fn height(x: f32, z: f32) -> f32 {
  let s = map_at(surf_t, x, z);
  let c = map_at(cover_t, x, z);
  return s.x + shown_depth(x, z, c.x);
}

fn fresnel(c: f32, f0: f32) -> f32 { return f0 + (1.0 - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let w = u32(U.res.x);
  let h = u32(U.res.y);
  if (id.x >= w || id.y >= h) { return; }
  let px = vec2<i32>(id.xy);
  var col = textureLoad(beauty_in, px, 0);
  var em = textureLoad(emit_in, px, 0);
  var ax = textureLoad(aux_in, px, 0);
  // the eye ray, in the simulation's frame
  let ndc = vec2<f32>((f32(id.x) + 0.5) / U.res.x * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / U.res.y * 2.0);
  let far_w = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
  let near_w = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
  let fw = (U.w2l * vec4<f32>(far_w.xyz / far_w.w, 1.0)).xyz;
  let nw = (U.w2l * vec4<f32>(near_w.xyz / near_w.w, 1.0)).xyz;
  let o = U.eye.xyz;
  let dir = normalize(fw - nw);
  // (what the march drew is at depth ax.y along the ray from the near plane, as the liquid's march
  // measures it; here distances run from the eye)
  let t_near = length(nw - o);
  let limit = select(1.0e6, ax.y + t_near, ax.w > 0.5 && ax.y > 0.0);
  // the slab the cover can be in, within the area
  let lo = vec3<f32>(U.area.x, U.yr.x, U.area.y);
  let hi = vec3<f32>(U.area.z, U.yr.y, U.area.w);
  let inv = 1.0 / select(dir, vec3<f32>(1e-9), abs(dir) < vec3<f32>(1e-9));
  let ta = (lo - o) * inv;
  let tb = (hi - o) * inv;
  let t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
  let t1 = min(min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z)), limit + 0.02);
  var hit = false;
  var t = t0;
  if (t1 > t0) {
    let stepl = U.yr.z;
    var prev = t0;
    var prev_d = 1.0;
    var iters = 0;
    loop {
      if (t > t1 || iters > 600) { break; }
      iters++;
      let p = o + dir * t;
      let d = p.y - height(p.x, p.z);
      if (d <= 0.0) {
        // refine between the last step above and this one
        var a = prev;
        var b = t;
        for (var k = 0; k < 6; k++) {
          let m = 0.5 * (a + b);
          let q = o + dir * m;
          if (q.y - height(q.x, q.z) <= 0.0) { b = m; } else { a = m; }
        }
        t = b;
        hit = true;
        break;
      }
      prev = t;
      prev_d = d;
      // half the way to where the ray would meet a flat surface at this height (no less than a map cell)
      t += clamp(0.5 * d / max(-dir.y, 0.02), stepl, 0.25);
    }
  }
  if (hit) {
    let p = o + dir * t;
    let c = map_at(cover_t, p.x, p.z);
    let s = map_at(surf_t, p.x, p.z);
    let snow = shown_depth(p.x, p.z, c.x);
    // the normal of the snow surface
    let e = 0.5 * (U.area.z - U.area.x) / U.res.z;
    let hx = height(p.x + e, p.z) - height(p.x - e, p.z);
    let hz = height(p.x, p.z + e) - height(p.x, p.z - e);
    let n = normalize(vec3<f32>(-hx, 2.0 * e, -hz));
    // (a steep face of the height field over surfaces much less steep themselves is the step at an
    // object's edge, not snow: what the march drew there stands. A steep bank's snow is steep too)
    let nys = sqrt(max(1.0 - dot(s.zw, s.zw), 0.0));
    let fake_wall = n.y < 0.45 && n.y < nys - 0.3;
    if (snow > 1e-4 && t <= limit + 0.01 && !fake_wall) {
      let l = U.sundir.xyz;
      let v = -dir;
      let wet = clamp(c.z, 0.0, 1.0);
      let albedo = mix(0.9, 0.62, wet) * U.sky.w;
      let ndl = dot(n, l);
      // snow's diffusion wraps the light a little round into its shade, and tints what scatters deep blue
      let wrap = max((ndl + 0.25) / 1.25, 0.0);
      let deep = vec3<f32>(0.80, 0.90, 1.0);
      var lit = U.sun.rgb * wrap * mix(deep, vec3<f32>(1.0), smoothstep(0.0, 0.6, ndl));
      let sky_v = 0.55 + 0.45 * n.y;
      lit += U.sky.rgb * sky_v * mix(deep, vec3<f32>(1.0), 0.6);
      var rgb = lit * albedo;
      // a wet surface is glossy
      let hv = normalize(l + v);
      let spec = pow(max(dot(n, hv), 0.0), mix(8.0, 120.0, wet)) * fresnel(dot(v, n), 0.02) * mix(0.1, 1.5, wet);
      rgb += U.sun.rgb * spec * max(ndl, 0.0);
      // crystals catching the sun: a facet every few millimetres, turned at random
      let cell = floor(p.xz * 900.0);
      let rnd = vec3<f32>(hash2(cell), hash2(cell + 17.0), hash2(cell + 41.0));
      let facet = normalize(n + (rnd - 0.5) * 1.6);
      let sparkle = pow(max(dot(facet, hv), 0.0), 900.0) * step(0.6, rnd.x) * U.sun.w * (1.0 - wet);
      let glint = U.sun.rgb * sparkle * 40.0 * max(ndl, 0.0);
      col = vec4<f32>(rgb + glint, 1.0);
      em = vec4<f32>(em.rgb + glint, em.w);
      ax = vec4<f32>(1.0, t - t_near, 0.0, 1.0);
    } else if (t <= limit + 0.01) {
      // bare: glaze or a wet surface is a clear, reflecting coat over what is there
      let gloss = clamp(c.w / 0.0005, 0.0, 1.0) + 0.5 * clamp(c.z, 0.0, 1.0) * (1.0 - step(1e-4, c.x));
      if (gloss > 0.01 && U.sundir.w > 0.0) {
        let n = vec3<f32>(s.z, 1.0, s.w);
        let nn = normalize(n);
        let v = -dir;
        let fr = fresnel(dot(v, nn), 0.02) * clamp(gloss, 0.0, 1.0) * U.sundir.w * edge_fade(p.x, p.z);
        let hv = normalize(U.sundir.xyz + v);
        let spec = pow(max(dot(nn, hv), 0.0), 400.0) * 20.0;
        let refl = U.sky.rgb * 1.1 + U.sun.rgb * spec;
        col = vec4<f32>(col.rgb * (1.0 - fr) + refl * fr, col.w);
        em = vec4<f32>(em.rgb + U.sun.rgb * spec * fr * 0.3, em.w);
      }
    }
  }
  // the fall's haze over the rays' length outside the area (below the air it falls from)
  if (U.haze.w > 0.0) {
    var d_end = select(1.0e4, ax.y + t_near, ax.w > 0.5 && ax.y > 0.0);
    // (up to where the ray leaves the falling air, if it climbs out of it)
    if (dir.y > 1e-4) { d_end = min(d_end, max((U.hz.x - o.y) / dir.y, 0.0)); }
    let lo2 = vec3<f32>(U.area.x, -1.0e3, U.area.y);
    let hi2 = vec3<f32>(U.area.z, U.hz.x, U.area.w);
    let ia = (lo2 - o) * inv;
    let ib = (hi2 - o) * inv;
    let e0 = max(max(max(min(ia.x, ib.x), min(ia.y, ib.y)), min(ia.z, ib.z)), 0.0);
    let e1 = min(min(max(ia.x, ib.x), max(ia.y, ib.y)), max(ia.z, ib.z));
    var inside = 0.0;
    if (e1 > e0) { inside = max(min(e1, d_end) - e0, 0.0); }
    let d_out = max(d_end - inside, 0.0);
    let tr = exp(-U.haze.w * d_out);
    col = vec4<f32>(col.rgb * tr + U.haze.rgb * (1.0 - tr), col.w);
  }
  textureStore(out_beauty, px, col);
  textureStore(out_emit, px, em);
  textureStore(out_aux, px, ax);
}
