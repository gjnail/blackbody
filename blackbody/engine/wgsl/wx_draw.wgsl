// Weather: the precipitation particles drawn over the render (engine/weather_render.py), one quad each,
// stretched along its path over the shutter. Each kind looks as it does in the air:
//  - snowflakes: soft, bright aggregates of ice that scatter light forward (they glow backlit against a
//    low sun); one passing close to the lens shows its shape, a ragged clump of crystals, lacy at its
//    edge, the six-fold arms of the crystals in it faint; a rare sparkle from a facet; wet snow is greyer
//    and glassier;
//  - graupel: small, opaque white balls (rime);
//  - hail: balls of ice, white at the core and clearer at the rim, shaded by the light;
//  - ice pellets (sleet) and drops: tiny clear lenses, a spot of the sky behind and a glint of the sun,
//    seen as streaks.
// Particles behind what the march drew (the liquid, the ground, the stand-ins, the snow cover) are hidden.

struct Params {
  vp: mat4x4<f32>,      // world -> clip
  l2w: mat4x4<f32>,     // simulation -> world
  eye: vec4<f32>,       // camera (world), time (s)
  res: vec4<f32>,       // width, height, shutter (s, 0: none), focal length (px)
  sky: vec4<f32>,       // sky light (rgb), snow brightness
  sun: vec4<f32>,       // key light (rgb), _
  sundir: vec4<f32>,    // toward the key light (world), _
  jit: vec4<f32>,       // sub-pixel jitter (px), opacity, size scale
};

@group(0) @binding(0) var<storage, read> packed: array<vec4<f32>>;
@group(0) @binding(1) var aux: texture_2d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,         // along the streak (px), across it (px)
  @location(1) col: vec3<f32>,        // its light (premultiplied by coverage)
  @location(2) glint: vec3<f32>,
  @location(3) len_px: f32,
  @location(4) hw: f32,               // half width (px)
  @location(5) alpha: f32,
  @location(6) dist: f32,
  @location(7) look: vec4<f32>,       // kind, crystal angle, true radius (px), its own phase (0..1)
};

fn hash1(x: u32) -> f32 {
  var s = x * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return f32((s >> 22u) ^ s) * (1.0 / 4294967296.0);
}

fn fresnel_d(cosi: f32, ior: f32) -> f32 {
  let r0 = (1.0 - ior) / (1.0 + ior);
  let f0 = r0 * r0;
  return f0 + (1.0 - f0) * pow(1.0 - clamp(cosi, 0.0, 1.0), 5.0);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  o.uv = vec2<f32>(0.0);
  o.col = vec3<f32>(0.0);
  o.glint = vec3<f32>(0.0);
  o.len_px = 0.0;
  o.hw = 1.0;
  o.alpha = 0.0;
  o.dist = 0.0;
  o.look = vec4<f32>(0.0);
  let a = packed[ii * 3u];
  let b = packed[ii * 3u + 1u];
  let c = packed[ii * 3u + 2u];
  let kind = u32(c.x + 0.5);
  let melted = c.y;
  let p1 = (U.l2w * vec4<f32>(a.xyz, 1.0)).xyz;
  let vw = (U.l2w * vec4<f32>(b.xyz, 0.0)).xyz;
  // without motion blur the eye still sees fast drops, pellets and hail as streaks; snow as flakes
  var shut = U.res.z;
  if (shut <= 0.0 && kind != 1u && kind != 2u) { shut = 1.0 / 96.0; }
  let p0 = p1 - vw * shut;
  let c0 = U.vp * vec4<f32>(p0, 1.0);
  let c1 = U.vp * vec4<f32>(p1, 1.0);
  if (c0.w < 0.05 || c1.w < 0.05) { return o; }
  let res = U.res.xy;
  let js = vec2<f32>(-U.jit.x, U.jit.y);
  let s0 = (c0.xy / c0.w) * 0.5 * res + js;
  let s1 = (c1.xy / c1.w) * 0.5 * res + js;
  let axis = s1 - s0;
  let alen = length(axis);
  var dir = vec2<f32>(0.0, 1.0);
  if (alen > 1e-3) { dir = axis / alen; }
  let nrm = vec2<f32>(-dir.y, dir.x);
  let size = a.w * 1e-3 * U.jit.w;                      // m across
  let true_r = 0.5 * size * U.res.w / c1.w;             // px
  let wpx = max(2.0 * true_r, 0.9);
  let half_w = 0.5 * wpx + 0.75;
  let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
  let side = corner.y * 2.0 - 1.0;
  let sp = mix(s0 - dir * half_w, s1 + dir * half_w, corner.x) + nrm * side * half_w;
  let cw = mix(c0.w, c1.w, corner.x);
  o.pos = vec4<f32>(sp / (0.5 * res) * cw, 0.5 * cw, cw);
  o.uv = vec2<f32>(corner.x * (alen + 2.0 * half_w) - half_w, side * half_w);
  o.len_px = alen;
  o.hw = half_w;
  // (its distance along the ray from the near plane, as the liquid's march measures the depth it writes:
  // ndc depth is 1 - near / view depth, near enough with the far plane far)
  o.dist = length(p1 - U.eye.xyz) * clamp(c1.z / c1.w, 0.0, 1.0);
  let view = normalize(p1 - U.eye.xyz);
  let l = U.sundir.xyz;
  let cos_back = dot(view, l);          // 1 when the sun is straight behind it (backlit)
  // how much of its footprint it covers (sub-pixel ones cover less), and less along a longer streak
  let cover = U.jit.z * min(2.0 * true_r / 0.9, 1.0) * wpx / (wpx + alen);
  let seed = bitcast<u32>(b.w * 16777216.0);
  var col = vec3<f32>(0.0);
  var glint = vec3<f32>(0.0);
  var alpha = cover;
  if (kind == 1u || kind == 2u) {
    // snow and graupel: bright ice that scatters light round it, most of it forward
    let wet = clamp(melted * 1.6, 0.0, 1.0);
    let albedo = mix(0.92, 0.55, wet) * U.sky.w;
    let fwd = 0.35 + 1.6 * pow(max(cos_back, 0.0), 6.0) + 0.25 * max(-cos_back, 0.0);
    col = (U.sky.rgb * 0.95 + U.sun.rgb * fwd * 0.5) * albedo;
    alpha = cover * mix(1.0, 0.75, wet);
    // now and then a facet of a crystal flashes the sun at the eye
    if (kind == 1u && hash1(seed ^ u32(U.eye.w * 30.0)) < 0.02 * (1.0 - wet)) {
      glint = U.sun.rgb * 2.5;
    }
  } else if (kind == 3u) {
    // hail: a ball of ice, white in its frozen layers, glassy at the rim
    col = (U.sky.rgb * 0.8 + U.sun.rgb * (0.4 + 0.6 * max(-cos_back, 0.0))) * mix(0.85, 0.6, melted);
    glint = U.sun.rgb * fresnel_d(sqrt(max(0.5 * (1.0 - cos_back), 0.0)), 1.31) * 0.6;
  } else {
    // pellets and drops: tiny lenses
    let ior = select(1.31, 1.333, kind == 0u);
    col = U.sky.rgb * 1.6;
    glint = U.sun.rgb * (fresnel_d(sqrt(max(0.5 * (1.0 - cos_back), 0.0)), ior) * 0.25 + 2.5 * pow(max(cos_back, 0.0), 8.0));
    alpha = cover * select(0.9, 0.7, kind == 0u);
  }
  o.alpha = alpha;
  o.col = col * alpha;
  o.glint = glint * alpha * (0.6 + 0.8 * hash1(seed + 5u));
  o.look = vec4<f32>(f32(kind), b.w * 6.2831853 + U.eye.w * (0.5 + b.w), true_r, b.w);
  return o;
}

struct FOut {
  @location(0) beauty: vec4<f32>,
  @location(1) emit: vec4<f32>,
  @location(2) mask: vec4<f32>,
};

@fragment
fn fs(i: VOut) -> FOut {
  let a = textureLoad(aux, vec2<i32>(i.pos.xy), 0);
  if (a.w > 0.5 && a.y > 0.0 && i.dist > a.y + 0.01) { discard; }
  let x = clamp(i.uv.x, 0.0, i.len_px);
  let dx = i.uv.x - x;
  let d = vec2<f32>(dx, i.uv.y);
  let r2 = dot(d, d);
  let hw2 = max(i.hw * i.hw, 0.25);
  var f = exp(-r2 * 2.5 / hw2);
  var shade = 1.0;
  let kind = u32(i.look.x + 0.5);
  let rpx = i.look.z;
  if (rpx > 3.0 && i.len_px < rpx) {
    // close enough to show its shape
    let q = d / max(rpx, 1e-3);
    let rr = length(q);
    if (kind == 1u) {
      // a snowflake: a ragged clump of crystals (each flake its own outline, turning as it tumbles), lacy
      // toward its edge, the six-fold arms of its crystals faint in it
      let th = atan2(q.y, q.x) + i.look.y;
      let ph = i.look.w * 37.0;
      let edge = 0.78 + 0.1 * sin(2.0 * th + ph) + 0.08 * sin(3.0 * th + 1.7 * ph) + 0.05 * sin(5.0 * th + 2.9 * ph)
               + 0.04 * pow(abs(cos(3.0 * th)), 4.0);
      let lace = 0.8 + 0.2 * sin(19.0 * rr + 7.0 * th + ph) * sin(11.0 * rr - 4.0 * th);
      f = (1.0 - smoothstep(edge * 0.6, edge, rr)) * mix(1.0, lace, smoothstep(0.15, 0.7, rr));
    } else if (kind == 2u || kind == 3u) {
      // a ball: its disc, shaded toward the edge
      f = 1.0 - smoothstep(0.85, 1.0, rr);
      shade = 0.65 + 0.35 * sqrt(max(1.0 - rr * rr, 0.0)) + select(0.0, 0.15 * (1.0 - smoothstep(0.0, 0.6, rr)), kind == 3u);
    }
  }
  var o: FOut;
  o.beauty = vec4<f32>((i.col * shade + i.glint) * f, i.alpha * f);
  o.emit = vec4<f32>(i.glint * f, 0.0);
  o.mask = vec4<f32>(0.0, 0.0, 0.0, i.alpha * f);
  return o;
}
