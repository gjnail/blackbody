// Spray droplets: every spray particle drawn as a small drop, stretched into a streak over the
// shutter, over the liquid render (premultiplied "over", so it also hides what is behind it a
// little). A drop of water is a tiny lens: it shows a bright spot of the sky, a glint of the key
// light (from any direction, as a sphere always has one), and glows bright when backlit. Drops
// behind the liquid surface are hidden.

struct Params {
  vp: mat4x4<f32>,     // world -> clip
  l2w: mat4x4<f32>,    // fire-local -> world
  n: vec4<f32>,        // grid dims, metres per cell
  org: vec4<f32>,      // grid corner (fire-local m), amount (opacity scale)
  res: vec4<f32>,      // width, height, shutter (s), focal length (px)
  size: vec4<f32>,     // droplet size (m), min width (px), index of refraction, rainbow
  eye: vec4<f32>,      // camera position (world), _
  sky: vec4<f32>,      // sky light (rgb), _
  sun: vec4<f32>,      // key light (rgb)
  sundir: vec4<f32>,   // toward the key light (world)
  jit: vec4<f32>,      // sub-pixel jitter of this sample (pixels)
};

@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var aux: texture_2d<f32>;
@group(1) @binding(0) var<uniform> U: Params;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) col: vec3<f32>,
  @location(2) len_px: f32,
  @location(3) hw: f32,
  @location(4) alpha: f32,
  @location(5) dist: f32,
  @location(6) glint: vec3<f32>,
};

fn hash1(x: u32) -> f32 {
  var s = x * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return f32((s >> 22u) ^ s) * (1.0 / 4294967296.0);
}

// Sunlight a cloud of water drops sends back toward the eye along view direction v (unit, world),
// relative to the key light: the primary bow about 42 degrees from the point opposite the sun, red
// outside and violet inside, the fainter secondary near 51 degrees with its colours reversed, and
// the brighter sky inside the primary (light leaving the drops at every angle under the bow's).
fn rainbow(v: vec3<f32>, to_sun: vec3<f32>) -> vec3<f32> {
  let a = degrees(acos(clamp(dot(v, -to_sun), -1.0, 1.0)));
  let p1 = vec3<f32>(42.2, 41.3, 40.6);
  let p2 = vec3<f32>(50.4, 51.4, 53.0);
  let d1 = (vec3<f32>(a) - p1) / 0.75;
  let d2 = (vec3<f32>(a) - p2) / 0.9;
  let bow = exp(-0.5 * d1 * d1) + 0.35 * exp(-0.5 * d2 * d2);
  let inner = 0.12 * smoothstep(40.5, 30.0, a) * smoothstep(0.0, 8.0, a);
  return bow + vec3<f32>(inner);
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
  let pk = W[ii];
  let b = unpack2x16unorm(pk.y);
  let cls = u32(floor(b.y * 3.0 + 1e-4));
  if (cls != 0u) { return o; }   // only spray
  let q = vec3<f32>(unpack2x16unorm(pk.x), b.x);
  let vxy = unpack2x16float(pk.z);
  let vzs = unpack2x16float(pk.w);
  let local = U.org.xyz + q * U.n.xyz * U.n.w;
  let p1 = (U.l2w * vec4<f32>(local, 1.0)).xyz;
  let vel = (U.l2w * vec4<f32>(vxy, vzs.x, 0.0)).xyz;
  let p0 = p1 - vel * U.res.z;
  let c0 = U.vp * vec4<f32>(p0, 1.0);
  let c1 = U.vp * vec4<f32>(p1, 1.0);
  if (c0.w < 0.02 || c1.w < 0.02) { return o; }
  let res = U.res.xy;
  let js = vec2<f32>(-U.jit.x, U.jit.y);
  let s0 = (c0.xy / c0.w) * 0.5 * res + js;
  let s1 = (c1.xy / c1.w) * 0.5 * res + js;
  let axis = s1 - s0;
  let alen = length(axis);
  var dir = vec2<f32>(1.0, 0.0);
  if (alen > 1e-3) { dir = axis / alen; }
  let nrm = vec2<f32>(-dir.y, dir.x);
  let r = hash1(ii * 2654435761u + 17u);
  let size = U.size.x * (0.4 + 1.2 * r * r);
  let wpx = max(size * U.res.w / c1.w, U.size.y);
  let half_w = 0.5 * wpx + 0.75;
  let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
  let side = corner.y * 2.0 - 1.0;
  let sp = mix(s0 - dir * half_w, s1 + dir * half_w, corner.x) + nrm * side * half_w;
  let cw = mix(c0.w, c1.w, corner.x);
  o.pos = vec4<f32>(sp / (0.5 * res) * cw, 0.5 * cw, cw);
  o.uv = vec2<f32>(corner.x * (alen + 2.0 * half_w) - half_w, side * half_w);
  o.len_px = alen;
  o.hw = half_w;
  o.dist = length(p1 - U.eye.xyz);
  // what the drop shows: a spot of sky, the key light's glint, a bright glow when backlit
  let view = normalize(p1 - U.eye.xyz);
  let l = U.sundir.xyz;
  let cos_back = dot(view, l);
  let reflect_glint = fresnel_d(sqrt(max(0.5 * (1.0 - cos_back), 0.0)), U.size.z) * 0.25;
  let forward = 0.9 * pow(max(cos_back, 0.0), 12.0);
  // a drop covers only part of its footprint, and less the longer its streak
  let cover = U.org.w * 0.45 * wpx / (wpx + alen);
  o.alpha = cover;
  o.col = U.sky.rgb * 0.55 * cover;
  o.glint = U.sun.rgb * (reflect_glint + forward) * cover * (0.6 + 0.8 * hash1(ii * 747796405u + 3u));
  if (U.size.w > 0.0) { o.col += U.sun.rgb * rainbow(view, l) * (0.6 * U.size.w * cover); }
  return o;
}

struct FOut {
  @location(0) beauty: vec4<f32>,
  @location(1) emit: vec4<f32>,
  @location(2) mask: vec4<f32>,
};

@fragment
fn fs(i: VOut) -> FOut {
  // behind the liquid surface: hidden
  let a = textureLoad(aux, vec2<i32>(i.pos.xy), 0);
  if (a.w > 0.5 && a.y > 0.0 && i.dist > a.y + 0.02) { discard; }
  let x = clamp(i.uv.x, 0.0, i.len_px);
  let dx = i.uv.x - x;
  let r2 = dx * dx + i.uv.y * i.uv.y;
  let f = exp(-r2 * 2.5 / max(i.hw * i.hw, 0.25));
  var o: FOut;
  o.beauty = vec4<f32>((i.col + i.glint) * f, i.alpha * f);
  o.emit = vec4<f32>(i.glint * f, 0.0);
  o.mask = vec4<f32>(0.0, 0.0, 0.0, i.alpha * f);
  return o;
}
