// Rain: falling drops drawn as streaks (each drop's path over the shutter) over the liquid render,
// scattered through the view between a near and a far distance as densely as the rain rate and the
// drops' fall speed make them (drops per cubic metre = drops landing per square metre per second /
// fall speed). Each frame draws a fresh set. Like the spray droplets they are tiny lenses: a spot of
// the sky, a glint of the key light, bright when backlit. Drops behind the liquid surface or under
// the ground are hidden.

struct Params {
  vp: mat4x4<f32>,     // world -> clip
  inv_vp: mat4x4<f32>, // clip -> world
  eye: vec4<f32>,      // camera position (world), frame seed
  fall: vec4<f32>,     // drop velocity (world m/s, wind included), drop diameter (m)
  res: vec4<f32>,      // width, height, shutter (s), focal length (px)
  vol: vec4<f32>,      // near (m), far (m), drops drawn, opacity per drop
  sky: vec4<f32>,      // sky light (rgb), ground height (world y; ground on when w > 0)
  sun: vec4<f32>,      // key light (rgb), ground on
  sundir: vec4<f32>,   // toward the key light (world)
  jit: vec4<f32>,      // sub-pixel jitter of this sample (pixels)
};

@group(0) @binding(0) var aux: texture_2d<f32>;
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
  let s = ii * 2654435761u + u32(U.eye.w) * 40503u;
  // a place in the view: a screen position and a distance, even per volume
  let sx = hash1(s + 1u) * 2.1 - 1.05;
  let sy = hash1(s + 2u) * 2.1 - 1.05;
  let n3 = U.vol.x * U.vol.x * U.vol.x;
  let f3 = U.vol.y * U.vol.y * U.vol.y;
  let d = pow(n3 + hash1(s + 3u) * (f3 - n3), 1.0 / 3.0);
  let far = U.inv_vp * vec4<f32>(sx, sy, 0.5, 1.0);
  let ray = normalize(far.xyz / far.w - U.eye.xyz);
  let p1 = U.eye.xyz + ray * d;
  var p0 = p1 - U.fall.xyz * U.res.z;
  if (U.sun.w > 0.5) {
    if (p1.y < U.sky.w) { return o; }
    if (p0.y < U.sky.w) { p0 = mix(p1, p0, (p1.y - U.sky.w) / max(p1.y - p0.y, 1e-5)); }
  }
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
  let r = hash1(s + 4u);
  let size = U.fall.w * (0.5 + r);
  let true_w = size * U.res.w / c1.w;
  let wpx = max(true_w, 0.8);
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
  let view = normalize(p1 - U.eye.xyz);
  let l = U.sundir.xyz;
  let cos_back = dot(view, l);
  let reflect_glint = fresnel_d(sqrt(max(0.5 * (1.0 - cos_back), 0.0)), 1.333) * 0.25;
  let forward = 2.5 * pow(max(cos_back, 0.0), 8.0);
  // a drop covers only part of its footprint (less than a pixel far away), and less the longer its
  // streak; what it shows is brighter than its share of the sky, as it gathers light from a wide cone
  let cover = U.vol.w * 1.1 * min(true_w / 0.8, 1.0) * wpx / (wpx + alen);
  o.alpha = cover;
  o.col = U.sky.rgb * 1.6 * cover;
  o.glint = U.sun.rgb * (reflect_glint + forward) * cover * (0.6 + 0.8 * hash1(s + 5u));
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
