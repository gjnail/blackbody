// Draw embers as camera-facing streaks from where they were at shutter open to where they are now,
// added into the beauty and emission buffers.

struct Params {
  vp: mat4x4<f32>,     // world -> clip
  l2w: mat4x4<f32>,    // fire-local -> world
  res: vec4<f32>,      // width, height, shutter (s), brightness
  k: vec4<f32>,        // ambient K, flame K (brightness reference), dynamic range, min width (px)
  k2: vec4<f32>,       // log10 luminance at flame K, fade-in (s), focal length (px), _
};

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> D: array<vec4<f32>>;
@group(0) @binding(3) var bb: texture_2d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var<storage, read> E: array<vec4<f32>>;
@group(0) @binding(6) var mask: texture_2d<f32>;  // y = depth (clip w) * coverage of colliders in the shot, z = coverage
@group(1) @binding(0) var<uniform> U: Params;

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) col: vec3<f32>,
  @location(2) len_px: f32,
  @location(3) hw: f32,
  @location(4) depth: f32,
};

fn clip_of(p: vec3<f32>) -> vec4<f32> {
  return U.vp * (U.l2w * vec4<f32>(p, 1.0));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let a = A[ii];
  let b = B[ii];
  let d = D[ii];
  o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  o.uv = vec2<f32>(0.0);
  o.col = vec3<f32>(0.0);
  o.len_px = 0.0;
  o.hw = 1.0;
  if (a.w <= 0.0) { return o; }
  let p1 = a.xyz;
  let p0 = a.xyz - b.xyz * U.res.z;
  let c0 = clip_of(p0);
  let c1 = clip_of(p1);
  if (c0.w < 0.05 || c1.w < 0.05) { return o; }
  let res = U.res.xy;
  let s0 = (c0.xy / c0.w) * 0.5 * res;
  let s1 = (c1.xy / c1.w) * 0.5 * res;
  var axis = s1 - s0;
  let alen = length(axis);
  var dir = vec2<f32>(1.0, 0.0);
  if (alen > 1e-3) { dir = axis / alen; }
  let nrm = vec2<f32>(-dir.y, dir.x);
  // width: projected size, at least the minimum so tiny sparks still register
  let wpx = max(d.y * U.k2.z / c1.w, U.k.w);
  let half_w = 0.5 * wpx + 0.75;
  let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
  let along = corner.x;
  let side = corner.y * 2.0 - 1.0;
  let sp = mix(s0 - dir * half_w, s1 + dir * half_w, along) + nrm * side * half_w;
  let cw = mix(c0.w, c1.w, along);
  o.pos = vec4<f32>(sp / (0.5 * res) * cw, 0.5 * cw, cw);
  o.uv = vec2<f32>(along * (alen + 2.0 * half_w) - half_w, side * half_w);
  o.len_px = alen;
  o.hw = half_w;
  o.depth = cw;
  // colour: blackbody at the ember's temperature, or the line colour of the emitter's colourant
  // (fireworks) at the same brightness; energy spread over the streak length
  let T = b.w;
  let u = clamp((T - 400.0) / (6500.0 - 400.0), 0.0, 1.0);
  let bbv = textureSampleLevel(bb, lin, vec2<f32>(u, 0.5), 0.0);
  let bright = min(pow(10.0, (bbv.a - U.k2.x) * U.k.z), 1.0e4);
  let fade_in = clamp((d.w - a.w) / max(U.k2.y, 1e-3), 0.0, 1.0);
  let energy = U.res.w * bright * d.x * fade_in * wpx / (wpx + alen);
  let tint = E[ii];
  let line = tint.rgb * (luma(bbv.rgb) / max(luma(tint.rgb), 1e-4));
  o.col = mix(bbv.rgb, line, clamp(tint.w, 0.0, 1.0)) * energy;
  return o;
}

struct FOut {
  @location(0) beauty: vec4<f32>,
  @location(1) emit: vec4<f32>,
};

@fragment
fn fs(i: VOut) -> FOut {
  // distance to the streak's centre segment, in pixels
  let x = clamp(i.uv.x, 0.0, i.len_px);
  let dx = i.uv.x - x;
  let r2 = dx * dx + i.uv.y * i.uv.y;
  // hidden behind a collider in the shot
  let m = textureLoad(mask, vec2<i32>(i.pos.xy), 0);
  if (m.z > 0.5 && i.depth > m.y / m.z) { discard; }
  let f = exp(-r2 * 2.5 / max(i.hw * i.hw, 0.25));
  var o: FOut;
  let c = i.col * f;
  o.beauty = vec4<f32>(c, 0.0);
  o.emit = vec4<f32>(c, 0.0);
  return o;
}
