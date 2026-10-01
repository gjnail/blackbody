// Lava and other molten liquids in the march (included by liq_march.wgsl): the march only finds a molten
// surface and leaves the hit in a buffer for liq_lava_shade.wgsl, which shades it. The glow's light on
// what is around (the footage's ground and objects, stand-in colliders) is lit here, from a map of it on
// the ground (liq_lava_map.wgsl): one lookup, not a loop over lights at every place the march lights a surface.

@group(0) @binding(22) var<storage, read_write> lava_gbuf: array<vec4<f32>>;   // per pixel: (p, footprint m), (normal, keep), (ray, kind)
@group(0) @binding(23) var<storage, read> lava_map: array<vec4<f32>>;          // liq_lava_map.wgsl

fn lava_on() -> bool { return U.lava[0].x > 0.5; }

// Leave a surface hit for a pass that shades it over the march (kind 1 molten: liq_lava_shade.wgsl; 2 ice:
// liq_ice_shade.wgsl): grid point p, normal, ray, a pixel's footprint there (m), holdout keep.
fn gbuf_hit(px: vec2<i32>, p: vec3<f32>, nrm: vec3<f32>, rd: vec3<f32>, footprint: f32, keep: f32, kind: f32) {
  let k = u32(px.x + px.y * i32(U.res.x)) * 3u;
  if (k + 2u >= arrayLength(&lava_gbuf)) { return; }
  lava_gbuf[k] = vec4<f32>(p, footprint);
  lava_gbuf[k + 1u] = vec4<f32>(nrm, keep);
  lava_gbuf[k + 2u] = vec4<f32>(rd, kind);
}

// Light of the molten surface's glow reaching a surface at x (fire-local m) facing nrm (key-light units): the
// map's light at x (taken at ground level), what comes from one direction by how the surface faces it, the
// rest from all round at half.
fn lava_light_at(x: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
  let n = vec2<i32>(i32(U.lava[7].x), i32(U.lava[7].y));
  if (n.x < 2 || arrayLength(&lava_map) < u32(2 * n.x * n.y)) { return vec3<f32>(0.0); }
  let q = vec2<f32>((x.x - U.lava[6].x) / U.lava[6].z, (x.z - U.lava[6].y) / U.lava[6].w) - vec2<f32>(0.5);
  let inside = clamp(min(min(q.x + 0.5, f32(n.x) - 0.5 - q.x), min(q.y + 0.5, f32(n.y) - 0.5 - q.y)) / 4.0, 0.0, 1.0);
  let b = vec2<i32>(floor(q));
  let f = q - floor(q);
  var e = vec4<f32>(0.0);
  var d = vec3<f32>(0.0);
  for (var o = 0; o < 4; o++) {
    let oo = vec2<i32>(o & 1, o >> 1);
    let c = clamp(b + oo, vec2<i32>(0), n - vec2<i32>(1));
    let wv = mix(vec2<f32>(1.0) - f, f, vec2<f32>(oo));
    let k = u32(c.x + n.x * c.y) * 2u;
    e += wv.x * wv.y * lava_map[k];
    d += wv.x * wv.y * lava_map[k + 1u].xyz;
  }
  let facing = clamp((dot(nrm, normalize(d + vec3<f32>(0.0, 1e-4, 0.0))) + 0.15) / 1.15, 0.0, 1.0);
  return e.rgb * mix(0.5, facing, clamp(e.w, 0.0, 1.0)) * (U.lava[3].w * inside);
}

// The footage at screen position uv relit by the glow, where it shows the ground at x (fire-local m): what to
// add to it. The footage is taken to be lit by the key light and the sky; the glow adds to that light,
// softly limited far up (as the fire's light on the footage is).
fn lava_relight(uv: vec2<f32>, x: vec3<f32>, nrm: vec3<f32>) -> vec3<f32> {
  let amb = max(luma(U.sky.rgb) + luma(U.sun.rgb) * max(U.sundir.y, 0.0), 0.01);
  let r = lava_light_at(x, nrm) / amb;
  return backdrop(uv) * r / (vec3<f32>(1.0) + 0.25 * r);
}
