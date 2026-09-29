// Surface builder, pass 4: smoothing along the surface. Each node averages the distance on a ring
// in its tangent plane (perpendicular to the distance gradient). That is mean-curvature flow on
// the level sets: bumps and particle graininess flatten out, while a flat surface stays exactly
// where it is (the distance is the same all around the ring). It is weighted by how dense the
// particles are, so the bulk of the liquid gets calm and glassy but thin sheets and droplets keep
// their shape.

struct Params {
  nf: vec4<f32>,   // surface grid dims; w = ring radius (surface cells)
  k: vec4<f32>,    // strength (0..1)
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var ww: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn at(q: vec3<f32>) -> f32 {
  return textureSampleLevel(src, lin, q / U.nf.xyz, 0.0).x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ni = vec3<i32>(U.nf.xyz);
  let c = vec3<i32>(id);
  if (any(c >= ni)) { return; }
  let s = textureLoad(src, c, 0);
  // how dense the particles are here, or a cell away (nodes just outside the surface count too)
  var dense = textureLoad(ww, c, 0).w;
  for (var i = 0; i < 6; i++) {
    var o = vec3<i32>(0);
    o[i >> 1] = select(-1, 1, (i & 1) == 1);
    dense = max(dense, textureLoad(ww, clamp(c + o, vec3<i32>(0), ni - vec3<i32>(1)), 0).w);
  }
  let w = U.k.x * dense;
  // only near the surface, and only where the particles are dense
  if (w <= 0.0 || abs(s.x) > 2.5) {
    textureStore(dst, c, s);
    return;
  }
  let q = vec3<f32>(c) + vec3<f32>(0.5);
  let g = vec3<f32>(at(q + vec3<f32>(1.0, 0.0, 0.0)) - at(q - vec3<f32>(1.0, 0.0, 0.0)),
                    at(q + vec3<f32>(0.0, 1.0, 0.0)) - at(q - vec3<f32>(0.0, 1.0, 0.0)),
                    at(q + vec3<f32>(0.0, 0.0, 1.0)) - at(q - vec3<f32>(0.0, 0.0, 1.0)));
  let gl = length(g);
  if (gl < 1e-4) {
    textureStore(dst, c, s);
    return;
  }
  let n = g / gl;
  var t1 = cross(n, vec3<f32>(0.0, 1.0, 0.0));
  if (dot(t1, t1) < 0.01) { t1 = cross(n, vec3<f32>(1.0, 0.0, 0.0)); }
  t1 = normalize(t1);
  let t2 = cross(n, t1);
  let r = U.nf.w;
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    let a = f32(i) * 0.7853982;
    sum += at(q + (t1 * cos(a) + t2 * sin(a)) * r);
  }
  textureStore(dst, c, vec4<f32>(mix(s.x, sum / 8.0, w), s.yzw));
}
