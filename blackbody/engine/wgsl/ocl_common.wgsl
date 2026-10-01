// Shared by the open-water layer's kernels (ocean_layer.py): a square grid of cells over the water
// around the simulation box, and where the box sits on it.

struct Layer {
  map: vec4<f32>,   // corner x, z (fire-local m), cell size (m), cells across
  box: vec4<f32>,   // the box's corner x, z (m), its cell size (m), the source margin (box cells)
  bdim: vec4<f32>,  // the box's cells x, z; blend width (layer cells); step (s)
};

fn lay_n() -> i32 { return i32(U.L.map.w); }

// Fire-local metres (x, z) of the centre of layer cell c.
fn lay_pos(c: vec2<i32>) -> vec2<f32> { return U.L.map.xy + (vec2<f32>(c) + vec2<f32>(0.5)) * U.L.map.z; }

// Layer cell coordinates (texel units, centres at +0.5) of a point.
fn lay_coord(x: vec2<f32>) -> vec2<f32> { return (x - U.L.map.xy) / U.L.map.z; }

// How far a layer cell is inside the box's source region (0 outside .. 1 well inside): the box's
// footprint less its calming layer, where the simulation's own surface is the truth.
fn box_share(x: vec2<f32>) -> f32 {
  let b = (x - U.L.box.xy) / U.L.box.z;
  let m = U.L.box.w;
  let e = min(min(b.x - m, U.L.bdim.x - m - b.x), min(b.y - m, U.L.bdim.y - m - b.y)) * U.L.box.z / U.L.map.z;
  return smoothstep(0.0, U.L.bdim.z, e);
}

// Box column coordinates (cells, centres at +0.5) of a point.
fn box_coord(x: vec2<f32>) -> vec2<f32> { return (x - U.L.box.xy) / U.L.box.z; }

// How close a cell is to the layer's outer edge (0 inside .. 1 at the edge), for the sponge that
// takes waves out before they could wrap around.
fn edge_ramp(c: vec2<i32>) -> f32 {
  let n = f32(lay_n());
  let q = vec2<f32>(c) + vec2<f32>(0.5);
  let e = min(min(q.x, n - q.x), min(q.y, n - q.y));
  return 1.0 - smoothstep(0.0, 0.12 * n, e);
}

fn wrapi(i: vec2<i32>, n: i32) -> vec2<i32> { return ((i % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n); }

// Bilinear sample of a 2D texture at layer coordinates q (texel centres at +0.5), clamped to the edge.
fn lay_samp(t: texture_2d<f32>, q: vec2<f32>) -> vec4<f32> {
  let n = lay_n();
  let p = q - vec2<f32>(0.5);
  let i0 = vec2<i32>(floor(p));
  let f = p - floor(p);
  let lo = vec2<i32>(0);
  let hi = vec2<i32>(n - 1);
  let a = textureLoad(t, clamp(i0, lo, hi), 0);
  let b = textureLoad(t, clamp(i0 + vec2<i32>(1, 0), lo, hi), 0);
  let c = textureLoad(t, clamp(i0 + vec2<i32>(0, 1), lo, hi), 0);
  let d = textureLoad(t, clamp(i0 + vec2<i32>(1, 1), lo, hi), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Solid or dry here (an island, a pier, a hull at the water line, the beach above it).
fn lay_blocked(dep: vec4<f32>) -> bool { return dep.y <= 0.0 || dep.x <= 0.02; }

// Shoaling: how much taller waves of deep-water wavenumber k0 stand in water d metres deep, as they
// slow and bunch up over the shallows (energy flux kept: sqrt(cg_deep / cg); Eckart's k(d)).
fn shoal(k0: f32, d: f32) -> f32 {
  let x = k0 * max(d, 1e-3);
  if (x > 6.0) { return 1.0; }
  let k = k0 / sqrt(tanh(x));
  let kd = k * max(d, 1e-3);
  let n = 0.5 * (1.0 + 2.0 * kd / sinh(min(2.0 * kd, 40.0)));
  let cg = n * sqrt(9.81 * tanh(x) / k0);               // c = omega / k, omega = sqrt(g k0)
  let cg0 = 0.5 * sqrt(9.81 / k0);
  return sqrt(cg0 / max(cg, 1e-4));
}
