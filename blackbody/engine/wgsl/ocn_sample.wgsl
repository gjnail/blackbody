// The sea on the open water, sampled from the layers ocean.py builds each frame (ocn_spectrum.wgsl,
// ocn_fft.wgsl, ocn_foam.wgsl). The including file declares:
//   oc_w: texture_3d<f32>   the layers: 2l = a (height, displacement x, z, d(disp x)/dz),
//                           2l + 1 = b (dh/dx, dh/dz, d(disp x)/dx, d(disp z)/dz) of spectrum layer l
//                           (l = 0..3 the cascades, 4..5 the waves the simulation carries)
//   oc_f: texture_3d<f32>   the foam (one layer, on the largest cascade tile): fresh, aged, bubbles, height
//   rep: sampler            linear, repeating
//   U.ocn   (on, 1/tile 0, 1/tile 1, level (m))
//   U.ocn2  (_, whitecaps, crest bound (m), rainbow)
//   U.ocx   array<vec4<f32>, 12>:
//     0  1/tile of the cascades 0..3     1  1/tile of layers 4 and 5, texels across, 1/texture layers
//     2  mean square slope of the cascades 0..3
//     3  foam frame drift (in units of the largest tile) x, z; mean square slope of the ripples past
//        the last cascade; pixel angle (rad)
//     4  foam gain, streaks, spread of the Jacobian, bubble glow   5  wind direction (x, z), speed (m/s), gusts
//     6  crest glow colour (rgb), crest glow amount            7  current (x, z m/s), time (s), whitecap Jacobian
//     8  the open-water layer (ocean_layer.py, in oc_l1): its corner x, z (m), 1 / its size (twice)
//     9  the layer on (1/0), its texels across, the sea's significant height (m), its peak wavenumber (1/m)
//     10  surge: height (m, 0 none), 2 / length (1/m), direction (x, z)
//     11  surge: crest position along its direction (m), kind (1 bore), speed (m/s), depth (m)
//     12  the stand-ins' colour (rgb), set (1/0)
//     13  the box's sides the open water flows through (-x, +x, -z, +z; 1/0): the others are walls
// Positions are fire-local metres (x, z). `fp` is the footprint of a pixel on the water (m): waves
// much shorter than it are left out of the geometry and go into the roughness instead.

fn ocean_on() -> bool { return U.ocn.x > 0.5; }


fn oc_inv(l: i32) -> f32 {
  if (l < 4) { return U.ocx[0][l]; }
  return U.ocx[1][l - 4];
}

// Layer k (0 .. 2 * layers - 1) of the sea at x0 (m): spectrum layer k / 2.
fn oc_tex(x0: vec2<f32>, k: i32) -> vec4<f32> {
  let w = (f32(k) + 0.5) * U.ocx[1].w;
  return textureSampleLevel(oc_w, rep, vec3<f32>(x0 * oc_inv(k / 2), w), 0.0);
}

// How much of each cascade to keep for a pixel footprint fp (m): 1 while its waves span several
// pixels, fading out as they shrink under one.
fn ocean_lod(fp: f32) -> vec4<f32> {
  let texel = vec4<f32>(1.0) / (U.ocx[0] * max(U.ocx[1].z, 1.0));
  return vec4<f32>(1.0) - smoothstep(2.0 * texel, 12.0 * texel, vec4<f32>(fp));
}

// Mean square slope of the waves left out at footprint fp (they roughen the surface instead).
fn ocean_rough2(fp: f32) -> f32 {
  if (!ocean_on()) { return 0.0; }
  return dot(vec4<f32>(1.0) - ocean_lod(fp), U.ocx[2]) + U.ocx[3].z;
}

// Smooth value noise in -1..1 (self-contained: the caustics pass includes this file too).
fn oc_hash(i: vec3<i32>) -> f32 {
  var s = u32(i.x) * 73856093u ^ u32(i.y) * 19349663u ^ u32(i.z) * 83492791u;
  s = s * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return f32((s >> 22u) ^ s) * (2.0 / 4294967296.0) - 1.0;
}

fn oc_noise(p: vec3<f32>) -> f32 {
  let i = vec3<i32>(floor(p));
  let f = p - floor(p);
  let u = f * f * (vec3<f32>(3.0) - 2.0 * f);
  var s = 0.0;
  for (var j = 0; j < 8; j++) {
    let o = vec3<i32>(j & 1, (j >> 1) & 1, j >> 2);
    let w = select(vec3<f32>(1.0) - u, u, o == vec3<i32>(1));
    s += w.x * w.y * w.z * oc_hash(i + o);
  }
  return s;
}

// Wind gusts: patches of stronger wind (cat's paws) drifting downwind, that roughen the ripples.
// 1 on average; the gusts setting sets how much it varies.
fn ocean_gust(x: vec2<f32>) -> f32 {
  let g = U.ocx[5].w;
  if (g <= 0.0) { return 1.0; }
  let wd = U.ocx[5].xy;
  let across = vec2<f32>(-wd.y, wd.x);
  let t = U.ocx[7].z;
  let q = x - wd * (0.8 * U.ocx[5].z * t);
  let s = vec2<f32>(dot(q, wd) / 22.0, dot(q, across) / 13.0);
  let n = oc_noise(vec3<f32>(s, t * 0.04)) + 0.5 * oc_noise(vec3<f32>(s * 2.3 + vec2<f32>(7.1, 3.3), t * 0.07));
  return max(1.0 + g * 1.8 * n, 0.05);
}

// ---- the open-water layer: waves spreading from the box, foam, the sea bed -----------------------

fn lay_on() -> bool { return U.ocx[9].x > 0.5; }

// The layer at x: (wave height (m), foam, water depth (m; < 0 solid at the water line), current
// speed (m/s)). Past its edge: still, deep water.
fn lay_at(x: vec2<f32>) -> vec4<f32> {
  let none = vec4<f32>(0.0, 0.0, 6.0e4, 0.0);
  if (!lay_on()) { return none; }
  let uv = (x - U.ocx[8].xy) * U.ocx[8].zw;
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return none; }
  let hn = 0.5 / max(U.ocx[9].y, 1.0);
  return textureSampleLevel(oc_l1, rep, clamp(uv, vec2<f32>(hn), vec2<f32>(1.0 - hn)), 0.0);
}

// The layer's wave slope at x (d/dx, d/dz).
fn lay_slope(x: vec2<f32>) -> vec2<f32> {
  if (!lay_on()) { return vec2<f32>(0.0); }
  let dl = 1.0 / (max(U.ocx[9].y, 1.0) * U.ocx[8].z);
  let ex = vec2<f32>(dl, 0.0);
  let ez = vec2<f32>(0.0, dl);
  return vec2<f32>(lay_at(x + ex).x - lay_at(x - ex).x, lay_at(x + ez).x - lay_at(x - ez).x) / (2.0 * dl);
}

// How much taller waves of deep-water wavenumber k0 stand in water d metres deep (shoaling: energy
// flux kept as they slow, sqrt(cg_deep / cg), Eckart's k(d)).
fn oc_shoal(k0: f32, d: f32) -> f32 {
  let x = k0 * max(d, 1e-3);
  if (x > 6.0) { return 1.0; }
  let k = k0 / sqrt(tanh(x));
  let kd = k * max(d, 1e-3);
  let nn = 0.5 * (1.0 + 2.0 * kd / sinh(min(2.0 * kd, 40.0)));
  return sqrt(0.5 * sqrt(9.81 / k0) / max(nn * sqrt(9.81 * tanh(x) / k0), 1e-4));
}

// How the sea's waves feel the bottom at depth d: the long ones grow as they shoal, then break
// (the significant height stays under about 0.6 of the depth); the short ones only die away at
// the very water line. (long cascades, short cascades)
fn ocean_depth_w(d: f32) -> vec2<f32> {
  if (d > 1.0e4 || U.ocx[9].z <= 0.0) { return vec2<f32>(1.0); }
  if (d <= 0.0) { return vec2<f32>(0.0); }
  let a = min(oc_shoal(U.ocx[9].w, d), 0.6 * d / U.ocx[9].z);
  return vec2<f32>(a, clamp(d / 0.3, 0.0, 1.0));
}

// Breaking over the shallows at depth d: 0 none .. 1 the waves there all break (surf).
fn ocean_surf(d: f32) -> f32 {
  if (d > 1.0e4 || U.ocx[9].z <= 0.0 || d <= 0.0) { return 0.0; }
  return smoothstep(0.55, 0.9, U.ocx[9].z * oc_shoal(U.ocx[9].w, d) / d);
}

// The cascades' weights at footprint fp over the bottom at x.
fn ocean_w(x: vec2<f32>, fp: f32) -> vec4<f32> {
  let dw = ocean_depth_w(lay_at(x).z);
  return ocean_lod(fp) * vec4<f32>(dw.x, dw.x, dw.y, dw.y);
}

// Height and horizontal displacement at the undisplaced point x0, cascades weighted by w.
fn ocean_disp_w(x0: vec2<f32>, w: vec4<f32>) -> vec3<f32> {
  return oc_tex(x0, 0).xyz * w.x + oc_tex(x0, 2).xyz * w.y + oc_tex(x0, 4).xyz * w.z + oc_tex(x0, 6).xyz * w.w;
}

fn ocean_disp(x0: vec2<f32>, fp: f32) -> vec3<f32> { return ocean_disp_w(x0, ocean_lod(fp)); }

// The undisplaced point whose choppy displacement lands on x (the finer cascades' displacement is
// centimetres at most: it is left out here).
fn ocean_x0_w(x: vec2<f32>, w: vec4<f32>) -> vec2<f32> {
  var x0 = x;
  for (var i = 0; i < 2; i++) { x0 = x - (oc_tex(x0, 0).yz * w.x + oc_tex(x0, 2).yz * w.y); }
  return x0;
}

fn ocean_x0(x: vec2<f32>, fp: f32) -> vec2<f32> { return ocean_x0_w(x, ocean_w(x, fp)); }

// The surge at x: a solitary long wave (a sech^2 hump) or a bore (a tanh front behind which the
// water stays raised), travelling along its direction. (height (m), slope along x, slope along z)
fn ocean_surge(x: vec2<f32>) -> vec3<f32> {
  let H = U.ocx[10].x;
  if (H <= 0.0) { return vec3<f32>(0.0); }
  let q = (dot(x, U.ocx[10].zw) - U.ocx[11].x) * U.ocx[10].y;
  let th = tanh(clamp(q, -20.0, 20.0));
  let sech2 = 1.0 - th * th;
  if (U.ocx[11].y > 1.5) {
    // a tsunami: a bore, with the trough the sea draws back into running ahead of it
    let tp = tanh(clamp((q - 6.0) / 3.0, -20.0, 20.0));
    let s2p = 1.0 - tp * tp;
    return vec3<f32>(0.5 * H * (1.0 - th) - 0.6 * H * s2p,
                     (-0.5 * H * U.ocx[10].y * sech2 + 0.4 * H * U.ocx[10].y * s2p * tp) * U.ocx[10].zw);
  }
  if (U.ocx[11].y > 0.5) {
    return vec3<f32>(0.5 * H * (1.0 - th), -0.5 * H * U.ocx[10].y * sech2 * U.ocx[10].zw);
  }
  return vec3<f32>(H * sech2, -2.0 * H * U.ocx[10].y * sech2 * th * U.ocx[10].zw);
}

// Surface height above the level (m) at x: the sea's waves (feeling the bottom), a surge, and the
// waves spreading from the box.
fn ocean_eta(x: vec2<f32>, fp: f32) -> f32 {
  if (!ocean_on()) { return 0.0; }
  let la = lay_at(x);
  let dw = ocean_depth_w(la.z);
  let w = ocean_lod(fp) * vec4<f32>(dw.x, dw.x, dw.y, dw.y);
  let x0 = ocean_x0_w(x, w);
  // (the layer's waves stop where the swash zone begins: what the box hands it from a wave running
  // up the beach does not stand as water on the sand past the box)
  let wet = smoothstep(0.1, 0.4, la.z / max(U.ocx[9].z, 0.05));
  return oc_tex(x0, 0).x * w.x + oc_tex(x0, 2).x * w.y + oc_tex(x0, 4).x * w.z + oc_tex(x0, 6).x * w.w
       + ocean_surge(x).x + la.x * wet;
}

// Derivatives of the displaced surface at x0, the cascades weighted by w:
// (dh/dx, dh/dz, d(disp x)/dx, d(disp z)/dz) and d(disp x)/dz.
struct OcDeriv {
  b: vec4<f32>,
  dxz: f32,
};

fn ocean_deriv(x0: vec2<f32>, w: vec4<f32>) -> OcDeriv {
  var o: OcDeriv;
  o.b = oc_tex(x0, 1) * w.x + oc_tex(x0, 3) * w.y + oc_tex(x0, 5) * w.z + oc_tex(x0, 7) * w.w;
  o.dxz = oc_tex(x0, 0).w * w.x + oc_tex(x0, 2).w * w.y + oc_tex(x0, 4).w * w.z + oc_tex(x0, 6).w * w.w;
  return o;
}

// The same for the waves the simulation carries (layers 4 and 5, the low parts of cascades 0 and 1).
fn ocean_deriv_low(x0: vec2<f32>, w: vec4<f32>) -> OcDeriv {
  var o: OcDeriv;
  o.b = oc_tex(x0, 9) * w.x + oc_tex(x0, 11) * w.y;
  o.dxz = oc_tex(x0, 8).w * w.x + oc_tex(x0, 10).w * w.y;
  return o;
}

// Normal of the displaced surface from its derivatives (unnormalised, y = the Jacobian).
fn ocean_n_of(d: OcDeriv) -> vec3<f32> {
  let b = d.b;
  let jy = (1.0 + b.z) * (1.0 + b.w) - d.dxz * d.dxz;
  return vec3<f32>(b.y * d.dxz - (1.0 + b.w) * b.x, max(jy, 0.08), d.dxz * b.x - (1.0 + b.z) * b.y);
}

// The cascades' weights at footprint fp over the bottom, with the ripples roughened in a gust.
fn ocean_gust_w(x: vec2<f32>, fp: f32) -> vec4<f32> {
  var w = ocean_w(x, fp);
  let g = ocean_gust(x);
  w.z *= mix(1.0, g, 0.5);
  w.w *= g;
  return w;
}

// Unit surface normal at x.
fn ocean_normal(x: vec2<f32>, fp: f32) -> vec3<f32> {
  if (!ocean_on()) { return vec3<f32>(0.0, 1.0, 0.0); }
  let w = ocean_gust_w(x, fp);
  let n = ocean_n_of(ocean_deriv(ocean_x0_w(x, w), w));
  let sl = ocean_surge(x).yz + lay_slope(x);
  return normalize(n - vec3<f32>(sl.x, 0.0, sl.y) * n.y);
}

// How far the surface has folded over at x (whitecaps where crests break): 0 none .. 1 fully.
// Crests break at the scale of the waves that carry the energy: ripples folding over make no foam.
fn ocean_fold(x: vec2<f32>, fp: f32) -> f32 {
  if (!ocean_on()) { return 0.0; }
  let w = ocean_w(x, fp) * vec4<f32>(1.0, 1.0, 0.0, 0.0);
  let d = ocean_deriv(ocean_x0_w(x, w), w);
  let j = (1.0 + d.b.z) * (1.0 + d.b.w) - d.dxz * d.dxz;
  let thr = U.ocx[7].w;
  let sig = U.ocx[4].z;
  return select(0.0, clamp((thr + 0.3 * sig - j) / (0.9 * sig), 0.0, 1.0), U.ocn2.y > 0.0);
}

// The sea's slope that the simulation cannot carry (all of it, less the waves the simulation's grid
// holds), laid onto the simulated surface where it is the sea's own surface.
fn ocean_chop_slope(x: vec2<f32>, fp: f32) -> vec2<f32> {
  if (!ocean_on()) { return vec2<f32>(0.0); }
  let w = ocean_gust_w(x, fp);
  let x0 = ocean_x0_w(x, w);
  let nf = ocean_n_of(ocean_deriv(x0, w));
  let nl = ocean_n_of(ocean_deriv_low(x0, w));
  return -(nf.xz / nf.y - nl.xz / nl.y);
}

// The sea's foam at the undisplaced point x0: fresh whitecap foam, aged foam, bubbles under the
// surface (from the foam's drifting frame on the largest cascade's tile).
fn ocean_foam(x0: vec2<f32>, fp: f32) -> vec3<f32> {
  if (!ocean_on()) { return vec3<f32>(0.0); }
  return textureSampleLevel(oc_f, rep, vec3<f32>(x0 * U.ocx[0].x - U.ocx[3].xy, 0.5), 0.0).xyz;
}
