// Integer hashing and gradient noise with analytic derivatives (quintic interpolation).

fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
  var v = v_in * 1664525u + vec3<u32>(1013904223u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v = v ^ (v >> vec3<u32>(16u));
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}

fn pcg1(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// Uniform in [0, 1) from an integer seed.
fn rand1(seed: u32) -> f32 { return f32(pcg1(seed)) * (1.0 / 4294967296.0); }

// Three uniforms in [0, 1) from an integer seed.
fn rand3(seed: u32, salt: u32) -> vec3<f32> {
  return vec3<f32>(pcg3d(vec3<u32>(seed, salt, seed ^ 0x9e3779b9u))) * (1.0 / 4294967296.0);
}

// Gradient in [-1, 1]^3 for an integer lattice point.
fn grad3(p: vec3<f32>) -> vec3<f32> {
  let h = pcg3d(vec3<u32>(vec3<i32>(p) + vec3<i32>(1048576)));
  return vec3<f32>(h) * (2.0 / 4294967295.0) - vec3<f32>(1.0);
}

// Gradient noise value (about -0.8..0.8).
fn gnoise(x: vec3<f32>) -> f32 {
  let i = floor(x);
  let f = fract(x);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let va = dot(grad3(i), f);
  let vb = dot(grad3(i + vec3<f32>(1.0, 0.0, 0.0)), f - vec3<f32>(1.0, 0.0, 0.0));
  let vc = dot(grad3(i + vec3<f32>(0.0, 1.0, 0.0)), f - vec3<f32>(0.0, 1.0, 0.0));
  let vd = dot(grad3(i + vec3<f32>(1.0, 1.0, 0.0)), f - vec3<f32>(1.0, 1.0, 0.0));
  let ve = dot(grad3(i + vec3<f32>(0.0, 0.0, 1.0)), f - vec3<f32>(0.0, 0.0, 1.0));
  let vf = dot(grad3(i + vec3<f32>(1.0, 0.0, 1.0)), f - vec3<f32>(1.0, 0.0, 1.0));
  let vg = dot(grad3(i + vec3<f32>(0.0, 1.0, 1.0)), f - vec3<f32>(0.0, 1.0, 1.0));
  let vh = dot(grad3(i + vec3<f32>(1.0, 1.0, 1.0)), f - vec3<f32>(1.0, 1.0, 1.0));
  return va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
       + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
       + u.z * u.x * (va - vb - ve + vf) + (-va + vb + vc - vd + ve - vf - vg + vh) * u.x * u.y * u.z;
}

// Gradient noise with its derivatives: .x = value, .yzw = d/dx, d/dy, d/dz.
fn gnoise_d(x: vec3<f32>) -> vec4<f32> {
  let i = floor(x);
  let w = fract(x);
  let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
  let du = 30.0 * w * w * (w * (w - 2.0) + 1.0);
  let ga = grad3(i);
  let gb = grad3(i + vec3<f32>(1.0, 0.0, 0.0));
  let gc = grad3(i + vec3<f32>(0.0, 1.0, 0.0));
  let gd = grad3(i + vec3<f32>(1.0, 1.0, 0.0));
  let ge = grad3(i + vec3<f32>(0.0, 0.0, 1.0));
  let gf = grad3(i + vec3<f32>(1.0, 0.0, 1.0));
  let gg = grad3(i + vec3<f32>(0.0, 1.0, 1.0));
  let gh = grad3(i + vec3<f32>(1.0, 1.0, 1.0));
  let va = dot(ga, w);
  let vb = dot(gb, w - vec3<f32>(1.0, 0.0, 0.0));
  let vc = dot(gc, w - vec3<f32>(0.0, 1.0, 0.0));
  let vd = dot(gd, w - vec3<f32>(1.0, 1.0, 0.0));
  let ve = dot(ge, w - vec3<f32>(0.0, 0.0, 1.0));
  let vf = dot(gf, w - vec3<f32>(1.0, 0.0, 1.0));
  let vg = dot(gg, w - vec3<f32>(0.0, 1.0, 1.0));
  let vh = dot(gh, w - vec3<f32>(1.0, 1.0, 1.0));
  let k = -va + vb + vc - vd + ve - vf - vg + vh;
  let value = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
            + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
            + u.z * u.x * (va - vb - ve + vf) + k * u.x * u.y * u.z;
  let deriv = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
            + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg)
            + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z
            + du * (vec3<f32>(vb, vc, ve) - vec3<f32>(va)
                    + u.yzx * vec3<f32>(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
                    + u.zxy * vec3<f32>(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
                    + u.yzx * u.zxy * k);
  return vec4<f32>(value, deriv);
}

fn fbm3(p: vec3<f32>, octaves: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < octaves; i++) {
    s += a * gnoise(q);
    q = q * 2.03 + vec3<f32>(1.7, -2.3, 0.9);
    a *= 0.5;
  }
  return s;
}

// Divergence-free noise: the curl of three decorrelated noise potentials.
fn curl_noise(p: vec3<f32>) -> vec3<f32> {
  let a = gnoise_d(p);
  let b = gnoise_d(p + vec3<f32>(31.416, -47.853, 12.793));
  let c = gnoise_d(p + vec3<f32>(-63.215, 18.467, 91.123));
  return vec3<f32>(c.z - b.w, a.w - c.y, b.y - a.z);
}
