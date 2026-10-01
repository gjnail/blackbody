// The sea's spectrum at time t, ready for the inverse FFT (ocn_fft.wgsl). Each texel of h0_t holds
// one wave vector's h0(k) (xy) and conj(h0(-k)) (zw), on the lattice of its layer's tile (index i
// is wavenumber i below n/2, i - n from there). A wave travels along its wave vector; its partner
// travels the other way, so the field is real. The current carries the whole pattern, and every
// texel of the result holds the surface at the centre of its texel.
//
// Two real fields go into each complex FFT (a + i b), two complex numbers per texel:
//   main: per layer l, texel layer 2l   = (h + i dx, dz + i ddx/dz)
//                      texel layer 2l+1 = (dh/dx + i dh/dz, ddx/dx + i ddz/dz)
//         (dx, dz: the choppy crests' horizontal displacement)
//   jac:  per layer l, (ddx/dx + i ddz/dz, ddx/dz + i h): what the foam needs (the fold of the surface)
//   sim:  per layer l, (h + i ux, uz + i uy): the height and the orbital velocity at the surface

struct Params {
  k: vec4<f32>,                // n, time (s), depth (m), choppiness
  c: vec4<f32>,                // current x, z (m/s), surface tension / density, _
  tile: array<vec4<f32>, 2>,   // 1 / tile size (1/m) of each layer
};

@group(0) @binding(0) var h0_t: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// i * a
fn ci(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(-a.y, a.x); }

struct Wave {
  kv: vec2<f32>,   // wave vector (1/m)
  k: f32,
  w: f32,          // angular frequency (rad/s)
  ht: vec2<f32>,   // height amplitude at t
  hd: vec2<f32>,   // the same with the backward partner negated (for velocities)
};

fn wave_at(id: vec3<u32>, l: u32) -> Wave {
  let n = u32(U.k.x);
  let ix = select(i32(id.x), i32(id.x) - i32(n), id.x >= n / 2u);
  let iz = select(i32(id.y), i32(id.y) - i32(n), id.y >= n / 2u);
  let inv_l = U.tile[l / 4u][l % 4u];
  var o: Wave;
  o.kv = vec2<f32>(f32(ix), f32(iz)) * (6.28318530718 * inv_l);
  o.k = length(o.kv);
  o.w = sqrt((9.81 * o.k + U.c.z * o.k * o.k * o.k) * tanh(min(o.k * U.k.z, 20.0)));
  let h = textureLoad(h0_t, vec3<i32>(vec2<i32>(id.xy), i32(l)), 0);
  let ph = o.w * U.k.y;
  let e = vec2<f32>(cos(ph), sin(ph));
  let a = cmul(h.xy, vec2<f32>(e.x, -e.y));
  let b = cmul(h.zw, e);
  // carried by the current, and shifted half a texel so each texel holds its centre
  let half = 0.5 / (f32(n) * inv_l);
  let s = -dot(o.kv, U.c.xy) * U.k.y + (o.kv.x + o.kv.y) * half;
  let sh = vec2<f32>(cos(s), sin(s));
  o.ht = cmul(a + b, sh);
  o.hd = cmul(a - b, sh);
  return o;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = u32(U.k.x);
  if (id.x >= n || id.y >= n) { return; }
  let o = wave_at(id, id.z);
  let kh = o.kv / max(o.k, 1e-6);
  let chop = U.k.w;
  let ht = o.ht;
  // spectra of the eight real fields
  let dx = ci(ht) * (chop * kh.x);
  let dz = ci(ht) * (chop * kh.y);
  let dxz = ht * (-chop * kh.x * o.kv.y);
  let hx = ci(ht) * o.kv.x;
  let hz = ci(ht) * o.kv.y;
  let dxx = ht * (-chop * kh.x * o.kv.x);
  let dzz = ht * (-chop * kh.y * o.kv.y);
  let p = vec2<i32>(id.xy);
  let l = i32(id.z);
  textureStore(dst, vec3<i32>(p, 2 * l), vec4<f32>(ht + ci(dx), dz + ci(dxz)));
  textureStore(dst, vec3<i32>(p, 2 * l + 1), vec4<f32>(hx + ci(hz), dxx + ci(dzz)));
}

@compute @workgroup_size(8, 8, 1)
fn jac(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = u32(U.k.x);
  if (id.x >= n || id.y >= n) { return; }
  let o = wave_at(id, id.z);
  let kh = o.kv / max(o.k, 1e-6);
  let chop = U.k.w;
  let dxx = o.ht * (-chop * kh.x * o.kv.x);
  let dzz = o.ht * (-chop * kh.y * o.kv.y);
  let dxz = o.ht * (-chop * kh.x * o.kv.y);
  textureStore(dst, vec3<i32>(vec2<i32>(id.xy), i32(id.z)), vec4<f32>(dxx + ci(dzz), dxz + ci(o.ht)));
}

@compute @workgroup_size(8, 8, 1)
fn sim(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = u32(U.k.x);
  if (id.x >= n || id.y >= n) { return; }
  let o = wave_at(id, id.z);
  let kh = o.kv / max(o.k, 1e-6);
  // linear waves: the horizontal velocity at the surface is in phase with the height along the
  // wave's travel (stronger in shallow water), the vertical one a quarter period ahead
  let deep = 1.0 / max(tanh(min(o.k * U.k.z, 20.0)), 0.05);
  let uh = o.hd * (o.w * deep);
  let ux = uh * kh.x;
  let uz = uh * kh.y;
  let uy = ci(o.hd) * (-o.w);
  textureStore(dst, vec3<i32>(vec2<i32>(id.xy), i32(id.z)), vec4<f32>(o.ht + ci(ux), uz + ci(uy)));
}
