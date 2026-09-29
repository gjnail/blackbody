// Burnable colliders. Every burnable collider owns a region of the object-burn atlas laid out in
// its own frame (a grid of the simulation's cell size around it), so the burn travels with the
// object when it moves or turns. `init` marks the cells just outside its surface and lays out
// patchy fuel; `main` steps them (see burn_common.wgsl), sampling the gas where the cell is now.
//!include common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl
//!include colliders.wgsl
//!include burn_common.wgsl

struct Params {
  g: Grid,
  sp: vec4<f32>,    // catch temperature, 1 / catch time (1/s), creep speed (m/s), 1 / burn time (1/s)
  sp2: vec4<f32>,   // 1 / smoulder time (1/s), water on (1/0), how fast water puts a surface out (1/s), atlas depth (cells)
  pat: vec4<f32>,   // coverage (0..1), patch frequency (1/m), seed, _
  cnt: vec4<f32>,   // emitter count
  em: array<Emitter, MAX_EMITTERS>,
  ccnt: vec4<f32>,  // collider count
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var atlas: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var water: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> slots: array<BurnSlot>;
@group(0) @binding(6) var lin: sampler;
@group(1) @binding(0) var<uniform> U: Params;

// Which collider owns atlas cell c, and c's position in that collider's frame (m); ok = false if none.
struct Owner { k: i32, q: vec3<f32>, ok: bool };

fn owner_of(c: vec3<i32>) -> Owner {
  var o: Owner;
  o.ok = false;
  for (var i = 0; i < i32(U.ccnt.x); i++) {
    let slot = i32(U.col[i].m2.w);
    if (slot < 0) { continue; }
    let s = slots[slot];
    let z0 = i32(s.lo.w);
    if (c.z >= z0 && c.z < z0 + i32(s.dims.z) && c.x < i32(s.dims.x) && c.y < i32(s.dims.y)) {
      o.k = i;
      o.q = s.lo.xyz + (vec3<f32>(f32(c.x), f32(c.y), f32(c.z - z0)) + 0.5) * s.dims.w;
      o.ok = true;
      return o;
    }
  }
  return o;
}

fn world_at(k: Collider, q: vec3<f32>) -> vec3<f32> { return k.a.xyz + yaw_to_world(q, k.b.w); }

@compute @workgroup_size(8, 8, 4)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let dims = vec3<i32>(textureDimensions(dst));
  if (any(c >= dims)) { return; }
  let o = owner_of(c);
  if (!o.ok) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  let k = U.col[o.k];
  let cell = slots[i32(k.m2.w)].dims.w;
  let d = col_sdf(k, world_at(k, o.q)) / cell;
  if (d < 0.0 || d >= 1.5) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  // patchy fuel, fixed to the object
  let p = o.q * U.pat.y + vec3<f32>(U.pat.z * 3.17, f32(o.k) * 11.3, U.pat.z * 1.93);
  let v = clamp(0.5 + fbm3(p, 3) * 2.2, 0.0, 1.0);
  var fuel = 1.0 - smoothstep(U.pat.x - 0.06, U.pat.x + 0.06, v);
  if (U.pat.x >= 0.999) { fuel = 1.0; }
  fuel *= 0.75 + 0.5 * clamp(0.5 + fbm3(p * 3.1 + vec3<f32>(7.3), 2), 0.0, 1.0);
  textureStore(dst, c, vec4<f32>(fuel, 0.0, 1.0, 0.0));
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let dims = vec3<i32>(textureDimensions(dst));
  if (any(c >= dims)) { return; }
  let b = textureLoad(src, c, 0);
  if (b.z < 0.5) {
    textureStore(dst, c, b);
    return;
  }
  let o = owner_of(c);
  if (!o.ok) {
    textureStore(dst, c, b);
    return;
  }
  let k = U.col[o.k];
  let s = slots[i32(k.m2.w)];
  let wp = world_at(k, o.q);
  let h = U.g.n.w;
  let n = U.g.n.xyz;
  let pg = (wp - U.g.org.xyz) / h;
  var T = 0.0;
  if (all(pg >= vec3<f32>(0.0)) && all(pg <= n)) { T = samp_c(scal, lin, pg, n).x; }
  var wet = 0.0;
  for (var i = 0; i < i32(U.cnt.x); i++) {
    let e = U.em[i];
    if (e.k.w > 0.0) { wet += e.k.w * emitter_mask(e, wp, h); }
  }
  if (U.sp2.y > 0.5 && all(pg >= vec3<f32>(0.0)) && all(pg < n)) {
    wet += U.sp2.z * clamp(textureLoad(water, vec3<i32>(pg), 0).x, 0.0, 1.0);
  }
  var nb = 0.0;
  if (b.y < 1.0 && b.x > 0.0) {
    let z0 = i32(s.lo.w);
    let lo = vec3<i32>(0, 0, z0);
    let hi = vec3<i32>(i32(s.dims.x), i32(s.dims.y), z0 + i32(s.dims.z));
    for (var j = 0; j < 27; j++) {
      let off = vec3<i32>(j % 3, (j / 3) % 3, j / 9) - vec3<i32>(1);
      if (j == 13) { continue; }
      let q = c + off;
      if (any(q < lo) || any(q >= hi)) { continue; }
      if (burn_alight(textureLoad(src, q, 0))) { nb = max(nb, 1.0 / length(vec3<f32>(off))); }
    }
  }
  textureStore(dst, c, burn_step(b, T, nb, wet, U.g.bc.w, s.dims.w, U.sp, U.sp2.x));
}
