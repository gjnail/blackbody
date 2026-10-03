// Liquid <- fabric: the momentum the water's drag gave the cloth (cloth_predict.wgsl adds it up per vertex, LIMP), the
// water loses: both feel the same force, opposite ways. So a sheet dragged through a tank pushes a wave ahead of it and
// leaves a wake, a net held across a stream slows the flow through it, and a flag flapping in the surf stirs the water.
//  - splat: each vertex's momentum onto the liquid's cells round it (trilinear, fixed point)
//  - apply: each face of the liquid's grid takes the momentum of the cells either side of it, over the water there
//!include common.wgsl
//!include liq_common.wgsl

const MOM_K: f32 = 1.0e9;      // momentum fixed point (kg m/s)

struct Params {
  g: Grid,           // the liquid's grid
  k: vec4<f32>,      // vertices, the liquid's density (kg/m^3), its particles a full cell holds, _
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> LIMP: array<vec4<f32>>;     // per vertex: momentum it took from the water
@group(0) @binding(2) var<storage, read_write> LM: array<atomic<i32>>;   // per cell: momentum to give the water (3)
@group(0) @binding(3) var vin: texture_3d<f32>;
@group(0) @binding(4) var dens: texture_3d<f32>;
@group(0) @binding(5) var vout: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= u32(U.k.x)) { return; }
  let p = LIMP[i].xyz;
  if (dot(p, p) <= 0.0) { return; }
  let n = gdim(U.g);
  let q = (X[i].xyz - U.g.org.xyz) / U.g.n.w - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, k >> 2);
    let c = b + o;
    if (!in_grid(c, n)) { continue; }
    let w3 = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    let m = clamp(-p * (w3.x * w3.y * w3.z) * MOM_K, vec3<f32>(-2.0e9), vec3<f32>(2.0e9));
    let j = 3u * nidx(c, n);
    atomicAdd(&LM[j], i32(round(m.x)));
    atomicAdd(&LM[j + 1u], i32(round(m.y)));
    atomicAdd(&LM[j + 2u], i32(round(m.z)));
  }
}

fn mom(c: vec3<i32>, n: vec3<i32>, k: u32) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return f32(atomicLoad(&LM[3u * nidx(c, n) + k])) / MOM_K;
}

fn water(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (!in_grid(c, n)) { return 0.0; }
  return clamp(textureLoad(dens, c, 0).x / max(U.k.z, 1.0e-6), 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 4)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c > n)) { return; }
  var v = textureLoad(vin, c, 0);
  let h = U.g.n.w;
  let cell = U.k.y * h * h * h;          // kg of water in a full cell
  for (var k = 0u; k < 3u; k++) {
    var e = vec3<i32>(0);
    e[k] = 1;
    let a = c - e;                       // (the face between cell a and cell c)
    let wa = water(a, n);
    let wc = water(c, n);
    let w = wa + wc;
    if (w < 0.2) { continue; }
    let pm = mom(a, n, k) + mom(c, n, k);
    v[k] += pm / (0.5 * w * cell + 1.0e-12) * 0.5;
  }
  textureStore(vout, c, v);
}
