// Hot matter's light on what is round it (stage.py): the matter's grid cut into blocks, and every block whose surface
// glows (mpm_surf_norm.wgsl's temperature, as the stage draws it: a blackbody, stage.wgsl matter_glow) a point light at
// its glow-weighted centre with the intensity of its glowing surface (radiance times area). ML[0].x: how many
// (finish); then two vec4 a light: (position, fire-local m; half the block), (intensity rgb, _).

struct Params {
  n: vec4<f32>,      // the matter's nodes (x, y, z), block (nodes)
  o: vec4<f32>,      // node 0 (fire-local m), node spacing (m)
  mg: vec4<f32>,     // the glow: on, its radiance at 1300 K, the table's first temperature (K), 1 / its step
  k: vec4<f32>,      // most lights, _, _, _
  mgb: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var surf: texture_3d<f32>;   // x: the distance to the surface (m)
@group(0) @binding(1) var look2: texture_3d<f32>;  // x: the temperature there (K)
@group(0) @binding(2) var<storage, read_write> ML: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> MLC: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

fn glow(T: f32) -> vec3<f32> {
  let f = (T - U.mg.z) * U.mg.w;
  if (f <= 0.0) { return vec3<f32>(0.0); }
  let fc = min(f, 14.999);
  let i = u32(fc);
  let t = fc - f32(i);
  let a = U.mgb[i];
  let b = U.mgb[i + 1u];
  var lg = mix(a.w, b.w, t);
  if (f > 15.0) { lg += (f - 15.0) * (b.w - a.w); }
  return mix(a.rgb, b.rgb, t) * pow(10.0, lg) * U.mg.y;
}

@compute @workgroup_size(4, 4, 4)
fn lights(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = i32(U.n.w);
  let dims = vec3<i32>(U.n.xyz);
  let lo = vec3<i32>(id) * b;
  if (any(lo >= dims)) { return; }
  let hi = min(lo + vec3<i32>(b), dims);
  let dx = U.o.w;
  var power = vec3<f32>(0.0);
  var centre = vec3<f32>(0.0);
  var weight = 0.0;
  for (var z = lo.z; z < hi.z; z++) {
    for (var y = lo.y; y < hi.y; y++) {
      for (var x = lo.x; x < hi.x; x++) {
        let c = vec3<i32>(x, y, z);
        // (nodes on its surface, each about a node spacing square of it, facing every way: half of it toward a point)
        if (abs(textureLoad(surf, c, 0).x) > 0.75 * dx) { continue; }
        let e = glow(textureLoad(look2, c, 0).x);
        let l = e.r + e.g + e.b;
        if (l <= 1.0e-6) { continue; }
        power += e;
        centre += vec3<f32>(c) * l;
        weight += l;
      }
    }
  }
  if (weight <= 1.0e-6) { return; }
  let i = atomicAdd(&MLC[0], 1u);
  if (i >= u32(U.k.x)) { return; }
  ML[1u + 2u * i] = vec4<f32>(U.o.xyz + centre / weight * dx, 0.5 * f32(b) * dx);
  ML[2u + 2u * i] = vec4<f32>(power * 0.5 * dx * dx, 0.0);
}

@compute @workgroup_size(1, 1, 1)
fn finish() {
  ML[0] = vec4<f32>(f32(min(atomicLoad(&MLC[0]), u32(U.k.x))), 0.0, 0.0, 0.0);
}
