// Fabric under the fire: the march stopped at the cloth, so what it gathered is in front of it. The
// cloth goes behind that (hidden where a collider in the shot or the footage's holdouts are in front
// of it), into the beauty, the emission (its glowing edges), depth and alpha. The fabric layer on its
// own is gathered over the anti-aliasing passes.
struct Params {
  p: vec4<f32>,   // width, height, pass, 1 / passes
  q: vec4<f32>,   // how far in front of a collider's surface its sphere-traced depth may be (m), _, _, _
};

@group(0) @binding(0) var beauty: texture_2d<f32>;
@group(0) @binding(1) var emit: texture_2d<f32>;
@group(0) @binding(2) var aux: texture_2d<f32>;
@group(0) @binding(3) var ccol: texture_2d<f32>;
@group(0) @binding(4) var caux: texture_2d<f32>;
@group(0) @binding(5) var cglow: texture_2d<f32>;
@group(0) @binding(6) var mask: texture_2d<f32>;
@group(0) @binding(7) var fab_in: texture_2d<f32>;
@group(0) @binding(8) var out_b: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var out_e: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var out_x: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var out_f: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.p.x || f32(px.y) >= U.p.y) { return; }
  let b = textureLoad(beauty, px, 0);
  let e = textureLoad(emit, px, 0);
  let x = textureLoad(aux, px, 0);
  let ca = textureLoad(caux, px, 0);
  let m = textureLoad(mask, px, 0);
  var vis = ca.w;
  if (vis > 0.0 && m.z > 1e-3) {
    // hidden behind an object in the shot (or the footage's matte, which has no depth: always in front)
    let hold = m.y / m.z;
    if (hold <= 0.0 || ca.x > hold + U.q.x) { vis *= 1.0 - clamp(m.z, 0.0, 1.0); }
  }
  let fa = clamp(b.a, 0.0, 1.0);
  let k = (1.0 - fa) * vis;
  let cc = textureLoad(ccol, px, 0).rgb * vis;
  let cg = textureLoad(cglow, px, 0).rgb * vis;
  let a = fa + k;
  var depth = x.y;
  if (k > 0.0) { depth = (x.y * fa + ca.y * k) / max(a, 1e-6); }
  textureStore(out_b, px, vec4<f32>(b.rgb + cc * (1.0 - fa), a));
  textureStore(out_e, px, vec4<f32>(e.rgb + cg * (1.0 - fa), e.a));
  textureStore(out_x, px, vec4<f32>(x.x, depth, x.z, max(x.w, a)));
  var f = vec4<f32>(cc, vis) * U.p.w;
  if (U.p.z > 0.5) { f += textureLoad(fab_in, px, 0); }
  textureStore(out_f, px, f);
}
