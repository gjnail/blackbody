// Heat, pass 2: each cell's mean enthalpy, temperature and frozen share from the particle sums
// (liq_therm_p2g.wgsl), and how much liquid it holds. The deep liquid of a narrow band (full, with no
// particles) keeps what it had.
//
// cell: x = mean enthalpy (kJ/kg), y = temperature (C), z = frozen share (0..1), w = liquid (x rest density)
//!include common.wgsl
//!include liq_common.wgsl
//!include liq_therm_common.wgsl

struct Params {
  g: Grid,
  th: Therm,
};

@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var dens: texture_3d<f32>;
@group(0) @binding(2) var prev: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let k = nidx(c, n) * 3u;
  let w = f32(acc[k]) / FX_F;
  let fl = textureLoad(dens, c, 0).x / max(U.th.c.x, 1e-3);
  var out = vec4<f32>(0.0, U.th.b.x, 0.0, 0.0);
  if (w > 1e-4) {
    let H = f32(acc[k + 1u]) / FX_H / w;
    let F = clamp(f32(acc[k + 2u]) / FX_F / w, 0.0, 1.0);
    out = vec4<f32>(H, cell_temp(U.th, H, F), F, fl);
  } else if (fl > 0.5) {
    let q = textureLoad(prev, c, 0);
    if (q.w > 0.05) {
      out = vec4<f32>(q.xyz, fl);
    } else {
      let H = h_of_temp(U.th, U.th.b.w, false);
      out = vec4<f32>(H, U.th.b.w, 0.0, fl);
    }
  }
  textureStore(dst, c, out);
}
