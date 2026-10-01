// Light a molten liquid casts on its surroundings, pass 3: the lights (liq_lava_lights.wgsl) summed on
// a map of the ground around the box, so the march reads the glow's light at a point with one lookup
// instead of a loop over the lights (inlined wherever the march lights a surface, that loop made its
// compile far longer). Per texel, at ground level: x = (light from all sides, rgb; how much of it
// comes from one direction), y = (that direction, unit, fire-local).
//
// map[2 * (i + nx * k)], map[2 * (i + nx * k) + 1] for texel (i, k).

struct Params {
  m: vec4<f32>,     // map corner x, z (fire-local m), texel size x, z (m)
  d: vec4<f32>,     // texels along x, z
};

@group(0) @binding(0) var<storage, read> lights: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> map: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nx = u32(U.d.x);
  let nz = u32(U.d.y);
  if (id.x >= nx || id.y >= nz) { return; }
  let x = vec3<f32>(U.m.x + (f32(id.x) + 0.5) * U.m.z, 0.0, U.m.y + (f32(id.y) + 0.5) * U.m.w);
  let count = min(u32(lights[0].x), (arrayLength(&lights) - 1u) / 2u);
  var e = vec3<f32>(0.0);
  var dir = vec3<f32>(0.0);
  for (var i = 0u; i < count; i++) {
    let a = lights[1u + 2u * i];
    let d = a.xyz - x;
    let r2 = dot(d, d);
    let l = lights[2u + 2u * i].rgb / (r2 + a.w * a.w);
    e += l;
    dir += d * inverseSqrt(max(r2, 1e-8)) * (l.r + l.g + l.b);
  }
  let s = e.r + e.g + e.b;
  let len = length(dir);
  let k = (id.x + nx * id.y) * 2u;
  map[k] = vec4<f32>(e, select(0.0, len / s, s > 1e-12));
  map[k + 1u] = vec4<f32>(select(vec3<f32>(0.0, 1.0, 0.0), dir / len, len > 1e-12), 0.0);
}
