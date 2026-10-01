// Matter's surface, for drawing: its distance and the rest of its look in one filterable texture for the stage
// (stage.wgsl m_phi): distance (m), clear, sparkle, wrap. Near the matter the distance is the particles' own (Zhu and
// Bridson); further off, the distance to the nearest seed of the jump flood (mpm_jfa.wgsl) plus that seed's.

struct Params {
  n: vec4<f32>,      // grid nodes (x, y, z), node spacing (m)
  b: vec4<f32>,      // the band's width (m), _
};

@group(0) @binding(0) var phi: texture_3d<f32>;
@group(0) @binding(1) var look1: texture_3d<f32>;
@group(0) @binding(2) var seeds: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(U.n.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var d = textureLoad(phi, c, 0).x;
  if (d >= 0.999 * U.b.x) {
    // (less a band's width: the seed nearest is not quite the surface nearest, and rays must never step past it)
    let s = textureLoad(seeds, c, 0);
    d = select(1.0e4, max(length(vec3<f32>(c) - s.xyz) * U.n.w + s.w - U.b.x, U.b.x), s.w >= 0.0);
  }
  textureStore(dst, c, vec4<f32>(min(d, 6.0e4), textureLoad(look1, c, 0).xyz));
}
