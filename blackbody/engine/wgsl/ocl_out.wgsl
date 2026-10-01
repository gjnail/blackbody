// The open-water layer as the renderer reads it: (wave height (m), foam, water depth (m), current speed (m/s)).

struct Params { L: Layer, };

@group(0) @binding(0) var wave: texture_3d<f32>;
@group(0) @binding(1) var foam: texture_2d<f32>;
@group(0) @binding(2) var dep: texture_2d<f32>;
@group(0) @binding(3) var vel: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;
//!include ocl_common.wgsl

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec2<i32>(id.xy);
  if (c.x >= lay_n() || c.y >= lay_n()) { return; }
  let h = textureLoad(wave, vec3<i32>(c, 0), 0).x;
  let d = textureLoad(dep, c, 0);
  let depth = select(d.x, -1.0, d.y <= 0.0);   // negative: solid at the water line
  textureStore(dst, c, vec4<f32>(clamp(h, -1.0e3, 1.0e3), textureLoad(foam, c, 0).x, min(depth, 6.0e4),
                                  length(textureLoad(vel, c, 0).xy)));
}
