// Vorticity at cell centres: .xyz = curl of velocity (1/s), .w = its magnitude.
//!include common.wgsl

struct Params { g: Grid };

@group(0) @binding(0) var vel: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn vc(c: vec3<i32>, d: vec3<i32>) -> vec3<f32> {
  return vel_centre(vel, clamp(c, vec3<i32>(0), d - vec3<i32>(1)));
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  let ih = 0.5 / U.g.n.w;
  let l = vc(c - vec3<i32>(1, 0, 0), d);
  let r = vc(c + vec3<i32>(1, 0, 0), d);
  let b = vc(c - vec3<i32>(0, 1, 0), d);
  let t = vc(c + vec3<i32>(0, 1, 0), d);
  let k = vc(c - vec3<i32>(0, 0, 1), d);
  let f = vc(c + vec3<i32>(0, 0, 1), d);
  let w = ih * vec3<f32>((t.z - b.z) - (f.y - k.y), (f.x - k.x) - (r.z - l.z), (r.y - l.y) - (t.x - b.x));
  textureStore(dst, c, vec4<f32>(w, length(w)));
}
