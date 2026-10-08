// The liquid's surface at given points (render/passes.py liquid_front: where the compositing passes see it), so the
// surface grid need not be read back whole: at each point its velocity (m/s) and the gradient of its distance (the
// normal's direction), fire-local axes, both filtered as the liquid's march samples them.

struct Params {
  n: vec4<f32>,   // the surface grid's dims (cells), points
};

@group(0) @binding(0) var surf_t: texture_3d<f32>;                   // distance (cells), velocity (LiquidRenderer.surf)
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var<storage, read> pts: array<vec4<f32>>;      // grid place (cells from its corner)
@group(0) @binding(3) var<storage, read_write> outp: array<vec4<f32>>;   // per point: velocity, then the gradient
@group(1) @binding(0) var<uniform> U: Params;

fn at(p: vec3<f32>) -> vec4<f32> {
  return textureSampleLevel(surf_t, lin, p / U.n.xyz, 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 64u;
  if (f32(i) >= U.n.w) { return; }
  let p = pts[i].xyz;
  let e = 0.75;   // (cells: the central differences' half step)
  let g = vec3<f32>(at(p + vec3<f32>(e, 0.0, 0.0)).x - at(p - vec3<f32>(e, 0.0, 0.0)).x,
                    at(p + vec3<f32>(0.0, e, 0.0)).x - at(p - vec3<f32>(0.0, e, 0.0)).x,
                    at(p + vec3<f32>(0.0, 0.0, e)).x - at(p - vec3<f32>(0.0, 0.0, e)).x);
  outp[2u * i] = vec4<f32>(at(p).yzw, 0.0);
  outp[2u * i + 1u] = vec4<f32>(g, 0.0);
}
