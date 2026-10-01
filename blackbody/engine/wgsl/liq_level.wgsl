// The open water's surface per column of cells (ocn_bc.wgsl): x = height (cells above the
// grid floor; the flat water level, or the sea's waves), yzw = the sea's orbital velocity at the
// surface (m/s). The including file declares ocn_t (texture_2d<f32>).

fn sea_at(xz: vec2<f32>) -> vec4<f32> {
  let d = vec2<i32>(textureDimensions(ocn_t));
  return textureLoad(ocn_t, clamp(vec2<i32>(floor(xz)), vec2<i32>(0), d - vec2<i32>(1)), 0);
}

fn level_at(xz: vec2<f32>) -> f32 { return sea_at(xz).x; }

// The sea's velocity at height y (cells) under the surface of that column: the orbital motion
// fades with depth as exp(k (y - surface)).
fn sea_velocity(p: vec3<f32>, kp: f32, h: f32) -> vec3<f32> {
  let s = sea_at(p.xz);
  return s.yzw * exp(min(p.y - s.x, 0.0) * h * kp);
}
