// Lume's light on a surface in the fire's volume (lume_volume.wgsl): the light falling on it from every way but the key
// light's and the set's lamps' own beams, from the light grid's mean and the way its light comes from (an L1 spherical
// harmonic): the fire's light, shadowed by the smoke and the set, the sky, dimmed where the smoke hides it, and the
// light the ground, the objects and the smoke send on. The including file declares `lin` (a linear sampler).

// The irradiance on a surface at simulation-grid point g (cells), facing n, from Lume's grids lv (the mean radiance)
// and lvd (the first moment over it); sim: the simulation grid's dims. Taken half a cell of the grid in front of it.
fn lume_irradiance(lv: texture_3d<f32>, lvd: texture_3d<f32>, g: vec3<f32>, sim: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let d = vec3<f32>(textureDimensions(lv, 0));
  let q = g * (d / sim) + n * 0.5;
  let m = samp_c(lv, lin, q, d).rgb;
  let w = samp_c(lvd, lin, q, d).xyz;
  return 3.14159265 * m * max(1.0 + 2.0 * dot(w, n), 0.0);
}
