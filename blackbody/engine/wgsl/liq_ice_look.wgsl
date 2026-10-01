// Ice, in the liquid renderer's march (included by liq_march.wgsl; needs its U, heat_t, lin and mir).
//
// Kept to a texture lookup: the march is a very large shader and every loop added to it costs minutes of
// driver compile. Where the surface is frozen (heat_t.w, the particles' frozen share on the surface
// grid) the march leaves out ripples, rain rings and foam, so ice keeps the shape it froze in; its
// frost, rime and crystal facets are shaded afterwards in a pass of their own (liq_ice_shade.wgsl), and
// how cloudy it is inside is in the dye field (liq_ice_dye.wgsl), which refracted rays already see.
//
// U.ice:  x = ice on (1/0), frost amount, _, melting (0..1)
// U.ice2: frost colour (rgb), crystal size (m)

fn ice_at(p: vec3<f32>) -> f32 {
  if (U.ice.x < 0.5) { return 0.0; }
  return clamp(textureSampleLevel(heat_t, lin, mir(p) / U.n.xyz, 0.0).w, 0.0, 1.0);
}
