// Footage decoding shared by the compositor and the liquid renderer (which refracts the footage).

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let x = max(c, vec3<f32>(0.0));
  let lo = x * 12.92;
  let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
  return select(hi, lo, x <= vec3<f32>(0.0031308));
}

// Footage code values -> scene-linear Rec.709 (0 sRGB, 1 Rec.709 / BT.1886, 2 linear, 3 ACEScg).
fn input_transform(c: vec3<f32>, kind: i32) -> vec3<f32> {
  if (kind == 0) { return srgb_to_linear(c); }
  if (kind == 1) { return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.4)); }
  if (kind == 3) {
    // ACEScg (AP1) -> linear Rec.709
    let m = mat3x3<f32>(1.70505, -0.13026, -0.02400, -0.62179, 1.14080, -0.12897, -0.08326, -0.01055, 1.15297);
    return m * c;
  }
  return c;
}
