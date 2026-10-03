// The lights in the set: how much of a lamp's intensity goes each way. The lamp buffer (renderer.pack_lamps) holds
// row k the lamp (Lamp: p position, radius; c power, kind; d aim, a spot's outer cosine or an area light's half
// width; e a spot's inner cosine or an area light's half height, smoke shadows, in the footage, has a profile), row
// LAMP_MAX + k its own axes across the aim (p.xyz the width axis, c.xyz the height axis; p.w, c.w an area light's half
// width and height), and from row 2 LAMP_MAX + k LAMP_PROF_V its light profile's table (an IES file: io/ies.py),
// LAMP_PROF_V rows of LAMP_PROF_H values: its intensity over its peak, from along the aim (row 0) to straight back, round
// the aim from the width axis. Included by the shaders that bind `lamps` (array<Lamp>).

const LAMP_MAX: i32 = 8;
const LAMP_PROF_V: i32 = 64;
const LAMP_PROF_H: i32 = 16;

fn lamp_cell(row: i32, j: i32) -> f32 {
  let r = lamps[row];
  var v = r.p;
  let q = j >> 2u;
  if (q == 1) { v = r.c; } else if (q == 2) { v = r.d; } else if (q == 3) { v = r.e; }
  return v[j & 3];
}

// Lamp k's light profile along w (unit, from the lamp): its table's share of the peak, bilinear.
fn lamp_profile(k: i32, w: vec3<f32>) -> f32 {
  let ax = lamps[LAMP_MAX + k];
  let gv = acos(clamp(dot(w, lamps[k].d.xyz), -1.0, 1.0)) * (f32(LAMP_PROF_V - 1) * 0.31830989);
  let hv = fract(atan2(dot(w, ax.c.xyz), dot(w, ax.p.xyz)) * 0.15915494 + 1.0) * f32(LAMP_PROF_H);
  let i0 = min(i32(gv), LAMP_PROF_V - 1);
  let i1 = min(i0 + 1, LAMP_PROF_V - 1);
  let j0 = min(i32(hv), LAMP_PROF_H - 1);
  let j1 = (j0 + 1) % LAMP_PROF_H;
  let fv = clamp(gv - f32(i0), 0.0, 1.0);
  let fh = clamp(hv - f32(j0), 0.0, 1.0);
  let base = 2 * LAMP_MAX + k * LAMP_PROF_V;
  let a = mix(lamp_cell(base + i0, j0), lamp_cell(base + i0, j1), fh);
  let b = mix(lamp_cell(base + i1, j0), lamp_cell(base + i1, j1), fh);
  return mix(a, b, fv);
}

// How much of lamp k's intensity (its peak) goes along w (unit, from the lamp): within a spot's cone, as an area light
// faces (its intensity falls with the cosine), by its light profile.
fn lamp_shape(k: i32, w: vec3<f32>) -> f32 {
  let lm = lamps[k];
  let kind = i32(lm.c.w + 0.5);
  let facing = dot(w, lm.d.xyz);
  var f = 1.0;
  if (kind == 1) { f = smoothstep(lm.d.w, lm.e.x, facing); }
  if (kind == 2) { f = max(facing, 0.0); }
  if (lm.e.w > 0.5 && f > 0.0) { f *= lamp_profile(k, w); }
  return f;
}
