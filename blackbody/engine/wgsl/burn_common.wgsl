// Burnable surfaces, shared by the floor (a layer of cells on the simulation grid) and objects (a
// region of the object-burn atlas per burnable collider, in the collider's own frame and in proportion
// to its size, so the burn moves with the object and grows and shrinks with it). Each spot: x = fuel
// left (1 = a full load), y = catching progress (1 = alight; below 0: soaked, -1 = dripping wet),
// z = burnable (1, or 2 once it has burnt), w = smoulder left (1 = just stopped flaming, 0 = burnt out).
// Needs colliders.wgsl for the object lookup; the includer declares `burn_obj` (texture_3d) and
// `slots` (array<BurnSlot>) where it looks objects up.

// A collider's region, as it was laid out (at the collider's size then; burn_from_layout carries it to its size now).
struct BurnSlot {
  lo: vec4<f32>,    // the region's corner in the collider's frame (m); w = z offset in the atlas
  dims: vec4<f32>,  // the region's size (cells); w = the largest cell's size (m)
  cell: vec4<f32>,  // a cell's size along each axis (m: a simulation cell, or more along an axis the collider
                    // is too big for)
  size: vec4<f32>,  // what the collider was scaled by then (col_scale)
};

// Where point q of collider k's frame (m) at the region's layout lies on the collider at its scale sc now, and
// back: in proportion to its size, and exactly q along an axis whose size has not changed (a still collider
// keeps the layout and the fuel pattern it was given).
fn burn_from_layout(q: vec3<f32>, sc: vec3<f32>, s: BurnSlot) -> vec3<f32> {
  return select(q * (sc / s.size.xyz), q, sc == s.size.xyz);
}
fn burn_to_layout(q: vec3<f32>, sc: vec3<f32>, s: BurnSlot) -> vec3<f32> {
  return select(q * (s.size.xyz / sc), q, sc == s.size.xyz);
}

// How much the collider (scale sc now) has grown since its region was laid out, the mean over its axes (exactly
// 1 while it has not).
fn burn_grown(sc: vec3<f32>, s: BurnSlot) -> f32 {
  if (all(sc == s.size.xyz)) { return 1.0; }
  return dot(sc / s.size.xyz, vec3<f32>(1.0 / 3.0));
}

fn burn_alight(b: vec4<f32>) -> bool { return b.z > 0.5 && b.y >= 1.0 && b.x > 0.0; }

// One step of a spot. T = gas temperature next to it, nb = how close its nearest burning neighbour
// is (1 = face, 1/sqrt(3) = corner, 0 = none), wet = extinguishing rate (1/s).
// sp = (catch temperature, 1 / catch time, creep speed (m/s), 1 / burn time); inv_smoulder = 1 / smoulder time;
// dry = 1 / drying time (0: water puts it out but does not soak it).
fn burn_step(b_in: vec4<f32>, T: f32, nb: f32, wet: f32, dt: f32, h: f32, sp: vec4<f32>, inv_smoulder: f32,
             dry: f32) -> vec4<f32> {
  var b = b_in;
  if (b.z < 0.5) { return b; }
  if (b.y < 0.0) {
    // soaked: it has to dry out before it can catch again (faster in hot gas)
    let heat = smoothstep(sp.x, sp.x * 1.5 + 0.02, T) * sp.y;
    b.y = min(b.y + (dry + 0.25 * heat) * dt, 0.0);
  } else if (b.y < 1.0) {
    if (b.x > 0.0 && wet <= 0.0) {
      // heats up in hot gas, or creeps from a burning neighbour: the front moves one cell in (cell / creep speed) s
      let heat = smoothstep(sp.x, sp.x * 1.5 + 0.02, T) * sp.y + sp.z / h * nb;
      if (heat > 0.0) {
        b.y = min(b.y + heat * dt, 1.0);
      } else {
        b.y = max(b.y - 0.25 * sp.y * dt, 0.0);  // a surface that was warming cools off again
      }
    }
  } else if (b.x > 0.0) {
    b.x = max(b.x - sp.w * dt, 0.0);
    if (b.x <= 0.0) { b.w = 1.0; }
  } else {
    b.w = max(b.w - inv_smoulder * dt, 0.0);
  }
  if (b.y >= 1.0) { b.z = 2.0; }  // it has burnt: its scorch stays when it is put out
  if (wet > 0.0) {
    // water puts it out (it can catch again later if it still has fuel), and soaks it
    let q = exp(-wet * dt);
    if (dry > 0.0) {
      b.y = mix(-1.0, min(b.y, 0.999), q);
    } else {
      b.y = min(b.y, 0.999) * q;
    }
    b.w *= q;
  }
  return b;
}

// How charred a spot looks: 0 untouched, 1 burnt out.
fn burn_char(b: vec4<f32>) -> f32 {
  if (b.z < 0.5 || (b.z < 1.5 && b.y < 1.0)) { return 0.0; }
  return 1.0 - smoothstep(0.0, 1.0, b.x);
}

// How wet a spot is (0 dry .. 1 dripping).
fn burn_wet(b: vec4<f32>) -> f32 {
  if (b.z < 0.5) { return 0.0; }
  return clamp(-b.y, 0.0, 1.0);
}

// Fuel, heat and smoke a spot gives off (per second, temperature, per second); surf = (fuel, heat,
// smoke, smoulder smoke) rates.
fn burn_emission(b: vec4<f32>, surf: vec4<f32>) -> vec3<f32> {
  if (burn_alight(b)) {
    let a = smoothstep(0.0, 0.15, b.x);
    return vec3<f32>(surf.x * a, surf.y * a, surf.z * a);
  }
  if (b.z > 0.5 && b.w > 0.0) { return vec3<f32>(0.0, surf.y * 0.45 * b.w, surf.w * b.w); }
  return vec3<f32>(0.0);
}
