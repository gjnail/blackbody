// Burnable surfaces, shared by the floor (a layer of cells on the simulation grid) and objects (a
// region of the object-burn atlas per burnable collider, in the collider's own frame, so the burn
// moves with the object). Each spot: x = fuel left (1 = a full load), y = catching progress
// (1 = alight), z = burnable (1/0), w = smoulder left (1 = just stopped flaming, 0 = burnt out).
// Needs colliders.wgsl for the object lookup; the includer declares `burn_obj` (texture_3d) and
// `slots` (array<BurnSlot>) where it looks objects up.

struct BurnSlot {
  lo: vec4<f32>,    // the region's corner in the collider's frame (m); w = z offset in the atlas
  dims: vec4<f32>,  // the region's size (cells); w = cell size (m)
};

fn burn_alight(b: vec4<f32>) -> bool { return b.z > 0.5 && b.y >= 1.0 && b.x > 0.0; }

// One step of a spot. T = gas temperature next to it, nb = how close its nearest burning neighbour
// is (1 = face, 1/sqrt(3) = corner, 0 = none), wet = extinguishing rate (1/s).
// sp = (catch temperature, 1 / catch time, creep speed (m/s), 1 / burn time); inv_smoulder = 1 / smoulder time.
fn burn_step(b_in: vec4<f32>, T: f32, nb: f32, wet: f32, dt: f32, h: f32, sp: vec4<f32>, inv_smoulder: f32) -> vec4<f32> {
  var b = b_in;
  if (b.z < 0.5) { return b; }
  if (b.y < 1.0) {
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
  if (wet > 0.0) {
    // water puts it out (it can catch again later if it still has fuel)
    let q = exp(-wet * dt);
    b.y = min(b.y, 0.999) * q;
    b.w *= q;
  }
  return b;
}

// How charred a spot looks: 0 untouched, 1 burnt out.
fn burn_char(b: vec4<f32>) -> f32 {
  if (b.z < 0.5 || b.y < 1.0) { return 0.0; }
  return 1.0 - smoothstep(0.0, 1.0, b.x);
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
