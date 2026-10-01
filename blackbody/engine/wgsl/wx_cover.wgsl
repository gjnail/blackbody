// Weather: what lies on the ground (and on the colliders' tops), one thread per cell of the cover map
// (engine/weather.py), each substep. It takes in what the precipitation particles left there this step
// (wx_step.wgsl) and keeps a snowpack:
//  - fresh snow settles at the density it falls at, which depends on the air's temperature (Hedstrom and
//    Pomeroy 1998: 70 kg/m^3 in hard frost, about 120 at the freezing point, wetter and heavier above);
//    graupel and ice pellets pack as a granular layer of about 500 kg/m^3;
//  - it melts by the heat the air gives it above freezing (turbulent exchange, stronger in wind), by the
//    warmth of what it lies on, and by rain falling on it; its meltwater is held up to a few per cent of
//    its mass and drains away past that; in the cold the water it holds refreezes;
//  - it settles under its own weight, faster when warm and wet (toward 300 kg/m^3 dry, 450 wet);
//  - drops that freeze on contact (freezing rain) build a clear glaze of ice, which melts back above
//    freezing; with no snow, the water that lands only wets the surface, which dries again.
// State per cell (rgba): snow depth (m), snow water equivalent (kg/m^2), liquid water held (kg/m^2),
// glaze (m). A copy goes to a texture for the renderer, with wetness in place of the water held.

struct Params {
  map: vec4<f32>,   // nx, nz, cell area (m^2), dt (s)
  air: vec4<f32>,   // air temperature at the ground (C), wind speed (m/s), heat speed-up, freezing rain (kg/m^2/s)
  rain: vec4<f32>,  // rain landing (kg/m^2/s), its temperature (C), build-up speed-up, snow already lying (m, first step)
};

@group(0) @binding(0) var<storage, read_write> cover_dep: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cover: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> wet: array<f32>;
@group(0) @binding(3) var surf: texture_2d<f32>;
@group(0) @binding(4) var out_cover: texture_storage_2d<rgba32float, write>;
@group(1) @binding(0) var<uniform> U: Params;

const FX_UG: f32 = 1.0e9;
const LF: f32 = 3.336e5;
const RHO_ICE: f32 = 917.0;

fn fresh_density(t: f32) -> f32 {
  if (t <= 0.0) { return 67.92 + 51.25 * exp(t / 2.59); }
  return min(119.17 + 20.0 * sqrt(t), 250.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nx = u32(U.map.x);
  let nz = u32(U.map.y);
  if (id.x >= nx || id.y >= nz) { return; }
  let i = id.y * nx + id.x;
  let area = U.map.z;
  let dt = U.map.w;
  let sf = textureLoad(surf, vec2<i32>(id.xy), 0);
  var c = cover[i];
  var w = wet[i];
  // (sped up as a time-lapse of the build-up: what lands counts that many times over)
  let k = max(U.rain.z, 1.0);
  let snow = f32(atomicExchange(&cover_dep[i * 4u + 0u], 0u)) / FX_UG / area * k;    // kg/m^2
  let water = f32(atomicExchange(&cover_dep[i * 4u + 1u], 0u)) / FX_UG / area * k;
  let glaze = f32(atomicExchange(&cover_dep[i * 4u + 2u], 0u)) / FX_UG / area * k;
  let grain = f32(atomicExchange(&cover_dep[i * 4u + 3u], 0u)) / FX_UG / area * k;
  if (sf.x < -100.0) {
    // no surface here (no ground): nothing lies
    textureStore(out_cover, vec2<i32>(id.xy), vec4<f32>(0.0));
    return;
  }
  let ta = U.air.x;
  // snow does not stay on a slope much steeper than its angle of repose: it slides or is never caught
  let ny = sqrt(max(1.0 - dot(sf.zw, sf.zw), 0.0));
  let hold = smoothstep(0.42, 0.7, ny);
  if (U.rain.w > 0.0 && sf.y <= 1.0) {
    // snow lying from before the shot (settled: 150 kg/m^3), where the surface is cold enough to keep it
    let d0 = U.rain.w * hold;
    c = vec4<f32>(d0, d0 * 150.0, 0.0, c.w);
  }
  // what fell: snow at its fresh density, grains packed, glaze as clear ice
  let rain_in = U.rain.x * dt;
  var depth = c.x;
  var swe = c.y;
  var held = c.z;
  var gl = c.w;
  let fr = U.air.w * dt;
  if (sf.y <= 0.0 || ta <= 0.0) { gl += (glaze + fr) / RHO_ICE; } else { w = min(w + (glaze + fr) * 2.0, 1.0); }
  depth += (snow / fresh_density(ta) + grain / 500.0) * hold;
  swe += (snow + grain) * hold;
  // water landing on snow soaks into it; on bare ground it wets it
  let liquid_in = water + rain_in;
  if (swe > 0.0) { held += liquid_in; } else { w = min(w + liquid_in * 2.0, 1.0); }
  // heat: from the air above freezing (turbulent exchange), from the surface under it, from the rain
  let speed = max(U.air.z, 1.0);
  let h_air = 6.0 + 4.0 * U.air.y;
  var q = h_air * ta + 15.0 * sf.y + U.rain.x * 4186.0 * max(U.rain.y, 0.0);
  if (swe > 0.0) {
    if (q > 0.0) {
      // melting: the snow becomes water held in what is left of it
      let melt = min(q * dt * speed / LF, swe);
      let rho = depth / max(swe, 1e-9);
      swe -= melt;
      held += melt;
      depth = max(depth - melt * rho, 0.0);
    } else if (held > 0.0) {
      // cold: the water it holds freezes into it
      let frz = min(-q * dt * speed / LF, held);
      held -= frz;
      swe += frz;
    }
    // it holds about 5 % of its mass as water; the rest drains away (it wets what is under it)
    let cap = 0.05 * swe;
    if (held > cap) {
      w = min(w + (held - cap) * 2.0, 1.0);
      held = cap;
    }
    // settling: toward 300 kg/m^3 dry, 450 wet, over hours (faster warm)
    if (depth > 1e-5) {
      let rho = swe / depth;
      let goal = select(300.0, 450.0, held > 0.01 * swe);
      let rate = (1.0 / 21600.0) * exp(0.08 * min(ta, 0.0)) * speed;
      let rho2 = rho + (goal - rho) * min(rate * dt, 1.0);
      depth = swe / max(rho2, 50.0);
    }
    if (swe < 1e-4) {
      w = min(w + (swe + held) * 2.0, 1.0);
      swe = 0.0;
      held = 0.0;
      depth = 0.0;
    }
  } else {
    held = 0.0;
    depth = 0.0;
  }
  // glaze melts above freezing
  if (gl > 0.0 && q > 0.0 && swe <= 0.0) {
    gl = max(gl - q * dt * speed / (LF * RHO_ICE), 0.0);
    w = min(w + 0.1, 1.0);
  }
  // a wet surface dries
  w = max(w - dt * speed * select(1.0 / 1800.0, 1.0 / 600.0, ta > 5.0), 0.0);
  c = vec4<f32>(depth, swe, held, gl);
  cover[i] = c;
  wet[i] = w;
  let wetness = select(w, clamp(held / max(0.05 * swe, 1e-6), 0.0, 1.0), swe > 0.0);
  textureStore(out_cover, vec2<i32>(id.xy), vec4<f32>(depth, swe, wetness, gl));
}
