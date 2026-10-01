// Clouds: what the water does in each cell (engine/cloud.py), every substep: a bulk scheme in the manner of
// Kessler (1969) for warm rain and Lin, Farley and Orville (1983) for ice, with the latent heat of every
// change of phase going into the air's temperature.
//  - condensation: vapour beyond saturation condenses (saturation adjustment), into cloud water above
//    freezing and more and more into cloud ice the colder it is (all ice below -40 C); cloud evaporates
//    into air that is not saturated;
//  - glaciation: in mixed cloud ice grows at the water's expense (it holds less vapour: Bergeron);
//    cloud water freezes at -40 C, cloud ice melts above freezing;
//  - warm rain: cloud water past a threshold coalesces into rain (autoconversion), rain sweeps up cloud
//    water (accretion), rain evaporates below the cloud in unsaturated air (and cools it: downdrafts);
//  - snow: cloud ice aggregates into snow, which grows by deposition in mixed cloud and sublimates in
//    dry air; snow collecting supercooled cloud water (riming) becomes graupel when heavily rimed;
//  - graupel and hail grow by sweeping up cloud water and rain in the updraft; rain carried above the
//    freezing level freezes into them (Bigg);
//  - melting: snow and graupel fall below the freezing level and melt into rain (taking the heat that
//    melts them from the air: the melting layer cools).
//!include cloud_common.wgsl

@group(0) @binding(0) var A_in: texture_3d<f32>;
@group(0) @binding(1) var B_in: texture_3d<f32>;
@group(0) @binding(2) var A_out: texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var B_out: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<storage, read> base: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> U: Cloud;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let a = textureLoad(A_in, c, 0);
  let b = textureLoad(B_in, c, 0);
  let b0 = base[u32(c.y) * 2u];
  let b1 = base[u32(c.y) * 2u + 1u];
  let th0 = b0.x;
  let pi = b0.y;
  let p = b1.w;
  let dt = U.g.bc.w;
  var tk = (th0 + a.x) * pi;
  var qv = max(a.y, 0.0);
  var qc = max(a.z, 0.0);
  var qr = max(a.w, 0.0);
  var qi = max(b.x, 0.0);
  var qs = max(b.y, 0.0);
  var qg = max(b.z, 0.0);
  let on = U.vort.w > 0.5;
  if (on) {
    // 1. condensation and evaporation of cloud (saturation adjustment, mixed phase)
    var tc = tk - CK;
    let fw = c_liquid_share(tc);
    let qsw = c_qvs_w(tc, p);
    let qsi = c_qvs_i(tc, p);
    let qvs = fw * qsw + (1.0 - fw) * qsi;
    let L = fw * CLV + (1.0 - fw) * CLS;
    let dqs = qvs * L / (CRV * tk * tk);
    let gamma = 1.0 + L / CCP * dqs;
    if (qv > qvs) {
      let d = (qv - qvs) / gamma;
      qv -= d;
      qc += fw * d;
      qi += (1.0 - fw) * d;
      tk += L / CCP * d;
    } else if (qc + qi > 0.0) {
      var deficit = (qvs - qv) / gamma;
      let ec = min(qc, deficit);
      qc -= ec;
      qv += ec;
      tk -= CLV / CCP * ec;
      deficit -= ec;
      let ei = min(qi, deficit);
      qi -= ei;
      qv += ei;
      tk -= CLS / CCP * ei;
    }
    tc = tk - CK;
    // 2. glaciation: Bergeron in mixed cloud, homogeneous freezing, cloud ice melting
    if (tc < -40.0) {
      qi += qc;
      tk += CLF / CCP * qc;
      qc = 0.0;
    } else if (tc < 0.0 && qc > 0.0) {
      let f = (1.0 - c_liquid_share(tc)) * (1.0 - exp(-dt / max(U.micro.w, 1.0)));
      let d = qc * f;
      qc -= d;
      qi += d;
      tk += CLF / CCP * d;
    } else if (tc > 0.0 && qi > 0.0) {
      let d = min(qi, qi * tc * dt / 60.0);
      qi -= d;
      qc += d;
      tk -= CLF / CCP * d;
    }
    tc = tk - CK;
    // 3. warm rain
    let autoc = U.micro.y * max(qc - U.micro.x, 0.0);
    let accr = 2.2 * qc * pow(qr, 0.875);
    let dr = min(qc, (autoc + accr) * dt);
    qc -= dr;
    qr += dr;
    // rain evaporating into unsaturated air
    let qsw2 = c_qvs_w(tc, p);
    if (qr > 0.0 && qv < qsw2 && qc <= 0.0) {
      let er = min(qr, (1.0 - qv / qsw2) * qr * dt / 300.0);
      qr -= er;
      qv += er;
      tk -= CLV / CCP * er;
    }
    tc = tk - CK;
    // 4. snow, graupel and hail
    let qsi2 = c_qvs_i(tc, p);
    if (tc < 0.0) {
      // ice aggregating into snow (Lin et al.)
      let saut = 1.0e-3 * exp(0.025 * tc) * max(qi - 1.0e-4, 0.0);
      // snow sweeping up cloud ice
      let saci = 0.1 * exp(0.025 * tc) * qi * pow(qs * 1.0e3, 0.8);
      let ds = min(qi, (saut + saci) * dt);
      qi -= ds;
      qs += ds;
      // deposition growth of snow in ice-supersaturated air; sublimation in dry air
      let sdep = qs * (qv / qsi2 - 1.0) / 600.0;
      let dd = clamp(sdep * dt, -qs, max(qv - qsi2, 0.0));
      qs += dd;
      qv -= dd;
      tk += CLS / CCP * dd;
      // riming: snow sweeping up supercooled cloud water; heavily rimed snow becomes graupel
      let sacw = 4.0e-3 * qc * pow(qs * 1.0e3, 0.8);
      let rim = min(qc, sacw * dt);
      qc -= rim;
      qs += rim;
      tk += CLF / CCP * rim;
      if (qc > 5.0e-4) {
        let gs = min(qs, 0.5 * rim + qs * 1.0e-3 * dt);
        qs -= gs;
        qg += gs;
      }
      // graupel and hail sweeping up cloud water and rain in the updraft, freezing them on
      let gacw = 1.0e-2 * qc * pow(qg * 1.0e3, 0.8);
      let gacr = 5.0e-3 * qr * pow(qg * 1.0e3, 0.8);
      let gw = min(qc, gacw * dt);
      let gr = min(qr, gacr * dt);
      qc -= gw;
      qr -= gr;
      qg += gw + gr;
      tk += CLF / CCP * (gw + gr);
      // rain carried above the freezing level freezes (Bigg): into graupel, hail
      let frz = min(qr, qr * 1.0e-4 * (exp(-0.66 * tc) - 1.0) * dt);
      qr -= frz;
      qg += frz;
      tk += CLF / CCP * frz;
      // graupel sublimating in dry air
      if (qv < qsi2 && qc <= 0.0) {
        let gsub = min(qg, qg * (1.0 - qv / qsi2) * dt / 900.0);
        qg -= gsub;
        qv += gsub;
        tk -= CLS / CCP * gsub;
      }
    } else {
      // 5. melting: snow and graupel below the freezing level melt into rain
      let k = min(tc, 10.0) / 600.0;
      let ms = min(qs, qs * k * dt);
      let mg = min(qg, qg * 0.5 * k * dt);
      qs -= ms;
      qg -= mg;
      qr += ms + mg;
      tk -= CLF / CCP * (ms + mg);
      // melting snow evaporates a little into unsaturated air too
      if (qv < qsw2) {
        let es = min(qs, qs * (1.0 - qv / qsw2) * dt / 600.0);
        qs -= es;
        qv += es;
        tk -= CLS / CCP * es;
      }
    }
  }
  // (safety: more than 25 g/kg of graupel and hail cannot be held aloft; the excess is shed as rain, as big
  // hail sheds the water it cannot freeze)
  if (qg > 0.025) {
    qr += qg - 0.025;
    qg = 0.025;
  }
  let th = tk / pi;
  // (for the renderer) cloud too small for the grid: air near saturation over a cell is partly cloudy, its
  // moister eddies condensed (a cloud fraction from the relative humidity, after Sundqvist): thin cloud that
  // softens the edges the cells would draw as cubes
  let tcf = tk - CK;
  let fwf = c_liquid_share(tcf);
  let qvsf = fwf * c_qvs_w(tcf, p) + (1.0 - fwf) * c_qvs_i(tcf, p);
  let rh = qv / max(qvsf, 1e-9);
  let cf = smoothstep(0.96, 1.0, rh);
  let sub = select(0.0, 0.002 * qvsf * cf * cf, on);
  textureStore(A_out, c, vec4<f32>(th - th0, max(qv, 0.0), max(qc, 0.0), max(qr, 0.0)));
  textureStore(B_out, c, vec4<f32>(max(qi, 0.0), max(qs, 0.0), max(qg, 0.0), sub));
}
