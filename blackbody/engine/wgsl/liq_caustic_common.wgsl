// Shared by the caustics passes: the liquid surface as the renderer sees it (without motion blur).
// The including file declares surf_t, lin and U: CParams.

struct CParams {
  n: vec4<f32>,       // grid dims, metres per cell
  nf: vec4<f32>,      // surface grid dims, surface cells per grid cell
  cm: vec4<f32>,      // caustic map dims (x, z), photons per texel along each side, _
  sun: vec4<f32>,     // toward the key light (grid space, unit), index of refraction
  absorb: vec4<f32>,  // absorption (1/m, rgb), murk extinction (1/m)
  lvl: vec4<f32>,     // water level (grid cells), open water on, _, blend width at the sides (cells)
};

fn box_range_c(ro: vec3<f32>, rd: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
  let safe = select(rd, vec3<f32>(1e-8), abs(rd) < vec3<f32>(1e-8));
  let inv = 1.0 / safe;
  let ta = (vec3<f32>(0.0) - ro) * inv;
  let tb = (n - ro) * inv;
  let t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
  return vec2<f32>(t0, t1);
}

fn phi_c(p: vec3<f32>) -> f32 {
  let n = U.n.xyz;
  var s = textureSampleLevel(surf_t, lin, p / n, 0.0).x / U.nf.w;
  let d = max(-p, p - n);
  if (U.lvl.y > 0.5) {
    s = max(s, d.y);
    let e = min(min(p.x, n.x - p.x), min(p.z, n.z - p.z));
    return mix(p.y - U.lvl.x, s, smoothstep(0.0, U.lvl.w, e));
  }
  return max(s, max(d.x, max(d.y, d.z)));
}

fn normal_c(p: vec3<f32>) -> vec3<f32> {
  let e = 0.75 / U.nf.w;
  let g = vec3<f32>(phi_c(p + vec3<f32>(e, 0.0, 0.0)) - phi_c(p - vec3<f32>(e, 0.0, 0.0)),
                    phi_c(p + vec3<f32>(0.0, e, 0.0)) - phi_c(p - vec3<f32>(0.0, e, 0.0)),
                    phi_c(p + vec3<f32>(0.0, 0.0, e)) - phi_c(p - vec3<f32>(0.0, 0.0, e)));
  let l = length(g);
  if (l < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
  return g / l;
}

fn fresnel_c(cosi: f32, eta_i: f32, eta_t: f32) -> f32 {
  let ci = clamp(cosi, 0.0, 1.0);
  let st = eta_i / eta_t * sqrt(max(0.0, 1.0 - ci * ci));
  if (st >= 1.0) { return 1.0; }
  let ct = sqrt(max(0.0, 1.0 - st * st));
  let rs = (eta_i * ci - eta_t * ct) / (eta_i * ci + eta_t * ct);
  let rp = (eta_t * ci - eta_i * ct) / (eta_t * ci + eta_i * ct);
  return 0.5 * (rs * rs + rp * rp);
}
