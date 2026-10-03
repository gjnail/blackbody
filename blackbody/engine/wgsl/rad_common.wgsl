// The frame's shared radiant sources (radiant.py), read: the radiation falling on a surface from all of them.
// The shader that includes this binds the list as `RL: array<vec4<f32>>` (three vec4 a source: (centre, softening m^2),
// (power of its volume W, of its surfaces W, owner, _), (its surfaces' power-weighted outward normal, _)) and its count
// as `RLC: array<u32>` (RLC[0]).

const RAD_SOURCES: u32 = 4224u;    // radiant.SOURCES
const RAD_PI: f32 = 3.14159265;

// The radiation (W/m^2) falling at x on a surface facing nrm (zero: as if it faced every source), from every source but
// those of `owner` (a thing does not warm itself by its own heat: -1 the gas and lava, -2 matter, 0..15 the objects).
// both: it takes radiation on either face (a sheet of cloth), else only on the face nrm points out of.
fn rad_irradiance(x: vec3<f32>, nrm: vec3<f32>, owner: f32, both: bool) -> f32 {
  let cnt = min(RLC[0], RAD_SOURCES);
  let ln = length(nrm);
  var e = 0.0;
  for (var i = 0u; i < cnt; i++) {
    let s1 = RL[3u * i + 1u];
    if (abs(s1.z - owner) < 0.5) { continue; }
    let s0 = RL[3u * i];
    let d = x - s0.xyz;
    let r2 = dot(d, d) + s0.w;
    let w = d * inverseSqrt(r2);             // (from the source toward x)
    var cr = 1.0;
    if (ln > 1.0e-6) {
      let c = -dot(nrm, w) / ln;
      cr = select(max(c, 0.0), abs(c), both);
      if (cr <= 0.0) { continue; }
    }
    // a volume throws its power every way; a surface forward (a flat one as the cosine, a closed one every way)
    var p = s1.x / (4.0 * RAD_PI);
    if (s1.y > 0.0) {
      let s2 = RL[3u * i + 2u];
      let lsn = length(s2.xyz);
      let flat = clamp(lsn / s1.y, 0.0, 1.0);
      let cs = max(dot(s2.xyz, w), 0.0) / max(lsn, 1.0e-12);
      p += s1.y * (flat * cs / RAD_PI + (1.0 - flat) / (4.0 * RAD_PI));
    }
    e += p * cr / r2;
  }
  return e;
}
