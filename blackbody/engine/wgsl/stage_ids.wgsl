// The set's compositing passes (render/passes.py): for every pixel, what its samples see first and how much of it each
// thing covers (Cryptomatte and the per-object mattes), the normal and the place of the surface seen, and how far that
// moves on the screen to the next frame and from the last (motion vectors).
//
// It is the stage (stage.wgsl, with every binding its own kernels have, found by the same trace()) and one more
// binding: IDS, a buffer holding this pass's parameters first (ID_HEAD vec4s, then what each piece is a piece of, four
// to a vec4), then ID_OUT vec4s for each pixel of the band of rows this dispatch does:
//   0..2  the six things covering most of the pixel, the most first: (code, coverage, code, coverage); code 0: nothing
//   3     the normal of what is seen (world axes), averaged over the samples that see something; how much of the
//         pixel they are
//   4     where it is (world m), likewise; 1 when the pixel saw more things than it keeps count of
//   5     how far it moves on the screen (pixels: u to the right, v up) to the next frame, then from the last one,
//         averaged over all the samples
// A code says what was seen: an object (its row + 1), the floor, the matter, splinters, or what a piece is a piece of
// (render/passes.py names them).
//
// IDS head: 0..3 fire-local -> clip at the last frame, 4..7 at this frame, 8..11 at the next (columns);
// 12: the band's first row, its rows, samples per pixel, simulation seconds per frame;
// 13: the codes of the floor, the matter and splinters, and whether the sky moves with the camera (1/0);
// 14: where the pixels start (vec4s), pieces (count), the footage's scale in the picture (x, y: picture uv -> plate
//     uv, where its holdouts are: the pass is traced at the picture's size, not the plate's as the stage is).
//!include stage.wgsl

const ID_HEAD: u32 = 16u;   // vec4s of parameters at the head of IDS (render/passes.py ID_HEAD)
const ID_OUT: u32 = 6u;     // vec4s per pixel
const ID_KEEP: i32 = 8;     // things a pixel keeps count of; the six covering most of it are written
const ID_SKY: f32 = 1.0e4;  // m: how far off the sky is taken to be (it moves only as the camera turns)

@group(0) @binding(39) var<storage, read_write> IDS: array<vec4<f32>>;

fn ids_mat(at: u32) -> mat4x4<f32> {
  return mat4x4<f32>(IDS[at], IDS[at + 1u], IDS[at + 2u], IDS[at + 3u]);
}

// The footage's lens on a pinhole uv (x_d = x_u (1 + k1 r_u^2), r of the half-diagonal): undistort the other way.
fn ids_distort(uv: vec2<f32>, k1: f32) -> vec2<f32> {
  if (k1 == 0.0) { return uv; }
  let asp = vec2<f32>(U.res.x / U.res.y, 1.0);
  let hd = 0.5 * length(asp);
  let u = (uv - vec2<f32>(0.5)) * asp / hd;
  return u * (1.0 + k1 * dot(u, u)) * hd / asp + vec2<f32>(0.5);
}

// Where fire-local p lands in the picture through m (pixels, y down), through the lens and the fit as stage_pixel
// looks; z < 0: behind the camera.
fn ids_px(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
  let c = m * vec4<f32>(p, 1.0);
  if (c.w <= 1e-6) { return vec3<f32>(0.0, 0.0, -1.0); }
  let ndc = c.xy / c.w;
  let uv = ids_distort(vec2<f32>(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y), U.fit.z);
  return vec3<f32>(((uv - vec2<f32>(0.5)) * U.fit.xy + vec2<f32>(0.5)) * U.res.xy, 1.0);
}

// How far a point at fire-local p moving at v (m/s) moves on the screen to the next frame and from the last (pixels,
// u right, v up): the camera's own motion is in the three frames' matrices.
fn ids_motion(p: vec3<f32>, v: vec3<f32>) -> vec4<f32> {
  let dt = IDS[12].w;
  let c = ids_px(ids_mat(4u), p);
  let f = ids_px(ids_mat(8u), p + v * dt);
  let b = ids_px(ids_mat(0u), p - v * dt);
  var out = vec4<f32>(0.0);
  if (c.z > 0.0 && f.z > 0.0) { out = vec4<f32>(f.x - c.x, c.y - f.y, 0.0, 0.0); }
  if (c.z > 0.0 && b.z > 0.0) { out = vec4<f32>(out.xy, b.x - c.x, c.y - b.y); }
  return out;
}

// What piece k is a piece of (its code).
fn ids_owner(k: u32) -> f32 {
  let v = IDS[ID_HEAD + k / 4u];
  let j = k % 4u;
  return select(select(v.w, v.z, j == 2u), select(v.y, v.x, j == 0u), j < 2u);
}

// The code of what a trace hit (0: nothing).
fn ids_code(h: Hit) -> f32 {
  if (h.id < 0) { return 0.0; }
  if (h.id < FLOOR) { return f32(h.id + 1); }
  if (h.id == FLOOR) { return IDS[13].x; }
  if (h.id == MATTER) { return IDS[13].y; }
  if (h.id == FRAY) { return IDS[13].z; }
  if (h.id >= PIECE) {
    let k = u32(h.id - PIECE);
    if (f32(k) < IDS[14].y) { return ids_owner(k); }
  }
  return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn ids_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(i32(id.x), i32(id.y) + i32(IDS[12].x));
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y || f32(id.y) >= IDS[12].y) { return; }
  let ns = max(i32(IDS[12].z + 0.5), 1);
  let footage = U.stage.w > 0.5;
  let seed = u32(px.x) * 1973u + u32(px.y) * 9277u + u32(U.depth.z) * 26699u;
  var code: array<f32, ID_KEEP>;
  var cov: array<f32, ID_KEEP>;
  var kept = 0;
  var over = 0.0;
  var nrm = vec3<f32>(0.0);
  var pos = vec3<f32>(0.0);
  var seen = 0.0;
  var mv = vec4<f32>(0.0);
  for (var i = 0; i < ns; i++) {
    // samples on a rotated grid over the pixel, each at its own time in the shutter (as stage_pixel's)
    var off = vec2<f32>(0.0);
    if (ns > 1) {
      off = vec2<f32>(fract(f32(i) * 0.7548776662 + 0.25), fract(f32(i) * 0.5698402910 + 0.6)) - vec2<f32>(0.5);
    }
    g_jit = rand1(seed + u32(i) * 31337u + 17u);
    g_tau = 0.0;
    if (U.res.w > 0.0) { g_tau = U.res.w * ((f32(i) + rand1(seed + u32(i) * 7919u)) / f32(ns) - 0.5); }
    let pp = vec2<f32>(px) + vec2<f32>(0.5) + off;
    let puv = pp / U.res.xy;
    let uv = undistort((puv - vec2<f32>(0.5)) / U.fit.xy + vec2<f32>(0.5), U.fit.z);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let pn = U.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let pf = U.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let ro_w = pn.xyz / pn.w;
    let rd_w = normalize(pf.xyz / pf.w - ro_w);
    let ro = (U.w2l * vec4<f32>(ro_w, 1.0)).xyz;
    let rd = normalize((U.w2l * vec4<f32>(rd_w, 0.0)).xyz);
    // the footage's own surfaces hide what is behind them, and its matte what it covers (as see() has it)
    var t_foot = 1.0e9;
    var keep = 1.0;
    if (footage) {
      let fuv = (puv - vec2<f32>(0.5)) * IDS[14].zw + vec2<f32>(0.5);
      if (U.foot.w > 0.5) {
        let fd = footage_distance(fuv, rd_w);
        if (fd > 0.0) { t_foot = max(fd - U.fwd.w / max(dot(rd_w, U.fwd.xyz), 1e-3), 0.0); }
      }
      if (U.foot.z > 0.5) { keep = 1.0 - clamp(textureSampleLevel(hold, lin, fuv, 0.0).x, 0.0, 1.0); }
    }
    let h = trace(ro, rd, 0.0, select(1.0e5, t_foot, footage), 1.0, !footage);
    let c = ids_code(h);
    if (c <= 0.0) {
      // the sky turns with the camera (the footage's own motion is not known)
      if (IDS[13].w > 0.5) { mv += ids_motion(ro + rd * ID_SKY, vec3<f32>(0.0)); }
      continue;
    }
    let p = ro + rd * h.t;
    var n = -rd;
    var v = vec3<f32>(0.0);
    if (h.id < FLOOR) {
      if (plain(h.id)) { n = plain_normal(h.id, p); } else { n = obj_normal(h.id, p, max(0.5 * U.fit.w * h.t, 2.0e-4)); }
      v = col_velocity(obj(h.id), p);
    } else if (h.id == FLOOR) {
      n = vec3<f32>(0.0, 1.0, 0.0);
    } else if (F_MARCH && h.id == MATTER) {
      n = matter_normal(p);
    } else if (F_PIECES && h.id >= PIECE) {
      let kp = u32(h.id - PIECE);
      let pose = piece_pose(kp);
      n = quat_rotate(pose[1], normalize(PL[u32(PC[kp].v.w) + u32(max(g_plane, 0))].xyz));
      v = PC[kp].v.xyz + cross(PC[kp].o.xyz, p - pose[0].xyz);
    }
    if (dot(n, rd) > 0.0) { n = -n; }
    mv += ids_motion(p, v);
    if (keep <= 0.0) { continue; }
    nrm += to_world_dir(n) * keep;
    pos += (ro_w + rd_w * h.t) * keep;
    seen += keep;
    var j = 0;
    loop {
      if (j >= kept || code[j] == c) { break; }
      j++;
    }
    if (j < kept) {
      cov[j] += keep;
    } else if (kept < ID_KEEP) {
      code[kept] = c;
      cov[kept] = keep;
      kept++;
    } else {
      over = 1.0;
    }
  }
  let inv = 1.0 / f32(ns);
  let o = u32(IDS[14].x) + (id.y * u32(U.res.x) + id.x) * ID_OUT;
  // the six covering the most of it, the most first
  var r = array<vec2<f32>, 6>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
  for (var k = 0; k < 6; k++) {
    var best = -1;
    for (var m = 0; m < kept; m++) {
      if (cov[m] > 0.0 && (best < 0 || cov[m] > cov[best])) { best = m; }
    }
    if (best < 0) { break; }
    r[k] = vec2<f32>(code[best], cov[best] * inv);
    cov[best] = 0.0;
  }
  IDS[o] = vec4<f32>(r[0], r[1]);
  IDS[o + 1u] = vec4<f32>(r[2], r[3]);
  IDS[o + 2u] = vec4<f32>(r[4], r[5]);
  let ws = select(0.0, 1.0 / seen, seen > 0.0);
  IDS[o + 3u] = vec4<f32>(nrm * ws, seen * inv);
  IDS[o + 4u] = vec4<f32>(pos * ws, over);
  IDS[o + 5u] = mv * inv;
}
