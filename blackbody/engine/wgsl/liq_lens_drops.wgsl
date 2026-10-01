// Drops on the lens: water that reached the camera (spray, splashes, rain) sits on the front element
// as drops, far out of focus. Each is a small lens of its own: it bends the picture behind it (more
// toward its edge, where it is steepest), softens it, and has a darker rim where it bends light
// away and a small bright highlight. Drops land at random over the shot, cling a few seconds while
// the bigger ones slowly run down, and dry off. Applied to the finished composite (the footage
// included: it is all behind the same lens), display and linear alike.
//
// Drops live on a grid of cells over the frame, one chance per cell per epoch of a few seconds;
// the neighbouring cells are checked too, so drops overlap cell edges.

struct Params {
  res: vec4<f32>,   // width, height, time (s), amount (0..3)
  cell: vec4<f32>,  // cell size (px), epoch (s), seed, _
};

@group(0) @binding(0) var src_lin: texture_2d<f32>;
@group(0) @binding(1) var src_disp: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var out_disp: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var out_lin: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn hash3(c: vec2<i32>, k: i32) -> vec4<f32> {
  var s = u32(c.x) * 73856093u ^ u32(c.y) * 19349663u ^ u32(k) * 83492791u ^ u32(U.cell.z) * 2654435761u;
  var o = vec4<f32>(0.0);
  for (var i = 0; i < 4; i++) {
    s = s * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    o[i] = f32((w >> 22u) ^ w) * (1.0 / 4294967296.0);
  }
  return o;
}

struct Drop {
  off: vec2<f32>,   // where the picture is seen from (px offset)
  rim: f32,         // darkening at the rim (0..1)
  soft: f32,        // blur radius (px)
  hi: f32,          // highlight
  cover: f32,       // 0..1
};

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let px = vec2<i32>(id.xy);
  if (f32(px.x) >= U.res.x || f32(px.y) >= U.res.y) { return; }
  let p = vec2<f32>(px) + vec2<f32>(0.5);
  let C = U.cell.x;
  let t = U.res.z;
  let ep = U.cell.y;
  var d: Drop;
  d.off = vec2<f32>(0.0);
  d.rim = 0.0;
  d.soft = 0.0;
  d.hi = 0.0;
  d.cover = 0.0;
  let base = vec2<i32>(floor(p / C));
  let chance = clamp(0.18 * U.res.w, 0.0, 0.9);
  for (var j = -2; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let c = base + vec2<i32>(i, j);
      let h0 = hash3(c, 0);
      // this cell's drop: which epoch it is in, how long ago it landed
      let ph = h0.x * ep;
      let k = i32(floor((t + ph) / ep));
      let age = t + ph - f32(k) * ep;
      let h = hash3(c, k + 1);
      if (h.x > chance) { continue; }
      let life = ep * (0.5 + 0.5 * h.y);
      if (age > life) { continue; }
      let R = C * (0.25 + 0.75 * h.z * h.z);
      // big drops run down the lens, slowly and then faster
      let run = select(0.0, (R / C - 0.6) * 60.0 * age * age, R > 0.6 * C);
      let ctr = (vec2<f32>(c) + vec2<f32>(0.2 + 0.6 * h.w, 0.2 + 0.6 * h.y)) * C + vec2<f32>(0.0, run);
      // it dries at the end of its life: shrinks and fades
      let dry = 1.0 - smoothstep(0.7 * life, life, age);
      let r = R * (0.6 + 0.4 * dry);
      let q = (p - ctr) / r;
      let rho2 = dot(q, q);
      if (rho2 >= 1.0) { continue; }
      // a spherical cap: its slope grows toward the edge; the picture seen through it is flipped and
      // shrunk around the centre (a lens far out of focus: mostly a soft, displaced blur)
      let z = sqrt(1.0 - rho2);
      let cov = smoothstep(1.0, 0.45, rho2) * dry * 0.9;
      if (cov > d.cover) {
        d.cover = cov;
        d.off = -q * r * (0.5 + 0.5 * (1.0 - z));
        d.rim = smoothstep(0.6, 1.0, rho2) * (1.0 - smoothstep(0.9, 1.0, rho2));
        d.soft = 0.45 * r;
        d.hi = pow(max(dot(normalize(vec3<f32>(q, z)), normalize(vec3<f32>(-0.4, -0.6, 0.7))), 0.0), 12.0);
      }
    }
  }
  let res = U.res.xy;
  let uv0 = p / res;
  let a_lin = textureLoad(src_lin, px, 0);
  let a_disp = textureLoad(src_disp, px, 0);
  if (d.cover <= 0.0) {
    textureStore(out_lin, px, a_lin);
    textureStore(out_disp, px, a_disp);
    return;
  }
  // what the drop shows: the picture from its offset, softened (a small ring of taps)
  var s_lin = vec4<f32>(0.0);
  var s_disp = vec4<f32>(0.0);
  for (var k = 0; k < 12; k++) {
    let a = f32(k) * 2.39996;
    let o = d.off + vec2<f32>(cos(a), sin(a)) * d.soft * sqrt((f32(k) + 0.5) / 12.0);
    let uv = clamp((p + o) / res, vec2<f32>(0.0), vec2<f32>(1.0));
    s_lin += textureSampleLevel(src_lin, lin, uv, 0.0);
    s_disp += textureSampleLevel(src_disp, lin, uv, 0.0);
  }
  s_lin /= 12.0;
  s_disp /= 12.0;
  let shade = 1.0 - 0.25 * d.rim;
  let hl = vec4<f32>(vec3<f32>(d.hi * 0.12), 0.0);
  let l_out = mix(a_lin, vec4<f32>(s_lin.rgb * shade, s_lin.a) + hl * max(dot(a_lin.rgb, vec3<f32>(0.333)), 0.3), d.cover);
  let d_out = mix(a_disp, vec4<f32>(s_disp.rgb * shade, s_disp.a) + hl, d.cover);
  textureStore(out_lin, px, l_out);
  textureStore(out_disp, px, clamp(d_out, vec4<f32>(0.0), vec4<f32>(1.0)));
}
