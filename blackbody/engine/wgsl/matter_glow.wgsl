// Hot matter's light on what is round it (stage.py): the matter's grid cut into blocks, and every block whose surface
// glows (mpm_surf_norm.wgsl's temperature, as the stage draws it: a blackbody, stage.wgsl matter_glow) a point light at
// its glow-weighted centre with the intensity of its glowing surface (radiance times area). ML[0].x: how many
// (finish); then two vec4 a light: (position, fire-local m; half the block), (intensity rgb, _).
//
// `lights` works out each block's light (BL, by block); `finish` keeps the brightest that fit after the hot objects'
// faces (MLC[0] of them, written before), in the order of the blocks: a fixed rule, so the lights do not flicker or
// differ between runs (an atomic append kept whichever the GPU happened to run first). How many glowed this frame goes
// in MLC[1], and the most left out in a frame since the stage last cleared it in MLC[2] (Engine.notices says so).

struct Params {
  n: vec4<f32>,      // the matter's nodes (x, y, z), block (nodes)
  o: vec4<f32>,      // node 0 (fire-local m), node spacing (m)
  mg: vec4<f32>,     // the glow: on, its radiance at 1300 K, the table's first temperature (K), 1 / its step
  k: vec4<f32>,      // most lights, blocks, _, _
  mgb: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var surf: texture_3d<f32>;   // x: the distance to the surface (m)
@group(0) @binding(1) var look2: texture_3d<f32>;  // x: the temperature there (K)
@group(0) @binding(2) var<storage, read_write> ML: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> MLC: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> BL: array<vec4<f32>>;   // each block's light, as ML's (w: its brightness)
@group(1) @binding(0) var<uniform> U: Params;

fn glow(T: f32) -> vec3<f32> {
  let f = (T - U.mg.z) * U.mg.w;
  if (f <= 0.0) { return vec3<f32>(0.0); }
  let fc = min(f, 14.999);
  let i = u32(fc);
  let t = fc - f32(i);
  let a = U.mgb[i];
  let b = U.mgb[i + 1u];
  var lg = mix(a.w, b.w, t);
  if (f > 15.0) { lg += (f - 15.0) * (b.w - a.w); }
  return mix(a.rgb, b.rgb, t) * pow(10.0, lg) * U.mg.y;
}

fn nblocks() -> vec3<u32> {
  let b = u32(U.n.w);
  return (vec3<u32>(U.n.xyz) + vec3<u32>(b - 1u)) / b;
}

@compute @workgroup_size(4, 4, 4)
fn lights(@builtin(global_invocation_id) id: vec3<u32>) {
  let nb = nblocks();
  if (any(id >= nb)) { return; }
  let bi = id.x + nb.x * (id.y + nb.y * id.z);
  let b = i32(U.n.w);
  let dims = vec3<i32>(U.n.xyz);
  let lo = vec3<i32>(id) * b;
  let hi = min(lo + vec3<i32>(b), dims);
  let dx = U.o.w;
  var power = vec3<f32>(0.0);
  var centre = vec3<f32>(0.0);
  var weight = 0.0;
  for (var z = lo.z; z < hi.z; z++) {
    for (var y = lo.y; y < hi.y; y++) {
      for (var x = lo.x; x < hi.x; x++) {
        let c = vec3<i32>(x, y, z);
        // (nodes on its surface, each about a node spacing square of it, facing every way: half of it toward a point)
        if (abs(textureLoad(surf, c, 0).x) > 0.75 * dx) { continue; }
        let e = glow(textureLoad(look2, c, 0).x);
        let l = e.r + e.g + e.b;
        if (l <= 1.0e-6) { continue; }
        power += e;
        centre += vec3<f32>(c) * l;
        weight += l;
      }
    }
  }
  if (weight <= 1.0e-6) {
    BL[2u * bi + 1u] = vec4<f32>(0.0);
    return;
  }
  let I = power * 0.5 * dx * dx;
  BL[2u * bi] = vec4<f32>(U.o.xyz + centre / weight * dx, 0.5 * f32(b) * dx);
  BL[2u * bi + 1u] = vec4<f32>(I, max(I.r + I.g + I.b, 1.0e-30));
}

var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> sel: vec4<u32>;
var<workgroup> glowing: u32;
var<workgroup> part: array<vec2<u32>, 256>;

// A block's brightness as a key that sorts as it does (a positive float's bits do); 0: it does not glow.
fn key_of(b: u32) -> u32 { return bitcast<u32>(max(BL[2u * b + 1u].w, 0.0)); }

@compute @workgroup_size(256, 1, 1)
fn finish(@builtin(local_invocation_index) t: u32) {
  let n = u32(U.k.y);
  let most = u32(U.k.x);
  let first = min(atomicLoad(&MLC[0]), most);
  // the brightness of the room-th brightest block, 8 bits at a time from the top (a radix select): sel.x and .y the
  // bits found so far and which they are, .z how many of the blocks that have them are still to take, .w 1 when every
  // glowing block fits
  let room = most - first;
  if (t == 0u) { sel = vec4<u32>(0u, 0u, room, 0u); }
  for (var digit = 0u; digit < 4u; digit++) {
    let shift = 24u - 8u * digit;
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    let s = sel;
    if (s.w == 0u) {    // (all fit: no more digits to find)
      for (var b = t; b < n; b += 256u) {
        let k = key_of(b);
        if (k != 0u && (k & s.y) == s.x) { atomicAdd(&hist[(k >> shift) & 255u], 1u); }
      }
    }
    workgroupBarrier();
    if (t == 0u) {
      if (digit == 0u) {
        var c = 0u;
        for (var j = 0u; j < 256u; j++) { c += atomicLoad(&hist[j]); }
        glowing = c;
        if (c <= s.z) { sel.w = 1u; }
      }
      if (sel.w == 0u) {
        var above = 0u;
        var bin = 0u;
        for (var j = 255; j >= 0; j--) {
          let c = atomicLoad(&hist[u32(j)]);
          if (above + c >= s.z) {
            bin = u32(j);
            break;
          }
          above += c;
        }
        sel = vec4<u32>(s.x | (bin << shift), s.y | (255u << shift), s.z - above, 0u);
      }
    }
    workgroupBarrier();
  }
  // the lights kept: brighter than the threshold T, and of those as bright the first `take` (in the order of the blocks)
  let s = sel;
  var T = s.x;
  var take = s.z;
  if (room == 0u) {
    T = 0xFFFFFFFFu;
    take = 0u;
  } else if (s.w != 0u) {
    T = 0u;
    take = 0u;
  }
  let per = (n + 255u) / 256u;
  let a = min(t * per, n);
  let e = min(a + per, n);
  var gt = 0u;
  var eq = 0u;
  for (var b = a; b < e; b++) {
    let k = key_of(b);
    if (k > T) { gt += 1u; } else if (k == T && k != 0u) { eq += 1u; }
  }
  part[t] = vec2<u32>(gt, eq);
  workgroupBarrier();
  for (var d = 1u; d < 256u; d <<= 1u) {
    var v = part[t];
    if (t >= d) { v += part[t - d]; }
    workgroupBarrier();
    part[t] = v;
    workgroupBarrier();
  }
  var before = vec2<u32>(0u);
  if (t > 0u) { before = part[t - 1u]; }
  var at = first + before.x + min(before.y, take);
  var eq_seen = before.y;
  for (var b = a; b < e; b++) {
    let k = key_of(b);
    var keep = k > T;
    if (k == T && k != 0u) {
      keep = eq_seen < take;
      eq_seen += 1u;
    }
    if (keep && at < most) {
      ML[1u + 2u * at] = BL[2u * b];
      ML[2u + 2u * at] = vec4<f32>(BL[2u * b + 1u].xyz, 0.0);
      at += 1u;
    }
  }
  if (t == 255u) {
    let kept = min(part[255].x + min(part[255].y, take), room);
    ML[0] = vec4<f32>(f32(first + kept), 0.0, 0.0, 0.0);
    atomicStore(&MLC[1], glowing);
    atomicMax(&MLC[2], glowing - min(kept, glowing));
  }
}
