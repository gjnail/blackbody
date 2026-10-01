// Sources. A stream source keeps each cell it covers topped up to the target particle count, so
// liquid pours out at the source velocity and the flow rate follows from speed times area. A fill
// source does the same once, filling its shape with liquid. The open water's own sources follow
// its surface (the level, or the sea's waves) and pour at its velocity: a top-up source (emitter
// c.w > 0) only refills cells more than a cell under it that have lost more than that share of
// their particles, the sea's fill (c.w < 0) fills everything under it. New particles carry their
// source's dye and density (emitter col = absorption rgb and scattering per metre, f.x = density
// relative to the liquid) when any source pours one.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl
//!include meshsdf.wgsl
//!include emitters.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,     // particles per cell, slot capacity, step seed, sub-cells per axis (ppc^(1/3), or 1)
  nb: vec4<f32>,    // narrow band on (1/0), band width (cells), the sea's peak wavenumber (1/m), the sea is on (1/0)
  cur: vec4<f32>,   // the open water's current (m/s, grid axes), particles carry dye and density (1/0)
  ecnt: vec4<f32>,  // source count
  em: array<Emitter, MAX_EMITTERS>,
};

@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> freelist: array<u32>;
@group(0) @binding(3) var<storage, read> cellcount: array<u32>;
@group(0) @binding(4) var sdf: texture_3d<f32>;
@group(0) @binding(5) var atlas: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> band: array<u32>;   // deep liquid (narrow band)
@group(0) @binding(7) var ocn_t: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> attr: array<vec4<u32>>;
//!include liq_level.wgsl
@group(1) @binding(0) var<uniform> U: Params;

fn sub_cell(k: u32, m: f32) -> vec3<u32> {
  let mi = u32(m);
  let j = k % (mi * mi * mi);
  return vec3<u32>(j % mi, (j / mi) % mi, j / (mi * mi));
}

fn alloc() -> i32 {
  let f = atomicSub(&ctr[C_FREE], 1);
  if (f > 0) { return i32(freelist[u32(f - 1)]); }
  atomicAdd(&ctr[C_FREE], 1);
  let s = atomicAdd(&ctr[C_COUNT], 1);
  if (s >= i32(U.k.y)) {
    atomicSub(&ctr[C_COUNT], 1);
    return -1;
  }
  return s;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  if (textureLoad(sdf, c, 0).x < 0.0) { return; }
  let ci = u32(c.x + n.x * (c.y + n.y * c.z));
  if (U.nb.x > 0.5 && band[ci] != 0u) { return; }   // deep liquid already
  let have = i32(cellcount[ci]);
  let want = i32(U.k.x);
  if (have >= want) { return; }
  let h = U.g.n.w;
  let wc = world_of(U.g, vec3<f32>(c) + vec3<f32>(0.5));
  let cnt = i32(U.ecnt.x);
  for (var s = 0; s < cnt; s++) {
    let em = U.em[s];
    if (em.d.x <= 0.0) { continue; }
    var sd = emitter_sdf(em, wc);
    if (sd > 0.87 * h) { continue; }
    let sea = em.c.w != 0.0;
    let lvl = level_at(vec2<f32>(c.xz) + vec2<f32>(0.5));
    if (sea) {
      // (still water is topped up a cell short of the level, so a passing trough is not filled in;
      // the sea's own surface, which has its troughs in it, right up to it)
      let margin = select(0.0, select(1.0, 1.0, U.nb.w > 0.5), em.c.w > 0.0);
      if (f32(c.y) + 0.5 > lvl - margin) { continue; }
      sd = max(sd, (f32(c.y) + 0.5 - lvl) * h);   // the surface is the top of the shape
    }
    // a top-up source (c.w > 0) only refills cells that have lost more than that share
    if (em.c.w > 0.0 && f32(have) >= (1.0 - em.c.w) * f32(want)) { continue; }
    if (U.nb.x > 0.5 && em.d.y > 0.5 && sd < -(U.nb.y + 1.0) * h) {
      band[ci] = 1u;   // deep inside a new volume of liquid: the grid carries it, no particles
      return;
    }
    let seed = ci * 2654435761u + u32(U.k.z) * 97u + u32(s) * 7919u;
    for (var k = have; k < want; k++) {
      // stratified: one particle per sub-cell of a jittered lattice, so a fresh volume starts at
      // an even density instead of the clumps and gaps of purely random placement
      let r = rand3(seed, u32(k) + 11u);
      let x = vec3<f32>(c) + (vec3<f32>(sub_cell(u32(k), U.k.w)) + r) / U.k.w;
      let w = world_of(U.g, x);
      if (emitter_sdf(em, w) > 0.0) { continue; }
      if (sea && x.y > level_at(x.xz)) { continue; }
      if (rand1(seed ^ (u32(k) * 747796405u + 1u)) > em.d.x) { continue; }
      let slot = alloc();
      if (slot < 0) { return; }
      let jr = rand3(seed, u32(k) + 101u) * 2.0 - vec3<f32>(1.0);
      var v = emitter_velocity(em, w);
      if (sea) { v = sea_velocity(x, U.nb.z, h) + U.cur.xyz; }
      let vel = v + jr * (length(v) * em.d.z);
      parts[u32(slot)] = new_particle(x, vel);
      if (U.cur.w > 0.5) { attr[u32(slot)] = make_attr(em.col, em.f.x); }
    }
    return;
  }
}
