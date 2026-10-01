// Weather: precipitation that fell into the liquid becomes liquid, one thread per liquid cell (engine/
// weather.py). wx_step.wgsl adds the mass, heat and downward momentum of every particle that enters a
// cell; once a cell has gathered a liquid particle's worth it becomes one, carrying that heat (its
// enthalpy, as the liquid's heat model counts it): snow and hail that fell into warm water arrive as
// cold, partly frozen water that melts there, cooling it; on water near freezing they float as slush.
// At most one a cell a step, placed in the top half of the cell, so the water is not packed.
//!include common.wgsl
//!include liq_common.wgsl
//!include noise.wgsl

struct Params {
  g: Grid,
  k: vec4<f32>,   // liquid particle mass (micrograms), liquid capacity (slots), heat on (1/0), step seed
  k2: vec4<f32>,  // dye carried (1/0)
};

@group(0) @binding(0) var<storage, read_write> liq_dep: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(2) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(4) var<storage, read_write> therm: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> attr: array<vec4<u32>>;
@group(1) @binding(0) var<uniform> U: Params;

const FX_HM: f32 = 0.0625;

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
  let i = nidx(c, n) * 4u;
  let mp = i32(U.k.x);
  let m = atomicLoad(&liq_dep[i]);
  if (m < mp || mp <= 0) { return; }
  let mh = atomicLoad(&liq_dep[i + 1u]);
  let mv = atomicLoad(&liq_dep[i + 2u]);
  let icy = atomicLoad(&liq_dep[i + 3u]);
  let slot = alloc();
  if (slot < 0) { return; }
  let fm = f32(m);
  let h = f32(mh) / FX_HM / fm;          // kJ/kg
  let vy = f32(mv) / 0.01 / fm;           // m/s
  // take one particle's share out of the cell's gathering
  let share = f32(mp) / fm;
  atomicSub(&liq_dep[i], mp);
  atomicSub(&liq_dep[i + 1u], i32(round(f32(mh) * share)));
  atomicSub(&liq_dep[i + 2u], i32(round(f32(mv) * share)));
  atomicSub(&liq_dep[i + 3u], i32(round(f32(icy) * share)));
  let seed = u32(slot) * 2654435761u + u32(U.k.w) * 97u;
  let r = rand3(seed, 3u);
  let x = vec3<f32>(c) + vec3<f32>(r.x, 0.5 + 0.5 * r.y, r.z);
  var P = new_particle(x, vec3<f32>(0.0, 0.3 * min(vy, 0.0), 0.0));
  P.p.w = 1.0e-6;   // (not new to the heat model's sources: it brings its own heat)
  parts[u32(slot)] = P;
  if (U.k2.x > 0.5) { attr[u32(slot)] = vec4<u32>(0u); }   // clear water (a slot may hold a dead one's dye)
  if (U.k.z > 0.5) {
    // seeded (cloudy ice) when it brought ice, else supercoolable water
    therm[u32(slot)] = vec2<f32>(h, select(-1.0, 0.85, icy > 0));
  }
}
