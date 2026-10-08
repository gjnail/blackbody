// Counts per block turned into where each block starts (an exclusive prefix sum), in one workgroup: the middle of the
// three passes that hand out places in a fixed order, whatever order the GPU runs its threads in. A kernel counts what
// each block of its items has (free slots, roots of pieces of ice) into cnt; this writes where each block's first goes
// into off[0..n) and the total after them (off[n]); the next kernel gives an item the place its block starts at plus how
// many of the block's come before it. (Atomic appends hand places out in the order the threads happen to run, which
// differs from run to run: weather.py, liquid_thermal.py.)
//
// With a block size given, a block counts what it holds that is not in cnt (live particles, from the free slots
// counted: min(size, items left) - cnt).

struct Params {
  k: vec4<f32>,   // blocks, block size (0: cnt as it is), items, _
};

@group(0) @binding(0) var<storage, read> cnt: array<vec4<u32>>;   // (four blocks a load: the loads are what is slow)
@group(0) @binding(1) var<storage, read_write> off: array<u32>;
@group(1) @binding(0) var<uniform> U: Params;

var<workgroup> part: array<u32, 256>;

fn counts(j: u32, n: u32) -> vec4<u32> {
  var c = cnt[j];
  if (U.k.y > 0.0) {
    let size = u32(U.k.y);
    let items = u32(U.k.z);
    var w = vec4<u32>(0u);
    for (var e = 0u; e < 4u; e++) {
      let s = (4u * j + e) * size;
      if (s < items) { w[e] = min(size, items - s); }
    }
    c = w - min(c, w);
  }
  // (past the last block: nothing)
  for (var e = 0u; e < 4u; e++) {
    if (4u * j + e >= n) { c[e] = 0u; }
  }
  return c;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) t: u32) {
  let n = u32(U.k.x);
  let nq = (n + 3u) / 4u;
  // each thread a run of blocks in order: its total, then the running totals over the threads (Hillis-Steele)
  let per = (nq + 255u) / 256u;
  let a = min(t * per, nq);
  let b = min(a + per, nq);
  var s = 0u;
  for (var j = a; j < b; j++) {
    let c = counts(j, n);
    s += c.x + c.y + c.z + c.w;
  }
  part[t] = s;
  workgroupBarrier();
  for (var d = 1u; d < 256u; d <<= 1u) {
    var v = part[t];
    if (t >= d) { v += part[t - d]; }
    workgroupBarrier();
    part[t] = v;
    workgroupBarrier();
  }
  var run = 0u;
  if (t > 0u) { run = part[t - 1u]; }
  for (var j = a; j < b; j++) {
    let c = counts(j, n);
    for (var e = 0u; e < 4u; e++) {
      if (4u * j + e < n) { off[4u * j + e] = run; }
      run += c[e];
    }
  }
  if (t == 255u) { off[n] = part[255]; }
}
