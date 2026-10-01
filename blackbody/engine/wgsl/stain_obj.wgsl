// Soot on colliders. Every collider owns a region of the soot atlas laid out in its own frame and in
// its own size (coordinates divided by col_scale), so soot laid down on an object stays on it when it
// moves or turns, and grows and shrinks with it when its size is animated. Each cell just outside a
// collider's surface samples the smoke where the cell is now and gathers soot from it, heaviest where
// the smoke is thick and hot. The floor and the box's closed sides keep their soot in the simulation
// grid (react.wgsl).
//!include common.wgsl
//!include meshsdf.wgsl
//!include colliders.wgsl

// A collider's region of the soot atlas, in size-relative coordinates (see col_scale).
struct SootSlot {
  lo: vec4<f32>,    // the region's corner; w = z offset in the atlas
  dims: vec4<f32>,  // the region's size (cells)
  cell: vec4<f32>,  // a cell's size along each axis
};

struct Params {
  g: Grid,
  st: vec4<f32>,    // soot rate (1/s), regions (colliders), _, _
  ccnt: vec4<f32>,
  col: array<Collider, MAX_COLLIDERS>,
};

@group(0) @binding(0) var scal: texture_3d<f32>;
@group(0) @binding(1) var sdf: texture_3d<f32>;
@group(0) @binding(2) var atlas: texture_3d<f32>;
@group(0) @binding(3) var stain: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var<storage, read> slots: array<SootSlot>;   // one region per collider, in collider order
@group(0) @binding(5) var lin: sampler;
@group(1) @binding(0) var<uniform> U: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let dims = vec3<i32>(textureDimensions(stain));
  if (any(c >= dims)) { return; }
  let regions = min(i32(U.st.y), i32(U.ccnt.x));
  for (var i = 0; i < regions; i++) {
    let s = slots[i];
    let z0 = i32(s.lo.w);
    if (c.z < z0 || c.z >= z0 + i32(s.dims.z) || c.x >= i32(s.dims.x) || c.y >= i32(s.dims.y)) { continue; }
    let k = U.col[i];
    let sc = col_scale(k);
    let cell = max(s.cell.x * sc.x, max(s.cell.y * sc.y, s.cell.z * sc.z));   // metres, the longest side
    let q = (s.lo.xyz + (vec3<f32>(f32(c.x), f32(c.y), f32(c.z - z0)) + 0.5) * s.cell.xyz) * sc;
    let w = col_to_world(k, q);
    let d = col_sdf(k, w) / cell;
    if (d < 0.0 || d >= 1.5) { return; }
    let h = U.g.n.w;
    let n = U.g.n.xyz;
    let pg = (w - U.g.org.xyz) / h;
    if (any(pg < vec3<f32>(0.0)) || any(pg > n)) { return; }
    let g = samp_c(scal, lin, pg, n);
    let add = U.st.x * max(g.z, 0.0) * (1.0 + max(g.x, 0.0)) * U.g.bc.w;
    if (add > 0.0) { textureStore(stain, c, vec4<f32>(textureLoad(stain, c).x + add, 0.0, 0.0, 0.0)); }
    return;
  }
}
