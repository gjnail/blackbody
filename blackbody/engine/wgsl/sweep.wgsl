// What a moving collider runs into, pushed out of its way. A cell counts as solid once its centre is
// inside a collider, and the reaction then empties it; the gas the collider pushes moves on a little
// slower than its surface (a cell's velocity is the mean of its faces'), so a collider driven through
// smoke caught up with the smoke in front of it, a cell at a time, and deleted it. Run after the fields
// are carried and before the reaction, whenever the colliders' distance has been rewritten: every cell
// a collider has just covered (outside it a step ago, inside it now) hands what it holds (smoke, fuel,
// heat, and the other carried fields) to the free cell just outside the surface nearest it, so the gas
// is shoved ahead and pressed together, and flows round the collider from there.
// Each covered cell has one such cell, found the same way by every cell that looks for it, so nothing is
// handed on twice; a free cell gathers from the covered cells within the reach around it. Covered cells
// deeper than the reach (a collider moving more than about that far in one step) are still emptied, and
// so are cells a surface has crept over by less than a small share of a cell (broken pieces and sand
// lying still, baked again every step, jitter across a cell centre now and then: no motion).
//!include common.wgsl

struct Params {
  g: Grid,
  a: vec4<f32>,  // x = reach (cells), y = how far (cells) a surface has to have moved over a cell to cover it
};

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var sdf_was: texture_3d<f32>;   // the colliders' distance a step ago (cells; negative inside)
@group(0) @binding(2) var sdf: texture_3d<f32>;       // and now
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(0) var<uniform> U: Params;

fn sd(c: vec3<i32>) -> f32 { return textureLoad(sdf, clamp(c, vec3<i32>(0), gdim(U.g) - vec3<i32>(1)), 0).x; }

fn covered(q: vec3<i32>) -> bool {
  let was = textureLoad(sdf_was, q, 0).x;
  let now = sd(q);
  return now < 0.0 && was >= 0.0 && was - now >= U.a.y;
}

// The free cell just outside the surface nearest covered cell q, along the distance's gradient there; q
// itself if there is none.
fn outside_of(q: vec3<i32>) -> vec3<i32> {
  let s = sd(q);
  let g = vec3<f32>(sd(q + vec3<i32>(1, 0, 0)) - sd(q - vec3<i32>(1, 0, 0)),
                    sd(q + vec3<i32>(0, 1, 0)) - sd(q - vec3<i32>(0, 1, 0)),
                    sd(q + vec3<i32>(0, 0, 1)) - sd(q - vec3<i32>(0, 0, 1)));
  if (dot(g, g) < 1e-12) { return q; }
  let nrm = normalize(g);
  let d = gdim(U.g);
  for (var j = 0; j < 3; j++) {
    // (half a cell past the surface, then further where that cell's centre is still inside)
    let e = vec3<i32>(floor(vec3<f32>(q) + vec3<f32>(0.5) + nrm * (0.5 - s + 0.5 * f32(j))));
    if (in_grid(e, d) && textureLoad(sdf, e, 0).x >= 0.0) { return e; }
  }
  return q;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = gdim(U.g);
  let c = vec3<i32>(id);
  if (any(c >= d)) { return; }
  var v = textureLoad(src, c, 0);
  let s = textureLoad(sdf, c, 0).x;
  let r = i32(U.a.x);
  // (only free cells near a surface whose distance has changed: next to a still collider nothing is covered)
  if (s < 0.0 || s >= f32(r) + 1.0 || s == textureLoad(sdf_was, c, 0).x) {
    textureStore(dst, c, v);
    return;
  }
  for (var z = -r; z <= r; z++) {
    for (var y = -r; y <= r; y++) {
      for (var x = -r; x <= r; x++) {
        let q = c + vec3<i32>(x, y, z);
        if (!in_grid(q, d) || !covered(q)) { continue; }
        if (all(outside_of(q) == c)) { v += textureLoad(src, q, 0); }
      }
    }
  }
  textureStore(dst, c, v);
}
