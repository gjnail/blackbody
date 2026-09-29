// Workgroup reduction into `partials` (declared by the including kernel).

var<workgroup> red: array<f32, 256>;

fn wg_index(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
  return wid.x + nwg.x * (wid.y + nwg.y * wid.z);
}

fn reduce_store(li: u32, v: f32, slot: u32) {
  red[li] = v;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) { red[li] = red[li] + red[li + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (li == 0u) { partials[slot] = red[0]; }
}
