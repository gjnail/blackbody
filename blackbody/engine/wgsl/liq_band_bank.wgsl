// Narrow band bookkeeping, at the start of each step's band passes. The bank holds (i32):
//   0  the band's running balance, in particles: +1 for every particle a band pass creates, -1 for
//      every one it frees, +/- particles-per-cell for every cell that joins or leaves the deep
//      liquid (which counts as full). Exact conservation keeps it at 0.
//   1  the balance as this step's passes start (every cell of the emission reads the same value)
//   2  the outflow of the deep liquid this step, in particles (fixed point, x 256)
//   3  last step's outflow
// The emission (liq_band_emit.wgsl) pays half the balance back each step, spread over the band in
// proportion to each cell's share of the outflow, so the small biases of the exchange (a band a few
// percent denser than rest next to the deep liquid, the volume correction pushing particles into it)
// cannot add up while the whole flow of the box passes through the deep region.

@group(0) @binding(0) var<storage, read_write> bank: array<atomic<i32>, 8>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  atomicStore(&bank[1], atomicLoad(&bank[0]));
  atomicStore(&bank[3], atomicLoad(&bank[2]));
  atomicStore(&bank[2], 0);
}
