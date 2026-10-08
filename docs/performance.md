# Performance and memory

How resolution, memory and speed trade off, and the tools for big and long simulations: the growing box, the disk cache and detail upres.

Resolution is set by *Domain › Voxels (longest side)*. Memory grows with the cube of it, and so does detail:

| Voxels (longest side) | Campfire grid | GPU memory | Simulate + preview on an RTX 3090 |
|---|---|---|---|
| 96 | 56×96×56 | 0.03 GB | 59 fps |
| 144 | 88×144×88 | 0.12 GB | 33 fps |
| 192 | 120×192×120 | 0.31 GB | 16 fps |
| 256 | 160×256×160 | 0.72 GB | 8 fps |
| 384 | 232×384×232 | 2.3 GB | 1.6 fps (final renders) |

(Measured with `python tools/benchmark.py`: campfire, 960×540 preview, adaptive substeps.)

Writing a VDB frame (all five grids, ZIP) takes about 1 s at 256 voxels on a 16-thread CPU; files are 50–80 MB at that size.

Tracked air or steam, colourants and spreading fire each keep one more field, about 16 bytes per voxel (0.1 GB at 256 voxels). A scene only pays for the ones it uses. A mesh is converted to a distance field once and kept in a disk cache (`%LOCALAPPDATA%\Blackbody\meshes` on Windows, `~/.cache/blackbody/meshes` elsewhere; `BLACKBODY_CACHE` moves it). Dense meshes use a banded conversion: a 58,000-triangle mesh takes about 0.05 s on an RTX 3090. *Detail upres* keeps about 28 bytes per fine voxel, so 2× costs about 0.25 GB for a 144-voxel campfire. `python tools/benchmark.py --features` measures what each feature costs on your GPU.

Liquids cost more per voxel than fire (particles, a second grid for the surface). The *Rock into a pond* preset (a 2 m pond, 24 cm deep), simulated and rendered at 960×540 on an RTX 3090:

| Voxels (longest side) | Grid | Particles | GPU memory | Simulate | Render |
|---|---|---|---|---|---|
| 128 | 128×56×128 | 2.0 M | 1.6 GB | 43 ms/frame | 28 ms |
| 192 | 192×88×192 | 6.8 M | 3.3 GB | 0.26 s/frame | 86 ms |
| 256 | 256×112×256 | 16 M | 4.7 GB | 0.41 s/frame | 0.20 s |

At 256 voxels this pond reaches the default *Particle limit* (16 M): the viewer says so, and sources hold back. Raise the limit (Liquid, advanced) if the GPU has the memory, about 70 bytes a particle (48 for the particle itself). Or turn on *Narrow band*: the same pond needs a fifth to an eighth of the particles.

While you work, the simulation runs at *Interactive resolution* (75% by default) and the viewer at *Preview* resolution. Idle frames refine to full quality automatically. Final renders use the full resolution times *Render › Final resolution*, with *Samples per pixel* and velocity-based motion blur.

## Big and long simulations

### A box that grows

Domain › *Grow to fit* starts the box small around the fire and grows it, keeping the cell size, wherever the smoke comes near an open side (*Grow up to* limits it, as a multiple of its size). Detail stays high near the fire, and no memory goes on empty air until the smoke needs it.

### Disk cache

Domain › *Disk cache* keeps every simulated frame on disk next to the project (`NAME.bbcache`), with the whole simulation saved every *Checkpoint every* frames: frames survive closing the app, and a stopped simulation resumes from its last checkpoint. The cache belongs to the settings it was simulated with, and to the version of the physics: changing the simulation, or updating to a Blackbody that simulates it differently, starts it afresh. Render farms: see [Command line and render farms](command-line.md).

## Detail for final renders

### Detail upres

Render › *Detail upres* (2 or 3) carries the fire and smoke on a grid that many times finer for final renders, moved by the simulated air plus small swirls where the air is spinning, and burns the fuel there too. It gives much finer detail for a fraction of the cost of simulating at that resolution (*Upres turbulence* sets the swirls). *Upres in the viewer* shows it while you work too.

### Resolution from the shot

With Render › *Resolution from the shot* (on by default), final renders with Detail upres left at 1 pick the detail upres (up to 3, within about 2 GB of GPU memory) until its cells are about 1.5 pixels on screen where the fire is largest in the shot. The voxels stay as you set them: a finer simulation grid changes how the fire burns and costs about four times as much, while the upres grid only carries the fire and adds the fine detail. Checked against footage of a real campfire, a grid twice as fine made the Campfire's flames about a fifth shorter and their edges no sharper.

### Motion blur from the cache

Every cached frame keeps the air's velocity (at half precision), so frames shown from the cache and renders from the disk cache get the same motion blur as live ones.
