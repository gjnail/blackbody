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

## Fitting the GPU's memory

Blackbody reads how much memory the card has (DXGI on Windows, NVML or the driver's files on Linux, the shared memory on Apple silicon; `blackbody info` shows it, in GB as the system shows them: a 24 GB card is 24 GB). It plans to fit a scene within 80% of the card's memory, less 0.4 GB for what its estimate leaves out (shaders, meshes, fabric, grass, matter, Lume and the driver's own blocks): 18.8 GB on a 24 GB card, 9.2 GB on 12 GB, 6.0 GB on 8 GB. The plan comes from the card's size alone, never from what happens to be free as the app starts, because it shapes the simulation (its voxels and detail, and so its disk cache): a scene is laid out the same on every start, whatever else is open. `BLACKBODY_GPU_MEMORY` (in GB, taken as given: 2.5 is planned as 2.5 GB, not rounded as a card's own reading is) stands in for what the card says, to try a smaller card, to give Blackbody less when other programs need the card, or to give every machine of a farm the same plan.

Before it makes a scene's grids, it estimates what they take: the simulation's grids and particles, drawing them (the light volume, a liquid's surface grid, a cached frame's textures) and the output's passes (about 320 bytes a pixel, 0.6 GB at 1920×1080). Against what is really made it is within about a tenth (fire, liquid, fire-and-liquid and sky scenes on an RTX 3090). Rather than make a scene that does not fit and fail partway through it, Blackbody steps its *Detail upres* down first (fire), then its voxels, until it fits, and a notice says what was cut (the grids keep at least a quarter of the plan, however large the output). A grid larger than the GPU's largest 3-D texture (16,384 cells a side with Vulkan on an RTX 3090, 2,048 with Direct3D 12) is cut the same way, counting the detail upres grid. The *Voxels* setting's tooltip and the Render dialog show what a final render needs; the status bar shows what Blackbody uses now and what it fits a scene within.

If the GPU runs out of memory anyway (other programs holding much of the card, or a scene heavy with what the estimate leaves out), or stops responding (a driver reset or a time-out loses its device), Blackbody starts its engine again on its own (on a new device when the old one is gone) and carries on. After a lost device it keeps the frames it had cached. After running out of memory it plans within a quarter less until the app is closed, and the notices say what that cut: a disk cache keeps its frames and layout, but frames cached only in memory are simulated again at the smaller layout. Frames not cached are simulated again. A disk cache is not simulated again for it: a layout the cache holds is kept while it fits the card's own plan (or on a card whose memory is not known), with a notice that it needs more than the GPU has had room for since. If it fails three times in a row it stops and asks for a restart (after running out of memory: close other programs that use the GPU first, or delete the disk cache to simulate the scene within less).

A disk cache keeps the layout it was simulated at. It records the voxels and detail upres the scene was fitted to, and while the same scene still fits the GPU at those, they are kept, so its frames survive a start with another plan: a smaller card, another `BLACKBODY_GPU_MEMORY`, a plan cut after running out of memory (against which it is judged by the card's own plan instead). Only a cache with frames in it keeps its layout. A notice says when that is less than the scene asks for; delete the cache to simulate it as set. A cached layout that no longer fits is fitted again, and simulated again. Rendering from the cache (`blackbody render --from-cache`) always takes the cached layout, on any card (see [Command line and render farms](command-line.md)).

## Big and long simulations

### A box that grows

Domain › *Grow to fit* (fire scenes) starts the box small around the fire and grows it, keeping the cell size, wherever the smoke comes near an open side (*Grow up to* limits it, as a multiple of its size). Detail stays high near the fire, and no memory goes on empty air until the smoke needs it. It grows only as far as the GPU's memory (the scene's share of the plan, the same on every run) and its largest 3-D texture allow, with a notice when one of those stops it. Each growth copies the simulation into the larger box on the GPU, in 10 to 50 ms (0.06 to 0.5 s when it went through the computer's memory, as it still does when the GPU has no room for both boxes at once).

### Disk cache

Domain › *Disk cache* keeps every simulated frame on disk next to the project (`NAME.bbcache`), with the whole simulation saved every *Checkpoint every* frames: frames survive closing the app, and a stopped simulation resumes from its last checkpoint. The cache belongs to the settings it was simulated with, and to the version of the physics: changing the simulation, or updating to a Blackbody that simulates it differently, starts it afresh. It keeps the layout it was simulated at while that fits the GPU (see [Fitting the GPU's memory](#fitting-the-gpus-memory)). Render farms: see [Command line and render farms](command-line.md).

## Detail for final renders

### Detail upres

Render › *Detail upres* (2 or 3) carries the fire and smoke on a grid that many times finer for final renders, moved by the simulated air plus small swirls where the air is spinning, and burns the fuel there too. It gives much finer detail for a fraction of the cost of simulating at that resolution (*Upres turbulence* sets the swirls). *Upres in the viewer* shows it while you work too.

### Resolution from the shot

With Render › *Resolution from the shot* (on by default), final renders with Detail upres left at 1 pick the detail upres (up to 3, within about 2 GB of GPU memory for the grids, and within half the plan on a card of 5 GB or less) until its cells are about 1.5 pixels on screen where the fire is largest in the shot. The voxels stay as you set them: a finer simulation grid changes how the fire burns and costs about four times as much, while the upres grid only carries the fire and adds the fine detail. Checked against footage of a real campfire, a grid twice as fine made the Campfire's flames about a fifth shorter and their edges no sharper. A larger card does not raise the 2 GB: on a 24 GB card it would take Fireball, Hillside fire, Gas cloud and Shed fire from upres 2 to 3, and make their final frames take 1.6 to 3.1 times as long (Fireball's from 0.34 to 1.04 s on an RTX 3090). Set *Detail upres* to 3 yourself for that detail.

### Motion blur from the cache

Every cached frame keeps the air's velocity (at half precision), so frames shown from the cache and renders from the disk cache get the same motion blur as live ones.

## Shader compiles

Each GPU kernel is compiled by the graphics driver the first time it is used, and the driver keeps the result, so later runs start at once. Cold, on an RTX 3090 with a 16-thread CPU and NVIDIA's Vulkan driver, the liquid renderer's march (`liq_march.wgsl`) takes about 20 s on an otherwise idle machine and 40 to 55 s on a busy one, with 1.8 GB of memory while it compiles; the other 131 kernels of a liquid scene take about 10 s together. Warm, a liquid scene compiles in under a second. The march used to take 17 minutes and 22 GB: the driver inlines every call of a function, and the march called its largest pieces (the colliders' distance, what a ray finally lands on, the path through the liquid) from dozens of places. They are now called from one or two each, and a test (`tests/test_precompile.py`) keeps the inlined size under a budget. The smaller march also draws the same picture faster: a frame at 1280×720 takes 10 to 50% less on the liquid presets measured (the sea breaking on a beach 67 to 32 ms) and two thirds less on the tsunami (181 to 64 ms); lava, shaded in a pass of its own, is unchanged. The app compiles the march each time it opens, in a thread, on its own GPU device (a moment when the driver has it already; a separate process would not do, as a running process does not see what another compiled after it started). A scene that needs the march meanwhile waits for that compile and takes what it made, rather than starting a second one. The app's other GPU work goes on during it, with the odd pause of up to about 0.4 s. `blackbody precompile` does the same from the command line, for the processes started after it. With Direct3D 12 the march takes about 3 minutes on the same machine, every time a process starts (see [Troubleshooting](troubleshooting.md)): use Vulkan where the driver offers it.
