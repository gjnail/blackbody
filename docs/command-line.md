# Command line and render farms

Render, simulate into a shared cache and render frame ranges on many machines, wedge settings, and check a scene's settings, all without the app.

```bash
blackbody render shot.bbfire -o renders/fire.####.exr
blackbody render shot.bbfire -o comp.mov --content composite --format prores422hq
blackbody render --preset campfire --footage plate.mov -o comp.mp4 --content composite
blackbody render shot.bbfire -o vdb/fire.####.vdb --frames 1001-1100 --res 256
blackbody render shot.bbfire -o fire.####.exr --set motion.wind_speed=3 --set shading.exposure=-0.5
blackbody render shot.bbfire -o wedge/fuel20.####.exr --set emitter.0.fuel=20
blackbody render shot.bbfire -o deep/fire.deep.####.exr
blackbody render shot.bbfire -o usd/shot.scene.usdc -o cam/shot.chan
blackbody render --preset campfire --usd shot.usd -o fire.####.exr
blackbody render --preset campfire --usd shot.usd --usd-offset 1000 -o fire.####.exr
blackbody simulate shot.bbfire --cache //server/cache/shot
blackbody render shot.bbfire --from-cache //server/cache/shot --frames 1001-1050 -o fire.####.exr
blackbody presets
blackbody settings motion
blackbody info
blackbody precompile
```

- `####` is the frame number. The output type follows the extension: `.exr` `.png` `.vdb` `.mov` `.mp4` `.webm`; `.obj` or `.usd` (`.usdc`, `.usda`) for the liquid's surface and the fabric; a `.usd` name with `.scene.` in it (or `--content scene`) for the shot as a USD scene, and with `.camera.` (or `--content camera`) for its camera alone; `.chan` for the camera as Nuke reads it. See [Outputs](outputs.md).

- `--set section.key=value` overrides any setting; `emitter.N.key`, `collider.N.key`, `light.N.key` and `fabric.N.key` reach one emitter, collider, light or fabric (numbered from 0). `blackbody settings` lists every name with its default and range. Values are checked before anything renders: a typo stops the job instead of rendering with a default.

- `--draft` renders at interactive quality.

- EXRs carry the compositing passes (motion vectors, normals and positions, object mattes, Cryptomatte: see [Outputs](outputs.md#compositing-passes)); `--no-passes` leaves them out, saving about two seconds a full-HD EXR frame (more in sets with many pieces). With `--exr-compression dwaa` or `dwab` an EXR that has them is written with ZIP: DWA is lossy and would garble the Cryptomatte.

- `--usd FILE` brings the camera, objects (as colliders) and lights in from a USD scene before rendering, and takes its frame range and frame rate, as File › Import USD scene does (with `--footage`, or a project with footage, only where the shot starts: the footage keeps its length and rate). `--usd-offset N` turns USD frame F into frame F − N, so `--usd-offset 1000` renders a 1001–1100 shot as frames 1–100; the camera and objects stay in step. `--usd-keep-range` keeps the project's or preset's own range and rate.

- **Simulate once, render anywhere.** `blackbody simulate` simulates into a disk cache folder (by default `NAME.bbcache` next to the project), with a checkpoint every few frames; run it again after a crash or a stop and it resumes from the last checkpoint. `blackbody render --from-cache FOLDER` then renders frames from that cache without simulating, so several machines can render different frame ranges of one simulation. They only read the cache, and refuse one simulated with other settings instead of overwriting it. `render --cache FOLDER` simulates and fills the cache as it renders.

- **GPU memory on a farm.** A scene is fitted to the GPU's memory before it is simulated (see [Performance and memory](performance.md#fitting-the-gpus-memory)): on a card without room for it as set, its *Detail upres* and then its voxels are stepped down, and the job's log says so. The plan comes from the card's size, so every machine with the same card lays a scene out the same, and the cache records the layout it was simulated at: `render --from-cache` renders that layout on any card (a machine with less memory may run out on a large one), and `simulate` resuming a cache keeps it while it fits. Set `BLACKBODY_GPU_MEMORY` to the same number of GB on every machine that simulates (the smallest card's, or less) so they all fit a scene alike, whatever their cards; `blackbody info` shows what a machine reads and the plan it makes.

- Each render also writes a `<name>_render.json` sidecar with the full scene and settings, for reproducibility.

- `blackbody precompile` compiles the slow GPU shaders now (the liquid renderer's, 20 to 55 s on an RTX 3090 with Vulkan, and any shader this machine has used that took more than 2 s and has changed since), so the first liquid render does not wait for them. Run it after installing or updating Blackbody or the graphics driver, on each farm machine (the driver keeps what it compiles, per machine). The app does the same in the background when it opens.

- **On a farm:** when the output goes to a log instead of a terminal, progress is written as whole lines (`Progress: 25.0%  Frame 1 of 4 …`, one per frame) in UTF-8. Exit codes: `0` done, `1` failed (the log ends with `Error: …`), `2` bad arguments, `130` cancelled.
