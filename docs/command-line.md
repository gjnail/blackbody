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
blackbody simulate shot.bbfire --cache //server/cache/shot
blackbody render shot.bbfire --from-cache //server/cache/shot --frames 1001-1050 -o fire.####.exr
blackbody presets
blackbody settings motion
blackbody info
```

- `####` is the frame number. The output type follows the extension: `.exr` `.png` `.vdb` `.mov` `.mp4` `.webm`; `.obj` or `.usd` (`.usdc`, `.usda`) for the liquid's surface and the fabric; a `.usd` name with `.scene.` in it (or `--content scene`) for the shot as a USD scene, and with `.camera.` (or `--content camera`) for its camera alone; `.chan` for the camera as Nuke reads it. See [Outputs](outputs.md).

- `--set section.key=value` overrides any setting; `emitter.N.key`, `collider.N.key`, `light.N.key` and `fabric.N.key` reach one emitter, collider, light or fabric (numbered from 0). `blackbody settings` lists every name with its default and range. Values are checked before anything renders: a typo stops the job instead of rendering with a default.

- `--draft` renders at interactive quality.

- `--usd FILE` brings the camera, objects (as colliders) and lights in from a USD scene before rendering.

- **Simulate once, render anywhere.** `blackbody simulate` simulates into a disk cache folder (by default `NAME.bbcache` next to the project), with a checkpoint every few frames; run it again after a crash or a stop and it resumes from the last checkpoint. `blackbody render --from-cache FOLDER` then renders frames from that cache without simulating, so several machines can render different frame ranges of one simulation. They only read the cache, and refuse one simulated with other settings instead of overwriting it. `render --cache FOLDER` simulates and fills the cache as it renders.

- Each render also writes a `<name>_render.json` sidecar with the full scene and settings, for reproducibility.

- **On a farm:** when the output goes to a log instead of a terminal, progress is written as whole lines (`Progress: 25.0%  Frame 1 of 4 …`, one per frame) in UTF-8. Exit codes: `0` done, `1` failed (the log ends with `Error: …`), `2` bad arguments, `130` cancelled.
