# Outputs

One render pass can write several outputs at once: a finished composite, a fire element with alpha for your editor, multi-layer EXRs for compositing, deep EXRs, OpenVDB volumes for 3D packages, and the liquid and the fabric as meshes.

One render pass can write several outputs at once.

| Output | Use it in | Notes |
|---|---|---|
| **Fire element · OpenEXR** | Nuke, After Effects, Fusion, Resolve, Flame | Scene-linear, premultiplied RGBA. Layers: `emission` (fire only), `glow`, `heat` (drives distortion), `depth` (metres), `light` (fire light on the ground and colliders: multiply the plate by it and add), `holdout` (matte of the colliders in the shot and the footage holdouts), `scorch`, `soot` and `wet`; with lights in the set, `lamps` (multiply the plate by 1 + lamps). ZIP, PIZ, DWAA and more; half or float. |
| **Fire element · PNG** | Any editor or motion-graphics app | 8 or 16 bit with alpha. Premultiplied or straight. |
| **Fire element · ProRes 4444** | Premiere, Final Cut, Resolve, After Effects | One file with alpha. |
| **Composite over the footage** | Delivery or review | ProRes 422 HQ, H.264, H.265 (10-bit), DNxHR. The footage's audio comes along, trimmed to the rendered frames: copied untouched when the container can hold it, otherwise converted (24-bit PCM in `.mov`, AAC in `.mp4`). |
| **Composite · OpenEXR** | Nuke, Resolve, Flame (grading, finishing) | The finished shot, scene-linear RGB. With Lume's *Light passes* on, also the set's light split by where it comes from: `light_key`, `light_sky`, `light_fire`, `light_lamps` (add a layer times the gain − 1 to turn that light up or down; see [Lume › Light passes](lume.md#light-passes)). |
| **Fire element · deep OpenEXR** | Nuke (DeepRead), Houdini, Katana | Up to 8 samples per pixel, or 16 (*16 samples per pixel*, or `--deep-samples 16` on the command line: thick, layered smoke kept in more slices, for objects merged deep inside it, at twice the memory), each with its premultiplied colour and alpha and its front and back depth (`Z`, `ZBack`, metres along the view axis), so the fire and smoke merge correctly with other deep renders. All the anti-aliasing samples go into them, so they flatten to exactly the anti-aliased image, and the embers are in them as light at their own depth. Written as `name.deep.####.exr`. Fire scenes and liquid scenes have them (fire-and-liquid and sky scenes do not); in a shot with layers, every fire and liquid layer's samples go into the one file, each at its own depth. |
| **Volume · OpenVDB** | Blender, Houdini, Maya, Cinema 4D, Unreal | `density`, `temperature`, `flame`, `fuel` (float grids) and `vel` (vec3). Scenes with steam add `steam` (condensed water, g/m³) and `vapour`; tracked air adds `oxygen_used` (0–1); colourants add `color` (vec3); soot stains add `soot`. With *Detail upres* the grids are written at the finer voxel size, and `vel` is the velocity the finer grid moves with (the simulated air plus its small swirls). A sky scene writes its water in g/m³: `density` (the cloud itself, its water and ice), `cloud_water`, `cloud_ice`, `rain`, `snow` and `hail`, and `vel`. Metres, y up. Validated with OpenVDB 11. |

**Premultiplied or straight alpha.** Fire is light, so much of it has little alpha.

- **Premultiplied (recommended):** keeps that exactly. Interpret the footage as *premultiplied* (After Effects: *Premultiplied – Matted With Color: black*; Resolve: *Alpha mode › Premultiplied*).

- **Straight:** folds the light into alpha, so the element drops onto footage with a plain *Normal* blend in any editor.

**Heat distortion in comp.** Use the `heat` layer as the amount for a displacement:

- Nuke: *IDistort* with a noise, or *STMap*.

- After Effects: *Displacement Map* with a fractal noise.

- Inside Blackbody's own composite, the haze is already applied.

**Liquid outputs.** A liquid element carries the footage as seen *through* the liquid, with alpha 1 where there is liquid, so it drops straight onto the same footage. Its EXR `emission` layer holds the sun glints and a molten liquid's glow (for extra glow in comp), `ground` multiplies the footage outside the liquid (wet and shadowed ground and caustics; use it as a multiply), `depth` is in metres and `speed` in m/s, and the mattes `water` (the liquid surface), `foam`, `spray` (the mist), `drops` (spray droplets) and `bubbles` let you grade each part on its own. A fire-and-liquid scene writes the fire's layers, and its VDB output writes the liquid alongside as `name.liquid.####.vdb`. The VDB of a liquid holds `density` (1 inside, 0 outside, the 0.5 iso-surface is the liquid surface: mesh it with Blender's *Volume to Mesh* or Houdini's *Convert VDB*), `vel` (m/s) and `spray`, `foam` and `bubbles` densities, on the render surface grid.

**VDBs from a disk cache.** A render from a disk cache (`blackbody render --from-cache`) writes each frame's VDB from the cache. The cache keeps the simulated air's velocity only when motion blur was on as it was simulated, and not the finer grid's swirls, so there `vel` is the simulated air's (empty without motion blur), and a sky's VDB has its water but no `vel`.

**A sky as a VDB.** A sky's VDB is placed and sized as the scene shows it: a voxel is a cell of the scene's box, so with *Atmosphere › Scale* at 1000 a metre in the file is a kilometre of sky (the scale is in the file as `blackbody_sky_metres_per_unit`). Its `vel` is in the file's metres per second of the shot (the wind, times the time-lapse, over the scale), so it moves the cloud as it moves in the shot, for motion blur.

**The liquid as a mesh.** *Liquid surface · USD* (or an output path ending in `.usd`, `.usda`, `.usdc` or `.obj` on the command line) writes the liquid's surface as a closed mesh in world metres, y up: one USD file for the whole shot, with time-sampled points, normals and per-point velocities (for motion blur) at `/World/Liquid/Surface`, and the spray, foam and bubbles as points at `/World/Liquid/Spray`, `Foam` and `Bubbles`; or one OBJ per frame. Render it in any renderer, or use it to cast the liquid's shadows and reflections in a CG set.

**The fabric as a mesh.** *Fabric · USD* (or a path ending in `.usd`, `.usda`, `.usdc` or `.obj` on the command line, in a scene without liquid) writes each fabric as a mesh in world metres, y up, as it burns: one USD file for the shot (`name_fabric.usdc`), with time-sampled points, normals and velocities at `/World/Fabric/<name>`, the burnt-through holes left out, its weave as UVs (`st`) and a `burn` value on every point (0 untouched to 1 burnt through) for shading the char; or one OBJ per frame. In a liquid scene, a command-line mesh output writes the liquid's surface, and the fabric beside it as `name.fabric.usd` (or `name.fabric.####.obj`).

**Deep liquid.** The deep OpenEXR output works for liquids too: the water surface and the spray mist in front of it as separate samples at their own depths.

**VDB in Blender.** Blender is z-up, so rotate the imported volume 90° about X. In a *Principled Volume*, use `density` for density and `flame` (times a strength) for emission with a *Blackbody* colour of about 1700 K. The file metadata records the Kelvin mapping: `blackbody_temperature_1_kelvin`.
