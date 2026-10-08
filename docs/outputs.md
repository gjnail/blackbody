# Outputs

One render pass can write several outputs at once: a finished composite, a fire element with alpha for your editor, multi-layer EXRs for compositing, deep EXRs, OpenVDB volumes for 3D packages, the liquid and the fabric as meshes, the shot as a USD scene (its camera, the objects as they fall and break, the sand, snow and mud, the grass and the embers), and the camera on its own for Nuke.

| Output | Use it in | Notes |
|---|---|---|
| **Fire element · OpenEXR** | Nuke, After Effects, Fusion, Resolve, Flame | Scene-linear, premultiplied RGBA. Layers: `emission` (fire only), `glow`, `heat` (drives distortion), `depth` (metres), `light` (fire light on the ground and colliders: multiply the plate by it and add), `holdout` (matte of the colliders in the shot and the footage holdouts), `scorch`, `soot` and `wet`; with lights in the set, `lamps` (multiply the plate by 1 + lamps). ZIP, PIZ, DWAA and more; half or float. |
| **Fire element · PNG** | Any editor or motion-graphics app | 8 or 16 bit with alpha. Premultiplied or straight. |
| **Fire element · ProRes 4444** | Premiere, Final Cut, Resolve, After Effects | One file with alpha. |
| **Composite over the footage** | Delivery or review | ProRes 422 HQ, H.264, H.265 (10-bit), DNxHR. The footage's audio comes along, trimmed to the rendered frames: copied untouched when the container can hold it, otherwise converted (24-bit PCM in `.mov`, AAC in `.mp4`). |
| **Composite · OpenEXR** | Nuke, Resolve, Flame (grading, finishing) | The finished shot, scene-linear RGB. With Lume's *Light passes* on, also the set's light split by where it comes from: `light_key`, `light_sky`, `light_fire`, `light_lamps` (add a layer times the gain − 1 to turn that light up or down; see [Lume › Light passes](lume.md#light-passes)). |
| **Fire element · deep OpenEXR** | Nuke (DeepRead), Houdini, Katana | Up to 8 samples per pixel, or 16 (*16 samples per pixel*, or `--deep-samples 16` on the command line: thick, layered smoke kept in more slices, for objects merged deep inside it, at twice the memory), each with its premultiplied colour and alpha and its front and back depth (`Z`, `ZBack`, metres along the view axis), so the fire and smoke merge correctly with other deep renders. All the anti-aliasing samples go into them, so they flatten to exactly the anti-aliased image, and the embers are in them as light at their own depth. Written as `name.deep.####.exr`. Fire scenes and liquid scenes have them (fire-and-liquid and sky scenes do not); in a shot with layers, every fire and liquid layer's samples go into the one file, each at its own depth. |
| **Volume · OpenVDB** | Blender, Houdini, Maya, Cinema 4D, Unreal | `density`, `temperature`, `flame`, `fuel` (float grids) and `vel` (vec3). Scenes with steam add `steam` (condensed water, g/m³) and `vapour`; tracked air adds `oxygen_used` (0–1); colourants add `color` (vec3); soot stains add `soot`. With *Detail upres* the grids are written at the finer voxel size, and `vel` is the velocity the finer grid moves with (the simulated air plus its small swirls). A sky scene writes its water in g/m³: `density` (the cloud itself, its water and ice), `cloud_water`, `cloud_ice`, `rain`, `snow` and `hail`, and `vel`. Metres, y up. Validated with OpenVDB 11. |
| **Scene · USD** | Houdini (Solaris), Blender, Maya, Nuke, Unreal, Omniverse | The shot's camera, every object as it is keyed, falls, floats and breaks (and its pieces), the ropes, the sand, snow, mud, jelly and clay as points, the grass as curves and the embers as points, frame by frame. Written as `name_scene.usdc` (on the command line, a name with `.scene.` in it, or `--content scene`). See [The shot as a USD scene](#the-shot-as-a-usd-scene). |
| **Camera · USD and Nuke .chan** | Nuke, Blender, Maya, Houdini, 3DEqualizer | The shot's camera alone, as a USD camera (`name_camera.usda`) and a `.chan` file (`name_camera.chan`). Nothing is simulated for it, so it is written in a moment. Every EXR carries it too, in its header. See [The camera](#the-camera). |

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

## The shot as a USD scene

*Scene · USD* writes what you need to light and render the shot in another package next to Blackbody's elements: the camera, and everything solid or grainy the simulation moved, in world metres with y up, one time code a frame at the shot's frame rate.

| Prim | What it holds |
|---|---|
| `/World/Camera` | The shot's camera (see [The camera](#the-camera)), and `/Render/Settings` with the picture's size. |
| `/World/Objects/<name>` | Every enabled object, moved frame by frame as it is keyed, falls, floats or is thrown: an Xform with its shape under it as `Shape` (a Cube, Sphere or Cylinder, or the mesh as modelled), scaled to its *Size*. Objects that hide nothing (*Hides fire* off) are guides, so they do not render. A broken object is hidden from the frame it breaks, and stays where it was until then. |
| `/World/Pieces/<name>/Piece_0001` … | A broken object's pieces (a person's or a car's parts too), each a mesh in its own frame, moved frame by frame, hidden until it breaks and once burnt to ash (where it was last seen). Their colour is `displayColor` per face, the cut faces in the material's inside colour, and the cut faces are the GeomSubset `cut`, for a material of their own. |
| `/World/Ropes/<name>` | Ropes, cables, chains and springs as curves, as wide as they are thick. |
| `/World/Matter` | Sand, snow, mud, jelly and clay as points: `ids` (the same grain keeps its id), `velocities` (m/s, for motion blur), `widths` (a sphere of the matter each grain stands for), `displayColor`, the primvar `material` (its index in the list `blackbody:materials`) and, when it heats, `temperature` (K). |
| `/World/Grass/<name>` | Every blade of a patch as a curve of five points, its width tapering to the tip, its colour as drawn (drier toward the tip, browned by heat, charred), and the primvars `burn` (0 unburnt, 0–1 burning, 1–2 the stubble cooling) and `burning`. |
| `/World/Embers` | Embers and sparks as points, with their velocities, sizes, the colour of their glow and the primvar `temperature` (K). |

Objects, pieces and the camera move by a translate and an orient (a quaternion), not a matrix: between frames, where motion blur samples them, a quaternion turns the short way round at a steady rate, so a piece spinning fast keeps its shape (a matrix interpolated entry by entry would shrink and shear it).

The points and curves are big (a million grains a frame), so they go into a file a frame beside the scene, `name_scene_points/name_scene.####.usdc`, which the scene reads as *value clips*: every program that reads USD sees animated points, and neither Blackbody nor the program holds the whole shot in memory. Keep the folder beside the scene when you move it.

Open it as any USD file: in Houdini as a Solaris *Sublayer* (or *File › Import › USD*), in Blender with *File › Import › Universal Scene Description*, in Nuke with its USD geometry reader. The files are checked by reading them back with Pixar's own USD library, not in each package.

## The camera

The camera lines up exactly with Blackbody's picture: rendered over the elements in another package, a CG object sits where the fire and the footage say it is. Blackbody's 2D placement (*Anchor*, *Scale* and *Roll*, which put the effect's base on a spot in the frame) is written as a real camera holds it: the roll as a roll of the camera about its view axis, the scale in the focal length, and the slide as the film back's offset (USD's `horizontalApertureOffset` and `verticalApertureOffset`).

- **USD** (`name_camera.usda`, and `/World/Camera` in a USD scene): position, rotation, focal length, sensor width and height (mm), film offset and clipping range on every frame. Blackbody's own USD import reads it back as the same camera, all but the film offset. With Composite › *Lens distortion* set, the camera also carries it as `blackbody:lens_k1` (below).
- **.chan** (`name_camera.chan`): one line a frame of position (m), rotation (degrees, Nuke's default order ZXY) and vertical field of view, as Nuke's *Camera › Import chan file* reads it. A .chan has no film offset, so where the *Anchor* slides the picture off the lens's centre it leaves that out; the Render window and the command line say by how many pixels before it renders. Set the Nuke camera's *window translate* to match, or use the USD camera.
- **EXR headers**: every EXR (element, composite and deep) carries the frame's camera as OpenEXR's standard `worldToCamera` and `worldToNDC` (4×4, a point times the matrix, as Imath has it). The camera's own frame looks down −z with +y up; `worldToNDC` is exactly the render's, its x and y over its w the picture's −1 to 1, y up.

A camera lined up with the footage is the same in every layer of a shot. When a layer places its effect with an *Anchor* of its own, it sees the shot through a camera of its own: a USD scene writes it as `/World/<layer>/Camera`, the .chan and the EXR headers hold the base layer's, and the Render window says which layers have their own.

## Shots with layers

Every output covers every layer of the shot:

- **Elements, composites and deep EXRs** are the layers merged, as the viewer shows them. A deep EXR holds every fire and liquid layer's samples, each at its own depth; the other layers are not in it (it says which before it renders).
- **VDBs and meshes**: the base layer's go to the path you gave and each other layer's beside it, named for the layer: `name.<layer>.####.vdb`, `name_liquid.<layer>.usdc`. A layer with nothing for an output (no liquid for a liquid mesh) has no file.
- **A USD scene**: the base layer's under `/World`, each other layer's under `/World/<layer>`.

## Limits

- Deep EXRs keep at most 16 samples a pixel: the march gathers them in a fixed set of bins, and their buffer must fit in one GPU buffer binding (2 GB on an RTX 3090), so there a 4K frame holds at most 8 a pixel. Fire-and-liquid and sky scenes have no deep EXRs (the Render window greys the box out and the command line refuses one).
- A fire-and-liquid scene's EXR has the fire's layers, not the liquid's `water`, `foam`, `spray`, `drops` and `bubbles` mattes.
- The USD scene: a hollow object, or one with an opening cut in it, is written as its outer shape until it breaks (its pieces are exact). What bullets throw up (debris, sparks, tracers) and the holes they leave are not in it; the liquid and the fabric have their own outputs. Matter is its grains as points, not a surface or a VDB. Velocities of matter rendered from the disk cache come from where each grain was a frame before (the cache keeps none), so the first frame of such a render has none.
- A .chan cannot hold the *Anchor*'s slide (above). The note before the render looks at the first, middle and last frames; if a keyed or tracked *Anchor* slides the picture further on other frames, the command line says so again after the render.
- No camera holds Composite › *Lens distortion*, which bends the element (and the composite) toward the frame's edges: CG rendered through the exported camera, or projected with the EXRs' `worldToNDC`, lines up at the centre of the frame but less and less toward the edges. The Render window and the command line say so before rendering. Distort the CG the same way in the comp: x_d = x_u (1 + k1 r²), with r measured from the centre of the frame in half-diagonals and k1 the *Lens distortion* (kept on the USD camera as `blackbody:lens_k1`).
- The USD camera's focal length and sensor are written in millimetres, as Blackbody shows them and its own USD import reads them. USD's schema has them in tenths of a scene unit, and the scene is in metres, so an importer that follows the schema to the letter (Blender's, for one) shows a lens and a sensor 100 times larger: the framing is the same (it depends on their ratio), but set the lens back before using depth of field there.
