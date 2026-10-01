# Moving shots and scene import

Bring the shot's camera, objects and lights in from your 3D package or tracker, follow a handheld camera with the 2D tracker, and use volumes from other programs as smoke.

## Moving shots

- **Handheld or moving camera, no 3D solve:** park the fire where it should be, then Tracking › Track the fire base (Ctrl+T). The fire follows that point through the shot. Dragging the ring afterwards offsets the whole track. Pick a spot with texture or a corner; the tracker follows position only, not rotation, scale or perspective.

![Tracking › Track the fire base (Ctrl+T) follows a point through a handheld pan; the fire stays put on the ground](media/gif/ui-track.gif "Tracking › Track the fire base (Ctrl+T) follows a point through a handheld pan; the fire stays put on the ground.")

- **Matchmoved shots:** File › Import camera track (.chan). This covers exports from Nuke, Blender, SynthEyes, 3DEqualizer and PFTrack. The camera switches to Free mode. Place the fire in the tracked scene with Camera › Fire position (metres, y up).

- **Keyframes:** click the ◆ next to a setting to key it on the current frame. Changing an animated setting keys it automatically. Keyframe *Master fuel* (Combustion) to ignite, grow or extinguish a fire.

## Scene import (USD)

**File › Import USD scene** brings the camera, objects and lights in from a USD file (`.usd`, `.usda`, `.usdc`, `.usdz`; Blender, Houdini, Maya, Cinema 4D and most trackers export it).

- **Camera:** Blackbody's free camera, keyed on every frame (position, rotation, focal length and sensor width).

- **Objects** become colliders (or emitters): polygon meshes, meshes inside instances (instanceable references), skinned characters (the skeleton's animation is baked into the mesh), point instancers (every instance of every prototype, as one object, with instancers inside instancers expanded too), Cube, Sphere, Cylinder, Capsule and Cone prims, curves, points and volumes. A cube, a sphere or an upright cylinder becomes the exact collider shape; other shapes become meshes.
  - Curves (basis curves: linear, Bézier, B-spline and Catmull-Rom; NURBS and Hermite curves through their points) become tubes as thick as their *widths* (2 cm without widths): cables, ropes, hair, branches.
  - Points become small solid blobs as wide as their *widths*: gravel, debris, a particle cache.
  - Volumes become smoke: a Volume emitter that fills the box with the volume's smoke (or keeps it topped up, when the volume's file is animated). With *Volumes become › Solid objects* in the import dialog they become the solid their field describes instead: a fog volume's thick part (above a quarter of its densest values) or a level set's inside, cell by cell, at up to 160 cells a side. The VDB is read by Blackbody itself (float grids, ZIP or Blosc LZ4 compression, half or full floats); a Zstd-compressed VDB has to be saved again with ZIP or LZ4.

  An object that stays still or moves without changing shape keeps its shape and is placed and keyed by position, rotation and size; one that deforms, or tumbles, is baked frame by frame.

- **Lights:** the first distant light becomes the key light (direction, colour, colour temperature and intensity; a lux value such as USD's default 50 000 maps to a *Key intensity* of 3), the first dome light becomes the ambient light, and its texture the environment image. Sphere, disk, rectangle and cylinder lights become lights in the set: point lights, area lights, or spot lights when their shaping cone is narrower than a hemisphere, with the intensity their surface gives off. Portal and geometry lights, and any distant and dome lights after the first, are reported, not imported.

- Invisible prims and guides are left out. Units and the up axis are converted (the stage's metres per unit; Z-up stages are turned y-up), and objects land where they are in the USD scene relative to Camera › *Fire position*. A single object can also be picked from a USD file as any mesh (`shot.usd#/World/Car`).

Alembic (`.abc`) has no reader for Python on Windows: convert it to USD first (Blender: File › Import › Alembic, then File › Export › Universal Scene Description; Houdini and Maya export USD directly).

## Smoke from a volume (OpenVDB)

An emitter's *Shape › Volume (VDB)* takes an OpenVDB file from Houdini, Blender, EmberGen or Blackbody itself, a numbered sequence of them (`smoke.####.vdb`), or a Volume prim in a USD file. Its density grid is the smoke; a temperature grid, if it has one, is its heat. *Volume* sets what the emitter does with it:
- *Fills the box with its smoke, once*: at *Ignite at* the smoke goes in as it is and then moves with the air, as a cloud of smoke, a gas leak or haze.
- *Keeps it topped up*: every step while it is on, the smoke is kept at least as thick as the volume, so an animated sequence (an explosion from another program) drives the smoke and the simulation carries it on.
- *Releases where it is dense*: a source that releases at the emitter's rates wherever the volume is dense.

For the first two, *Smoke* and *Fuel* are the amount where the volume is densest and *Heat* is its temperature. *Position*, *Size* and *Rotation* place the volume; *Z-up file* turns a Blender VDB up the right way. Fuel in the volume burns like any other: a VDB gas cloud goes up in a flame front.
