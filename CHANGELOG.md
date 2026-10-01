# Changelog

Notable changes to Blackbody. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Things that fall: any object can be a rigid body (Physics › Falls). It drops, tumbles, slides, bounces, stacks and
  knocks other things over, in fire, liquid and fire-and-liquid scenes alike (MuJoCo integrates it). The gas's drag
  pushes it (a blast, an updraft, the wind), it floats or sinks in water, a burnable one keeps burning as it falls, and
  emitters, lights and fabric pins attached to it go along. Falls from (when it is let go), Thrown at and Spinning at.
- Materials for objects (wood, stone, concrete, brick, steel, aluminium, glass, ice, plastic, rubber, ceramic,
  cardboard, foam, painted metal, plaster, fabric, earth), with handbook density, friction and bounce; Density,
  Friction and Bounce override them.
- Things that fall blocks in Create: a falling box, a bouncy ball, a boulder, a thrown steel ball, a steel drum, a
  domino run, a tower of blocks and a stack of crates. Presets: Knocking down a tower, Crates into a fire.
- The stage: without footage, a floor out to the horizon (concrete, boards, tiles, dirt, grass, sand, a 1 m checker or
  studio grey) under a sky in the scene's ambient light or HDRI, with objects drawn in CG in their materials. They are
  lit by the key light with soft shadows, the sky, the fire with shadows and the lights in the set, and glass and ice
  refract. Composite › Backdrop chooses it or the flat background colour (presets and saved scenes keep their colour).
- CG objects in footage: an object's Look (Automatic, CG or In the footage). A CG one goes over the footage lit by
  the shot's light, with its shadows on the real ground, hidden behind the footage's depth pass and matte.
- Fixed mesh objects hold falling things on their real shape (between the logs of a pile, on stairs).

- Layers: several effects in one shot (a campfire in front, a waterfall behind), each its own simulation with its own
  box and scale, composited back to front in the viewer, renders and the command line.
- Roto: shapes drawn around what is in front of the effect, keyed over the shot, hide the effect behind them.
- Build and Shot workspaces. The app opens in Build, on an empty stage with a camera of your own and a card to put
  in fire, water, cloth, smoke, snow or an object; Shot puts the effect in your footage. (The work view became Build.)
- A transform gizmo: arrows to move things along X, Y and Z, squares to stretch them, a ring to turn them, Ctrl to snap,
  and the size or place shown by the cursor while dragging.
- Things you can do to an object (right-click it, or the buttons on its page): set it on fire, make it float, sink, hot
  or freezing, hollow it out, soak, dry or let go of a cloth, choose what holds it and what it is made of, turn a source
  into fire, smoke, steam, water, lava or a force, start or stop it at a frame, attach it to another thing so it follows
  it, send it along a path you click out on the ground, drop it to the ground.
- Forces to place: a fan, an updraft, suction and a vortex push the air, and with it smoke, flame, embers and cloth.
- In Build, objects in a liquid scene are drawn as grey stand-ins.
- Text: type words, pick any installed font, and get solid 3D letters. Burning text has fire all over the letters,
  Text is a solid object (set it on fire and every letter catches at once), and Water letters fall and splash. Edit
  the words, font or size later from the right-click menu.
- Logos and pictures as shapes: Burning logo, Logo or picture and Water shape trace an SVG, PNG or JPG (its
  transparency, or what stands out from its background) into a solid, with Invert and a threshold.
- Several things at once: Ctrl+click, Shift+drag a box or Ctrl+A to select several; they move and turn together.
  Group, Ungroup, Duplicate, Delete, and Repeat in a row, a ring or scattered, with attached things coming along.
- Your own blocks: Save as a block keeps what you built (with what is attached to it and its meshes) under Yours
  in Create, as one .bbblock file to share; drop a block on the window to add it.
- The shot's camera from Build: Use this view, Key here (keys at two frames make a camera move) and Look through it.
- Timing on the timeline: each source's start and stop as a bar to drag, cloth let-go times, fills, and keys.
- Search everything (Ctrl+K): actions on the selection, building blocks, commands, settings, effects and objects.
- File › Pack project copies every file a project uses into a folder beside it and points the project at the copies.
- Line up the ground: drag a grid's corners onto a rectangle on the ground in the footage and the camera follows: the
  lens (from the vanishing points), the tilt, roll and height, with a perspective grid, the horizon and a 1.75 m
  figure drawn over the footage to check. Effects then stand on the real ground at real size; drag their base over
  it, or drop blocks onto the footage.
- Camera tracking for lined-up shots, for cameras that turn and cameras that travel (dollies, walks, cars, drones):
  about 60 spots followed through the footage with sub-pixel affine Lucas-Kanade matching (spots that look like
  their neighbours skipped, lost ones replaced), the camera's position and turn worked out on every frame from the
  spots on the ground (RANSAC, then a bundle adjustment of every camera and spot together), in real metres.
- Line up from the horizon when there is no rectangle in view, and on sloping ground with two upright lines (the world
  stays level; the slope becomes solid ground).
- Surfaces: line up a wall, a table or platform, a ramp or stairs in the footage and it becomes a solid there (in every
  layer, kept in place when the effect moves) that effects meet and go behind. Blocks dropped onto the footage, and
  the effect's base, land on the ground, table tops, steps and slopes.
- The lens is read from footage that records it (photo EXIF, phone video).
- Shot steps: a card in the Shot view that walks through putting an effect in footage, step by step.
- Contributor guide, code of conduct, security policy, issue and pull request
  templates, and CI that checks Windows, macOS and Linux.
- Ko-fi links in the README and on the website.

## [1.0.0] - 2026-09-30

This is the first public version.

- **Fire and smoke:** a reacting gas solver checked against real fires (flame height, plume speed, puffing). It has swirl, moving emitters and colliders, dousing, sparks, colourants and steam. Fire can spread over the ground and burnable objects, start spot fires, fill tracked air in closed rooms, run as flame fronts, burn soot onto surfaces, and catch on meshes, terrain and deforming meshes.
- **Liquids:** FLIP/APIC on the GPU with spray, foam and bubbles, floating rigid bodies, viscosity, surface tension, dye, two liquids, rain, open water, a narrow band, and a box that follows the action. The liquid is ray traced as a dielectric, with caustics and an underwater view.
- **The sea:** an FFT ocean to the horizon with whitecaps and foam streaks, open water around the box, beaches and surf, surges, bores, tsunamis, tides and currents.
- **Lava, and fire with water:** molten liquids with a glow and a crust, and fire, water and lava in one box.
- **Heat:** freezing into rigid floating ice, boiling (nucleate, transition and film), evaporation and steam.
- **Weather and clouds:** snow, sleet, freezing rain, graupel and hail decided by the column of air above, and a cloud simulation with microphysics.
- **Fabric:** cloth made from ten real materials. It burns through, soaks, drips and steams.
- **Into your footage:** holdouts, fire light on the footage, the lens and its haze, noise matching, lights in the set, OCIO, a 2D tracker, camera tracks, and USD and VDB import.
- **Outputs:** composites, ProRes 4444 and PNG elements, multi-layer and deep EXR, OpenVDB, and USD and OBJ meshes. The command line can share a disk cache across a render farm.
- **The app:** an Effects panel of presets, a Create tab of building blocks to start a scene from scratch, an Essentials page, a search across all settings, and an animation editor.
- **79 presets**, plus a tutorial, guides and a website.

[Unreleased]: https://github.com/gjnail/blackbody/compare/21e8050...HEAD
[1.0.0]: https://github.com/gjnail/blackbody/commit/21e8050
