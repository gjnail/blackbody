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
- Things that break (Properties › Breaking): objects cut beforehand into chunks, bricks in running bond, glass
  shards or wood splinters, glued at joints that give way when pulled, sheared or bent past the material's
  strength. Hollow objects break as shells (a vase, a crate). Standing ones are held by their base or their edges.
  Hundreds of pieces, drawn on the stage and in footage, hiding the fire and the liquid behind them; the gas and the
  water flow round them and are pushed by them; dust where they break. Blocks: brick wall, glass pane, concrete
  pillar, wooden crate, vase. Presets: Ball through a brick wall, Stone through a window, Vase off a table.
- Ropes, springs, hinges and ball joints (Properties › Joint): an object hangs on a rope (it swings, goes slack and
  is caught with a jolt), bounces on a spring, turns on a hinge or swings about a ball joint, from a fixed point or
  from another object (a beam, a crane's jib, a moving arm, another falling thing). Ropes and steel cables are drawn
  hanging in a curve when slack, springs as coils; they snap, and hinges tear out, past Breaks at. Ropes and hinges
  blocks: wrecking ball, rope swing, door on hinges, hanging lamp, weight on a spring, seesaw. Preset: Wrecking ball.
- A warning names a falling thing that starts inside something fixed (it is pushed out when it is let go).
- Sand, snow, mud, jelly and clay (matter): bodies of it, and streams poured from nozzles, simulated as hundreds of
  thousands of particles on the GPU (MPM) in every kind of scene but the sky. Sand pours and piles at its angle of
  repose, wet sand holds steeper shapes, snow packs and breaks up, mud slumps until its yield stress holds it, jelly
  springs back and wobbles, clay stays squashed. Things that fall land on it or sink into it as their weight says,
  and keyframed objects plough through it. Drawn on the stage and in footage: sand and snow glint, snow lets light
  in, mud is glossy and jelly clear. Sand, snow & mud blocks (sand pile, sand pour, sand column, snowball, snow
  drift, mud, jelly, lump of clay) and presets: Sand from a hopper, Snowballs at a wall, Ball dropped on jelly, Crate
  through mud. The guide: docs/matter.md.
- Explosions: an emitter's Blast (kilograms of TNT) goes off when it ignites, and its blast wave throws the things
  that fall, blows breakable things apart piece by piece and scatters sand and snow, by the impulse a blast that size
  gives at that distance. The viewer marks each charge, its fireball's size and when it goes off. An Explosion block
  (Fire) and a preset, Blast in a yard.
- Lightning: a kind of light. A bolt from its Position to where it Strikes, jagged at every scale and branching,
  flashes a few times (its return strokes, the branches in the first only), glows white-hot with a halo, lights the
  set and the smoke through the flash, and sets fire to what it strikes. A Lightning block (Lights) and a preset,
  Lightning strikes a post.
- Tilted objects: Tilt and Roll (Properties › Shape) tip any object over after its Rotation: a ramp, a leaning wall, a
  wheel on its side. Keyframe them to tip a tray or flip a flap. The fire, the liquid, sand and snow, cloth and things
  that fall all meet it at its slope: a box slides down a plank steeper than its friction angle and stays put on a
  gentler one. A Ramp block (Objects) and Ball down a ramp (Things that fall).
- Motors: a hinge's Motor speed (turns a minute, keyframeable) and Motor strength (its most torque) drive it round,
  pushing back on what it is joined to. Wheels drive a cart; a motor too weak for its load stalls; a speed keyed down
  to 0 brakes it. Machines blocks in Create: a motor cart, a turntable that flings what is on it, a windmill. A
  preset, Cart off a ramp: a burning cart jumps a ramp into a tower of blocks.
- Presets can attach things to their objects (a fire riding on a cart), and loading a preset into a shot brings its
  attachments instead of keeping the shot's old ones.
- Grass and plants (Create › Grass & plants): patches of lawn, long grass, wheat and reeds, every blade simulated. They
  bend in the wind in rolling waves and in the fire's draught, spring back, part round what moves through them, and
  grow on the ground or on the objects under them (a grassy hillside). Fire catches in dry grass and runs through it
  by itself, faster downwind, burning it down to black stubble and feeding the flames with its fuel and smoke; fresh
  grass is hard to light. Drawn in the same light as the smoke, glowing where the sun is behind it, in fire and in
  liquid scenes. A preset, Meadow fire, and a guide, docs/grass.md.
- Cloth and grass are drawn into one layer, so each hides what is behind it of the other.
- Burnt things look burnt on the stage: a burnable object browns as it heats, blackens with glowing embers as it
  burns, greys with ash and smoulders; the burnt floor blackens too.
- Things that break and burn: an object both Breakable and Burnable burns piece by piece. Each piece catches where
  flames touch it or from a burning piece glued to it, burns (thicker ones longer), feeding the fire, smoulders, and
  most crumble to ash; the glue between pieces weakens as they char, so a burning structure falls in. A preset, Shed
  on fire.
- Sand, snow, mud, jelly and clay are solid to the smoke and the water: smoke goes round a heap of sand, water poured
  on it runs off and pools at its foot, and a pour pushes the air aside.
- Snow melts where hot gas touches it, its water joining the liquid in a fire-and-liquid box.
- Sand and water: in liquid and fire-and-liquid boxes the water and the matter push each other (a wave shoves jelly
  and it floats; grains in the water are lighter by it; water running over sand drags it along, so a pour digs a
  crater and a gully). Dry sand the water touches gets damp (darker, glossier, holding together); the water seeps on
  into it, and sand it soaks through lets go, so a sand castle the water reaches slumps and the flow carries it off.
  A Sand castle block in Create, and a preset, Sand castle and a wave.
- Sand, snow, mud, jelly and clay in the interface: the object list, Properties, outlines and handles in the viewer (a
  pour's nozzle and direction), Made of, Let go, Pour it from here and Start/Stop pouring from the right-click menu,
  timing lanes, a Sand & mud chip in Create; Effects gains Falling & breaking and Sand, snow & mud chips.
- Make it fall and Drop it at this frame on any object (right-click it, the Fall chip on its page, or Ctrl+K), and
  Stop it falling. Falling things are drawn and picked where the simulation has them in the viewer, tilted as they
  tumble, and outlined in green. In Build, liquid scenes show objects in their materials instead of grey stand-ins.
- Make it breakable and Make it break into (Chunks, Bricks, Shards, Splinters) on any box, ball or cylinder
  (right-click it or the Breakable chip on its page), and Stop it breaking. Once it has broken, the viewer no longer
  draws or picks it as the whole object.
- Hang it on a rope, Put it on a spring, Hinge it and Put it on a ball joint from the right-click menu, Tie it with a
  rope to and Hinge it to another object, and Take it off its joint. The viewer draws ropes and springs as they hang
  (a dashed line before the simulation has run), and renaming an object keeps what is tied to it.
- Lightning in the viewer: the bolt drawn down its channel with its branches, and a handle to drag where it strikes.
- Tilted objects drawn and stretched along their tilt; Tilt and Roll hidden on balls, a motor's settings shown only on
  hinges; a Machines group in Create with icons of its own.
- The README and website rewritten for the new features, with a 90-second film, new screen recordings of the app
  throughout, and the tutorial's screenshots retaken.
- Grass and plants in the interface: the object list and Properties, the patch and how tall it grows drawn in the
  viewer, Set it on fire, Dry it out, Kind and Grow on objects too from the right-click menu, a Grass chip in Create
  with icons of its own.

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
  about 60 spots followed through the footage with sub-pixel affine Lucas-Kanade matching on the picture properly
  downsized (averaged, not every Nth pixel, so a spot lasts some 30 frames), spots that look like their neighbours
  skipped and lost ones replaced. Each spot is placed where the rays of the frames that saw it meet (on walls, posts
  and parked cars as well as the ground) and the camera's position and turn worked out on every frame from them, in
  real metres: the test courtyard dolly holds within 2 px, solved in about 10 s, without needing the line-up's lens
  to be exactly right.
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
