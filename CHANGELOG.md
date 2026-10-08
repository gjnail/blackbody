# Changelog

Notable changes to Blackbody. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Build more with joints and ropes: an object can have several joints (*More joints*: a plank hung level from two
  ropes, a swing on two ropes), a rope goes over several posts and pulleys (found by itself, or named in *Goes over*)
  and can have weight (*Has weight*: it hangs in its own curve, drapes over what it meets and holds on a post by
  friction), and chains give way where they are tied. New blocks: Swing on two ropes, Rope over two pulleys, Rope
  bridge. A falling mesh collides by its own shape (a bowl holds a ball), and fixed hollow barrels, pipes and balls
  are hollow to falling things too.
- Scenes fit the GPU's memory before anything is made: Blackbody reads the card's memory, estimates what a scene
  needs, and steps Detail upres and then the voxels down with a notice saying what it cut, instead of failing part
  way through. If the GPU still runs out, or its driver resets, the app starts its engine again on its own and carries
  on (a disk cache keeps its frames and layout) instead of needing a restart. The status bar shows the GPU memory in
  use, and the Voxels tooltip what a final render needs. A disk cache keeps the layout it was simulated at, and a
  render from it reads it as it is (`BLACKBODY_GPU_MEMORY` tells a farm machine its size). Grow to fit stops at what
  the card has room for and at its largest 3-D texture, says so, and is offered only in fire scenes, where it works.
- Importing: a VDB's velocity drives the smoke ("Velocity from the volume"), a VDB sequence blends between its frames
  instead of jumping, and a VDB keeps its own detail down to half a simulation cell (it was cut to 192 cells a side);
  bit-shuffled Blosc VDBs read. USD points that move or carry velocities come in as a particle source (a fill in a
  liquid scene), IES profiles on disc and rect lights, portal and geometry lights and shadow-off lights are kept, and
  the command line's `--usd` takes the stage's frame range and rate (`--usd-offset` renumbers a 1001-based shot).
  Many meshes and VDBs no longer stop a scene with "Too many meshes for the GPU": they are packed in columns, and what
  still does not fit is left out with a notice.
- Compositing passes in the EXRs: motion vectors (forward and backward, in pixels, for VectorBlur and retiming),
  normals and positions (N, P), a matte for every named thing in the set (objects, broken pieces, ropes, the liquid,
  the fabric) and Cryptomatte (objects and materials), in fire, liquid and fire-and-liquid renders, in element and
  composite EXRs (traced through the footage's lens distortion). The fire's own motion is in the vectors. On by default
  for EXRs ("EXR compositing passes" in the Render window, --no-passes on the command line); an EXR with them asked for
  DWAA or DWAB is written with ZIP instead, which keeps the ids exact. They add one and a half to two seconds to a
  full-HD EXR frame, mostly writing.
- Export the shot to other programs: a USD scene of the whole shot (the camera, every object moving frame by frame,
  broken pieces, people's and cars' parts, ropes, sand, snow and mud as points, grass as curves, embers as points),
  and the camera alone as USD or a Nuke .chan. Every EXR now carries the camera in its header (worldToCamera,
  worldToNDC). VDB, mesh and scene outputs cover every layer of a shot (each other layer's files beside the base
  layer's); before, only the base layer's were written. What an output leaves out (bullet debris, hollow shapes, the
  .chan's slide off centre, the lens distortion) is said before the render.
- Notices: what the engine leaves out or cuts short is now shown in the app and printed by the command line, instead
  of being dropped silently. Objects past 16 (not drawn either), sources past 16 (counting the water level's and
  lightning's slots), set lights past 8 (a lightning bolt takes 4 while it flashes), fabrics past 16, grass patches
  past 8 and blades thinned, sand, snow and mud past their material slots, weather particles that could not spawn,
  bodies the water's push limit held back, mesh errors and a colour setup that failed. Repeat warns before it makes
  copies past the limits. Grass shares its blade budget fairly between patches (the first one took it all).
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
- Lume, a path-traced lighting engine for the set drawn in CG (Properties › Lume, off by default): light bounces from
  surface to surface, shadows are as soft as each light is big (the sun's disc, each fire light, hot metal, lamps,
  through the smoke), an HDRI's own sun is found and shadowed (importance sampling with MIS), glass, ice and jelly
  refract and bounce light inside, and highlights are GGX. Lamps are sphere lights, seen in reflections. The viewer
  sharpens while you look, and a denoiser clears the grain. It is measured against ground truth (tools/lume_bench: the
  same scenes rendered by Mitsuba 3 with Lume's own material) and agrees with it to within 0.1% in mean brightness on
  every one of its ten test scenes. Caustics are traced from the lights, from each lamp's surface, through glass and
  off mirror-like metal: the bright spot a glass ball focuses into its shadow, the sky's glow through it and the light
  it throws on round it. The light paths also leave their light in a cache of where they land, so a caustic seen
  through its own glass or in a mirror is focused too. Its samples are spread evenly (Owen-scrambled Sobol), and after
  the same time it has 0.036 to 0.86 times the error of Mitsuba's best sampler for each scene on nine of the ten (the
  tenth, a furnace test that renders in a few milliseconds, 1.2 times). Lume compiles
  its own camera kernel for what the set has, on first use (a few seconds). A lamp's light off glossy metal, plastic
  and painted coats is traced from the lamp (no sparkles), and the denoiser carries its estimate of the noise through
  its levels (half the error it left before, thin highlights kept).
- Lume lights the whole frame: the set, the smoke and steam, and the cloth and grass. In the smoke it traces the light
  at each point from the flames (each an optically thick blackbody), the key light and the sky, scattered and bounced
  in the smoke, on a grid of at most 56 cells a side, settling while the frame holds still. Checked against Mitsuba
  inside thick pale smoke and steam and darker smoke, it is within a tenth on average, where the classic engine's
  estimate is 1.4 to 5 times too bright. So under Lume smoke can look darker than under classic: set Smoke colour to
  what the smoke is (pale for wood and grass smoke, near black for oil and rubber; steam near white).
- Lume traces water: its surface a rough dielectric with exact Fresnel and its ripples, the water absorbing in its
  colour and its murk scattering, foam a white lace, spray and bubbles scattering, the floor under it lit by the caustics
  its surface makes, in the water's colour, and Snell's window from underwater. (The wave sea, lava, ice, dye, fire and
  water in one box, and liquid scenes with fabric or grass are still drawn by the classic water renderer.)
- Lights as real lights: an area light is a panel of its own width and height that Lume samples over the solid angle it
  covers (the classic engine lights by a disc as big); a light can take a measured profile from an IES file (and its
  brightness from it), and its output can be given in lumens. USD rect lights come in with their size and their IES
  profile; old projects' area lights become squares of the same area.
- A physical sky (Lighting › Sky: Physical): the sky, the sun's colour and the light the air throws on the set from
  a physically based atmosphere (Hillaire 2020): blue overhead, paler toward the horizon, reddening at sunset, with
  Haze, Ground brightness and Altitude. It lights the stage and Lume, and the water reflects it.
- Plain balls, boxes and cylinders on the stage are hit exactly by each ray instead of marched toward: Lume is up to 26
  times faster on such sets, and the classic stage is faster too.
- Things that break (Properties › Breaking): objects cut beforehand into chunks, bricks in running bond, glass
  shards or wood splinters, glued at joints that give way when pulled, sheared or bent past the material's
  strength. Hollow objects break as shells (a vase, a crate). Standing ones are held by their base or their edges.
  Hundreds of pieces, drawn on the stage and in footage, hiding the fire and the liquid behind them; the gas and the
  water flow round them and are pushed by them; dust where they break. Blocks: brick wall, glass pane, concrete
  pillar, wooden crate, vase. Presets: Ball through a brick wall, Stone through a window, Vase off a table.
- Wood burns as wood does (wood that Breaks and is Burnable: engine/wood_fire.py), spot by spot over its pieces'
  surfaces (a grid of spots on each face, the seams between pieces burning as one), so its char starts where the
  flames touch it and spreads from there rather than a whole board darkening at once. It catches where enough heat
  reaches it, its surface brought to its piloted ignition temperature by the flames' and hot smoke's heat as its
  thermal inertia says (about 5 s in flames for pine, never below about 12 kW/m^2), so hot smoke trapped under a roof
  can flash a room over. Fire climbs it fast and creeps over it at about a millimetre a second, slower downward. It
  chars at its charring rate (0.65 mm a minute for softwoods, 0.5 for hardwoods), its heat release fiercest as it
  catches and falling as its char thickens, goes out where too little heat reaches it, chars through its whole
  thickness (from one face or both: a wall burning from inside takes twice as long as a board in the flames), and its
  glue weakens with what is left of it. Spreading fire › Burn speed-up (keyframable) time-lapses its burning while the
  flames and smoke move at their own speed. On the stage, char cracks into blocks that grow as it deepens, glowing in
  its cracks in patches, greying with ash where it faces up, its front ragged and running on across the seams between
  boards. Preset: Shed on fire, rebuilt (a rubbish fire in the
  back corner of a pine shed with a door: it climbs the corner, flashes over, chars through and falls in).
- Breaking that looks real: a piece breaks away when something meets it faster than its material takes (glass and
  pottery about 2.5 m/s, brick 2, wood 7, steel 60), less so when what hits it is much lighter. Where each thing is
  first hit is found by running the shot once with nothing breaking, and its cracks crowd round that spot: small chips
  there, larger pieces further off. A crack runs on as far as the material lets it: glass and pottery shatter right
  through (the glass round a stone's hole cracks to the frame and stays in it), wood splits off where it is hit, metal
  not at all. Metal and plastic bend before they break and stay bent (Pattern › Bends (metal)): a post hit by a
  wrecking ball folds over. Meshes can break, a vase breaks into curved shards of mixed sizes, and things are drawn
  whole until their first crack opens. Joints are ten times stiffer (a wall no longer sways like jelly), and mortar
  holds 0.6 MPa.
- Bullets (Create › Guns & bullets: a pistol shot, a rifle shot, a shotgun blast, a machine-gun burst): a shot fires
  a real cartridge's bullets, from an air-rifle pellet to .50 BMG, from its muzzle toward where it is aimed (both
  keyframable), with Rounds, Rate of fire, Scatter, Muzzle speed, Tracer and Muzzle flash. They fly their real paths and
  do what real bullets do to what they meet: through glass, wood, drywall and car doors and on, slower; flattened on
  steel in a splash of lead and sparks; craters chipped out of concrete, brick and stone in a puff of dust; skipping off
  water, concrete and steel when they come in flatter than its critical angle. Glass cracks in its web round the hole,
  bottles burst, a steel gong swings, and what they hit is moved by what they carry. In the interface: a gun's muzzle
  and aim drawn in the viewer, with a handle to aim it. Presets: Shooting range, Bullet through glass, Bottles on a
  fence, Steel gong, Machine gun at dusk. (From the bullets session.) What they leave is drawn as it is: 3-D splinters standing out
  of a board's exit, faceted craters in concrete and brick, frosted crushed glass and its web of cracks, lead splashed on
  steel, MDF broken out round; cloth holed as finely as the bullet; and deep cavities in water.
- Wood: fifteen woods (pine, spruce, Douglas fir, oak, ash, maple, birch, walnut, cherry, mahogany, teak, cedar, balsa,
  plywood and MDF), each with its density, stiffness and strength along and across its grain from the USDA Wood
  Handbook, and drawn solid through: rings, grain, knots, pores, rays and heartwood, so a cut, a break or a bullet's
  hole shows the wood inside. It splits along its grain far more easily than it breaks across it: a board snaps in a
  jagged break with splinters standing out of both halves, a log splits along its rays. A board bends before it
  breaks (as stiff as its wood, about half its clear wood's strength, as sawn lumber is rated), and end grain struck by
  an edge or a corner cleaves. Presets: Breaking a board, Bullets through boards, Woods.
- People and cars (Properties › Physics › Build): an object built of many parts on joints. A person is a crash-test
  figure, 1.8 m and about 80 kg, its head, chest, pelvis, arms and legs turning within a body's reach (a knee bends
  only back). It stands braced until something hits it faster than 1.5 m/s, then goes limp and falls as a body does,
  over a car's bonnet or down the stairs. A car is 4.4 m and about 1300 kg on four sprung tyres; Drive (rear, all or
  none), Speed (its motor's, keyframable: 0 brakes, below 0 reverses) and Steer. It drives over grass leaving its
  tracks flattened, knocks walls down and carries a person on its bonnet. Blocks: People and cars. Presets: Crash
  test, Stunt fall.
- A rope or a steel cable goes round a post or a ball in its way: thrown over a beam, a crate hangs from it and swings
  under it. Chains (Rope is › Chain) are steel links: as heavy as a real chain of their thickness (12 mm: about 3 kg a
  metre), hanging in their own curve, catching on and draping over what they meet, piling up where they land, and
  holding a couple of hundred kilograms. Preset: Chain and rope.
- Grass is pushed aside by things that fall, broken pieces and people's and cars' parts; pressed flat by a wheel or a
  foot it lies down, a track that stands up again over about 40 s.
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
- Things that melt: wax, chocolate, aluminium and iron (Matter › Made of) warm in the fire, from its hot gas and its
  radiant heat on the side they face it, cool in the air and the water, and past their melting point melt into a
  runny or thick melt that runs and puddles and sets again as it cools. Hot metal glows as a blackbody at its
  temperature, as bright as a flame that hot, and lights the floor and objects round it. Their melts can be poured.
  Matter › Temperature; Domain › Heat speed (how much quicker than for real). Presets: Pouring molten iron, Chocolate
  by a fire.
- Hot and cold objects (an object's Temperature) warm or chill the wax, chocolate and metal that touch them, as fast
  as the two conduct heat (their thermal effusivities), and melt the snow resting on them: chocolate melts where it sits
  on a hot plate, sooner on steel than on wood, and snow on a hot plate melts from below. Preset: Chocolate in a hot
  pan.
- Things that burn: dry leaves, sawdust and coal (Matter › Made of) catch where the fire's heat takes them past their
  ignition point, give the fire their fuel as they burn (so it runs through a pile by itself: leaves in a flare of
  flame, sawdust smouldering, coal glowing) and burn down to a little ash.
- Snow, mud, clay and wet sand stick to the objects they touch, each as hard as it really does: a packing-snow ball
  thrown at a wall stays on it, powder snow slides off, dry sand does not stick.
- Things that fall land on cloth and rest in it, and the cloth carries their weight: a crate dropped onto a sheet tied
  at its corners sags it and stays there. Each vertex of cloth pushes back with at most 100 N, so a sheet cannot stop
  a wrecking ball however hard it is hit.
- Cloth that tears (Fabric › Tears, Tear strength): it rips where it is pulled too far, caught on something moving
  through it or overloaded, the rip running on from where it starts as a ragged slit (the vertex at its tip goes
  next, judged by the pull on the threads it has left).
- Fabric held by all its edges (Held by › All its edges): a trampoline, a sheet laced into a frame. Preset: Ball
  through a sheet (a crate rests in a sheet laced into a frame; a steel ball dropped beside it rips through, and the
  crate falls in after it).
- Cloth and sand, snow, mud, jelly and clay meet: matter cannot pass through fabric, from either side. Sand poured onto
  a sling heaps in it and weighs it down, and a sheet dropped onto a heap drapes over it and leaves it standing.
  Preset: Sand into a sling.
- Broken objects' pieces and sand, snow, mud, jelly and clay push each other: bricks from a wall knocked onto a heap
  land on it, dent it and are held up by it (and breakable things dropped onto sand rest on it).
- Matter fills meshes: Shape › Mesh fills an OBJ, STL or USD prim of your own (a chocolate bunny, a sand sculpture, a
  jelly from a mould), scaled by Size; the viewer outlines its mesh, and it sits on the ground and frames by the mesh's
  own size. Blocks: Matter shape, Chocolate letters (any text, in any font).
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
- Objects have heat. Every object warms and cools as the real thing would, by what it is made of (its heat
  capacity and emissivity, new in the materials) and its size, a temperature for its surface and one for its inside:
  steel evens out in seconds, a log's skin scorches while its heart stays cold. It takes heat from the air or the
  fire's gas moving past it, from what radiates (the fire, glowing matter, lava, other hot objects), from the ground
  (as fast as the floor's material takes heat), from what it touches (other objects, water with Heat and phase changes
  on, lava, sand, snow, wax and chocolate), and gives it back the same ways: it radiates, a plume of hot air rises off
  it, and it warms or chills what it touches. Hot, it glows as a blackbody (dull red at about 500 C, orange by 1000 C)
  and lights what is round it. Heat › Keeps its temperature holds one where it is (a hot plate, frozen ground);
  Domain › Heat speed runs it faster. Preset: Red-hot steel quenched (a glowing steel ball dropped into water).
- One list of what radiates heat, shared by objects, cloth and matter: the fire's gas at its physical temperature,
  glowing matter and lava, and hot objects' faces.
- Wind moves sand, snow, leaves, sawdust and ash: past its threshold the wind lifts the top layer and drives it along
  (saltation, Kawamura's flux, within twice what is measured), airborne grains are dragged by it and settle at their
  own speed, and dusty matter throws dust into the smoke (fire scenes).
- Grass slows the wind blowing through it, by its blades' frontal area.
- Lava (fire-and-liquid scenes) meets broken pieces and sand, snow and mud, floats or sinks objects by its density,
  heats matter and objects, and radiates its heat.
- Presets: *Snow blowing off a heap* (a 15 m/s gale carrying powder snow off a heap's windward face in a plume)
  and *Log on a lava flow* (a pine log riding high on the flow and carried off, a steel block half swallowed).
- Cloth pushes water: the drag a sheet feels from water goes back into the water.
- Lume, round 9:
  - Light passes (Lume › Light passes): a final render also writes the set's light split by where it comes from into
    the composite EXR, as the layers light_key, light_sky, light_fire and light_lamps, to turn each light up or down
    or recolour it in the comp. The export dialog gets Composite · EXR sequence.
  - Caustics from the fire's light: a glass beside a campfire focuses its flickering light, soft as a flame is big (with the fire's
    shadows on).
  - Dispersion (Lume › Dispersion): glass, ice, jelly and water bend each colour by its own index (Cauchy's formula
    from the material's Abbe number), for the rainbow fringe round a caustic and the fire of a gem; it makes nothing
    brighter or darker.
  - Clouds traced by Lume: the light a cloud scatters more than once is followed through up to 512 scatterings by the
    droplets' own phase, and comes out within 3% of Mitsuba path tracing the same cumulus from its sunlit side and
    from below, where the classic estimate is about 40% too dark (tools/lume_bench: cloud_blackbody.py,
    cloud_mitsuba.py).
  - Lighting › Environment from the footage: the shot itself lights the CG set, as a panorama round it, so an object
    takes the footage's colours from each side and reflects them.
  - Dyed water (ink, milk, blood, mud) traced by Lume, absorbing and scattering in the dye's colour.
- A test that every source file compiles (tests/test_compiles.py, in CI): a syntax error in a module only a window
  imports otherwise passes every other test.

### Changed

- A fire built from scratch moves at real speed: the solver's gas rises at about two-thirds of real speed at Time
  scale 1 (measured against McCaffrey and Zukoski's puffing rate at three sizes), which the presets already correct
  with Time scale 1.5 or more. A scene's first steady fire now gets the same 1.5, with a note; anything that moves at
  its own real speed (water, falling things, fabric, sand, grass, bullets) puts it back to 1, and a time scale set or
  keyed by hand is left alone.
- A block that boils water (the Hot plate, a hot pan) turns a Liquid scene into Fire and liquid, so its steam shows.
- Sources on tilted or tumbling objects turn with them (they turned only about the vertical).
- Matter's heat from radiation is now the physical one (it was about 15 times too strong), so things melt by the fire
  as slowly as they really do; Chocolate by the fire is retuned to match. Matter conducts to objects and the ground as
  two bodies in contact do (fast at first, slowing), and the ground conducts.
- Objects now warm and cool by default; presets with heaters, frozen ground or dry ice keep their temperature, and
  scenes saved before open with Keeps its temperature on for every object whose temperature is not 20 C.

### Fixed

- The first liquid scene no longer waits 8 to 20 minutes for its shaders: the liquid renderer's shader compiles cold
  in 20 to 55 s (and 1.8 GB of memory instead of up to 22 GB), and the app compiles it in the background as soon as
  it opens, so it is usually ready before a liquid is. The same picture draws faster too (the tsunami 181 to 64 ms a
  frame at 1280×720). `blackbody precompile` does it ahead from the command line (for farm machines). With
  Direct3D 12 it takes about 3 minutes in every new process: use Vulkan where the driver offers it.
- Rain, snow and hail fall the same way every time (which pieces were made, and where, depended on the order the GPU's
  threads ran, so a farm or a cleared cache gave another fall), and hailstones come to rest on the ground instead of
  about a centimetre above it. Glowing sand or lava past its 256 lights, and ice past its 8,192 pieces, keep the same
  ones every run (the brightest, by a fixed rule) and say so in the notices; the heat that fire, lava and hot objects
  give to fabric and sand is summed in a fixed order.
- Fabric simulates the same every time. A scene simulated after another in the same session could take the last
  scene's fabric and dust into its smoke on its first frame; fabric touching itself summed its pushes in an order
  the GPU chose; and on some drivers one shape of division rounds two ways from run to run, which the cloth's chaos
  grew into a different drape. A cloth dropped on a heap of sand no longer lets the heap's top grains through it
  when it falls fast.
- Two presets fixed, found by the new test of every preset: one of the tsunami's cars started inside the house, and
  Snow blowing off a heap made most of its snow below the ground (its heap was centred on it) and threw it away.
- Environment (HDRI), its rotation and strength, and Key light from environment can be set and work in fire scenes;
  Environment from the footage works in liquid and fire-and-liquid scenes; liquid scenes show the holdout matte and
  depth pass settings their engine already reads. Turning an environment off and on again no longer breaks every
  later render, and Lume no longer drops the HDRI's light after that.
- Final renders with motion blur no longer show a fine diagonal hatching: each pixel's shutter offset is hashed
  instead of taken from a lattice noise (the hatching's peak fell from about 250 to 11 times the background, the
  grain unchanged). Sand, snow and mud push the right broken piece past 2047 pieces (piece ids were half floats).
  Lume picks glass and mirror-like objects first for its eight caustic targets, so a glass ball listed after eight
  pots casts its caustic.
- Farm renders from the disk cache with a .vdb output write the cached fire (they wrote the unsimulated start state
  for every frame), placed by each frame's own transform; a frame neither live nor cached is an error, not a stale
  file. A sky scene's .vdb output writes its clouds (density, cloud water, ice, rain, snow, hail, velocity) instead of
  failing or writing a stale fire. Deep EXR no longer writes nothing in shots with layers, and 16 samples per pixel
  is an option. Fire scenes get a Fabric mesh export in the Export dialog.
- A clean install works: USD (usd-core) and OCIO (opencolorio) are installed with Blackbody, so USD import and colour
  management no longer go missing on a new machine (and in the standalone build). Python 3.12 or 3.13 is required and
  the launchers check for it, saying what to get. A failed or interrupted install is retried on the next launch, a
  changed requirements list installs what changed, and on Windows the launcher shows what went wrong and waits
  instead of doing nothing; every command-line command now runs in the console with its exit code.
- Bottles, vases and other brittle vessels burst when shot. A hollow object's wall was counted thin twice, so a .22
  left too little of itself in a 3 mm bottle wall to break it, and only the shards in its path came away: the rest
  stood on in a bottle's shape. Panes and plates keep their holed web as before.
- A glass pane stays whole until it is hit: a few slivers of bonds in its web, all but missing each other, gave way at
  rest and showed its cut seams before the bullet came.
- A cached frame saved before a breakable object's welds changed no longer loads (it re-armed broken welds among
  scattered pieces and threw them apart); the frame is simulated again.
- The disk cache is kept only for the version of the physics that simulated it and the objects as they were built:
  frames simulated before an update that changes how things move or break (Bottle shoot's standing bottles) are
  simulated again instead of shown. A checkpoint the objects no longer fit is passed over before anything is set up
  for it.
- Things broken into chunks (stone, concrete, pottery) and breakable meshes stand whole until they are hit. Every
  joint was loaded alike whatever its glued area, so where two chunks barely met it gave way as the shot began (the
  Concrete pillar dropped three joints and an anchor, puffing dust); joints now take their share by area. A standing
  mesh was glued to the ground by every piece face that looked down (under a chair's seat as well as its feet), its
  pieces reached 0.6 mm past the mesh, so set on the ground they started in it, and pieces filling in its hollows
  started inside each other and pushed it apart. Of 289 objects left at rest, standing or lying, 264 broke joints
  before (9,385 in all); 14 do now (43: a loose pile of logs settling, two ice balls).
- A pane hit near a corner no longer fails to simulate: the hit cut wedges a tenth of a millimetre thin against the
  frame, and small splinters came out as thin once their cut faces were moved in, which the physics refuses.
- Where a breakable is hit, which its cracks crowd round, is the same on every machine: the run that finds it stopped
  after 15 s of the machine's time as well, so on a slow or busy one Bottle shoot's last bottle was never hit in it and
  broke differently from run to run. It now stops on the work done, and takes about half as long as before.
- Dust thrown up by the wind or a blast showed in previews but not in final renders: it went only into the coarse
  smoke grid, and a final render draws the fine one.
- Lume: coloured glass casts its shadow in its colour (its shadow rays carried the tint as grey); with a background
  colour backdrop, what is seen through water is the backdrop; a clear tank of water no longer glows with light trapped
  in it by total reflection (the least murk is for open water only).
- Running scenes one after another: a scene the size of the last drew on the last one's solid velocity, destroyed
  with it (an error), and a frame cut short so left the sand holding the last scene's fabric (the next one failed).
- Broken joints were marked whole again every step, so things breaking threw up twice the dust.
- An object's outline in the fire render is its own silhouette: rays that only passed within a third of a fire-grid
  cell of a ball, box or cylinder counted as hitting it, so the cloth behind it was hidden in a band round it (a
  crate on a sheet showed a halo of floor) and the fire stopped short of its edges. They now go on to its surface,
  or past it.
- Wood's bending joints flew apart in scenes with small pieces: there the physics takes a shorter step, and the
  joints were made as soft as the wood by a weaker hold on a stiffer spring, which below a point rings up (a shed
  with a door header burst in a tenth of a second, nothing touching it). Past that point the wood's give now comes
  from a softer spring instead, the same give.

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
