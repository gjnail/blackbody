# Things that fall

Any object (collider) can fall: it drops, tumbles, slides, bounces, stacks and knocks into other things, and the smoke, the water and the wind push it about. Turn on *Falls* in its Properties (*Physics*), or add one from the *Things that fall* group in Create. It works in every kind of scene: a crate dropped into a campfire, a boulder into a pond, a tower of blocks knocked down in an empty set.

![Knocking down a tower](media/gif/tower_knockdown.gif "Knocking down a tower: a bowling ball through a tower of wooden blocks, and a domino run, on the stage in sunlight.")

## Making something fall

- *Falls* makes the object a free rigid body. Its keyframes now set only where it starts: from there it moves by itself.
- *Falls from* is when it is let go, in seconds from the first frame. Until then it is held where its keys put it, so it can be carried or lifted into place and then dropped. Negative values drop it during the pre-roll, so it has landed when the shot starts.
- *Thrown at* gives it a speed when it is let go (a thrown ball, a launched crate), and *Spinning at* a spin. One with moving keys also carries on at the speed its keys gave it.
- *Material* is what it is made of: how heavy, slippery and bouncy it is, and how it looks. *Density*, *Friction* and *Bounce* set their own values where you need them (0, or -1 for friction and bounce, uses the material's).

The *Things that fall* blocks are real size: a falling box, a bouncy ball, a 1 m boulder, a thrown steel ball, a steel drum, a domino run, a tower of blocks and a stack of crates.

## Materials

| Material | Density (kg/m³) | Friction | Bounce | Look |
|---|---|---|---|---|
| Wood (pine) | 550 | 0.45 | 0.35 | grain and knots |
| Stone (granite) | 2650 | 0.6 | 0.25 | speckled crystals |
| Concrete | 2400 | 0.65 | 0.2 | blotches and pores |
| Brick | 1900 | 0.6 | 0.15 | bricks and mortar |
| Steel, aluminium | 7850, 2700 | 0.45 | 0.45 | brushed metal |
| Glass, ice | 2500, 917 | 0.5, 0.05 | 0.5, 0.25 | clear: they refract what is behind them |
| Plastic, rubber | 950, 1100 | 0.35, 0.9 | 0.5, 0.8 | glossy, matte |
| Ceramic, cardboard, foam | 2300, 120, 40 | | | |
| Painted metal, plaster wall, fabric, earth | | | | |

The numbers are handbook values. Friction is the Coulomb coefficient: a block on a slope slides once the slope is steeper than the angle whose tangent is its friction (24° for wood). Bounce is the share of its speed a thing keeps off a hard floor. A hollow everyday thing uses its weight over the space it takes up: a cardboard box 120 kg/m³, a crate of slats about 300.

## What moves them

- **Each other and the ground.** Things stack and stand in towers, slide and roll, and knock each other over. Objects that stay put hold them up; keyframed ones push them (a door swinging into a pile of boxes, a car through crates). A fixed mesh holds them on its real shape (things rest between the logs of a pile, on stairs); a falling mesh collides as its outline (its convex hull). An image heightfield is terrain they land and roll on.
- **The air.** The gas pushes them with its drag, measured around each one every frame: a blast blows light things away, a fire's updraft lifts paper, wind slides a cardboard box. They push the gas back as they move, as every moving object does: a falling crate shoves the smoke aside.
- **Water.** In a liquid scene they float or sink by their density, bob, drift with the flow and tip, as *Floats* objects do (*Falls* also lets them fall through the air first).
- **Fire.** A *Burnable* one with Spreading fire on catches where flames touch it and keeps burning as it tumbles: its fire is in its own frame.

Things attached to a falling object (*Attach to* in the viewer's menus) go with it: a fire on a crate, a lamp on a swinging sign, the pins of a flag on a falling pole.

![Crates into a fire](media/gif/crates_in_fire.gif "Crates into a fire: three wooden crates dropped onto a campfire; they land on the logs, catch and burn.")

## Breaking

*Breaks* (Properties › *Breaking*) cuts an object beforehand into pieces glued together, which come apart where it is hit or loaded harder than it holds. A wall is knocked through, a pane shattered, a vase smashed, a crate splintered. A joint gives way when it is pulled apart harder than its strength over its area, sheared harder than that plus the friction of what presses it together, or bent past what its section holds. So a wall stands under its own weight, a soft knock cracks it, and a hard one brings it down, the bricks above the hole caving in.

- *Breaks into*:
  - *Chunks*: stone, concrete, pottery.
  - *Bricks*: a box laid as bricks in running bond, held by mortar.
  - *Shards*: glass, slivers radiating from where it is hit.
  - *Splinters*: wood, pieces long along the grain.
- *Pieces*: how many (more take longer to simulate).
- *Strength* scales how strongly the pieces hold.
- *Held by*: what holds one that does not fall: glued to the ground along its base (a wall), held round its edges (a pane in its frame), or nothing.
- A hollow object (*Hollow* above 0) breaks as a shell: a vase into strips of its wall, a crate wall by wall.
- With *Falls* on too, it falls whole until it hits something hard enough: a dropped vase shatters, a thrown crate splinters.

Broken faces show the material's inside: raw brick, pale wood, the green edge of glass. Where it breaks it throws up dust, as much as the material holds (mortar, plaster and soil most, glass hardly any), in the smoke's colour (Shading › *Smoke colour*: a pale brown for dust). Hundreds of pieces are drawn, shadowed, hide the fire and the liquid behind them, and the gas and the water flow round them and are pushed by them.

The *Things that fall* blocks include a brick wall, a glass pane in its frame, a concrete pillar, a wooden crate that splinters and a vase. Presets: *Ball through a brick wall*, *Stone through a window*, *Vase off a table*.

![Ball through a brick wall](media/gif/wall_smash.gif "Ball through a brick wall: a 260 kg steel ball punches through, the bricks above cave in and the dust rolls out.")

## How they look

Without footage, objects are drawn in CG in their materials (wood, stone, brick and so on), lit in the same light as the smoke: the key light with soft shadows, the sky (darker in corners and under things), the fire's own light with shadows, and the lights in the set. Glass and ice are clear: you see the fire and the set through them, bent, and the sky in them. *Own colour* draws one in a colour of its own.

Behind them is the stage (Composite › *No footage*): a floor out to the horizon under a sky in the ambient light of Lighting (or the environment HDRI), or a flat background colour. *Floor* is what the floor is made of: studio grey, concrete, wooden boards, tiles, dirt, grass, sand or a 1 m checker for judging scale. The fire lights the floor and the objects around it, and smoke shades them. Presets keep the flat background they were made on; set *Backdrop* to *Floor and sky* to put them on the stage.

## In your footage

*Look* says how an object is drawn: *CG*, *In the footage*, or *Automatic*. Automatic is CG without footage, and with footage for things that fall or float (they cannot be in your footage). An object *In the footage* stands in for the real thing in the shot: it is not drawn, but it hides the fire behind it, and the CG objects behind it.

CG objects go over the footage lit by the shot's light, and their shadows darken the footage where they fall on the ground or on an object in the footage. The footage's depth pass and matte hide them behind the real things in the shot.

![CG objects in the footage](media/img/cg-in-footage.jpg "A CG crate and ball (they fall, so they are CG) beside a fire in a shot: lit by the fire and the key light, with their shadows on the real ground.")

## Limits

- A scene holds up to 16 objects in all, falling or not (a broken one's pieces do not count: there can be hundreds).
- A mesh cannot break yet, and there are no hinges, ropes or chains yet.
- The strengths are effective ones, set so that things break as they look like they should: a joint feels the whole
  impact spread over its face, where a real brittle thing breaks at the tiny point it is hit.
- A falling mesh collides as its convex hull: its hollows and dents are filled in.
- Contacts are slightly soft, so bounces are within about 0.05 of a material's bounce, and the least a thing bounces is about 0.2. Things that start inside each other are thrown apart when they are let go (the log says which).
- In a liquid scene with grey stand-ins (Water › *Colliders*), things that fall or float are drawn as stand-ins in their material's colour.

Under the hood, MuJoCo (Apache-2.0) integrates the bodies: their contacts, friction and stacking, with the time step their size needs. The gas, the water and the cloth see them as moving objects every substep.
