# Sand, snow and mud

Sand, snow, mud, jelly and clay are *matter*: a body of it is made of tens or hundreds of thousands of particles that each carry how they have been squeezed and stretched, so it piles, pours, packs, slumps, wobbles and squashes as the real thing does. It works in every kind of scene but the sky: sand poured from a hopper on an empty stage, a snowball thrown at a wall, a crate dragged through mud.

![Sand from a hopper](media/gif/sand_hopper.gif "Sand from a hopper: forty litres of dry sand run out through a hole in the bottom of a hopper and heap up round its legs.")

## Making some

Add a block from the *Sand, snow & mud* group in Create: a sand pile, a sand pour, a column of sand let go, a snowball, a snow drift, mud, a block of jelly or a lump of clay. Each is a matter object in the scene, with its settings in Properties:

- *Made of*: what it is (below).
- *Shape*, *Position*, *Size*, *Rotation*: the body of it at the start: a box, a ball, a cylinder or a pile (a cone standing on its base).
- *Thrown at*: how fast it moves as it starts (a thrown snowball).
- *Let go at*: it is held where it is until then (a column of sand let go to collapse).
- *Pours*: instead of a body of it, a stream poured from a nozzle at *Position* (its *Size*'s first value is the nozzle's radius), *Thrown at* the speed it leaves at, *Flow* in litres a second, from *Pours from* until *Pours until*.
- *Own colour*: a colour of its own in place of its material's.

## What it is made of

| Made of | What it does |
|---|---|
| Sand | Dry grains: it pours, slides, and piles at its angle of repose (34°). Pulled apart, it falls apart. |
| Wet sand | Holds a steeper heap (38°) and a little stretch, so it clumps and holds a cut edge. |
| Snow | Fresh snow, 400 kg/m³: squeezed, it packs and gets harder; stretched, it breaks into lumps. |
| Packing snow | Wet snow, 600 kg/m³, for snowballs: it splats where it hits and breaks up as it falls. |
| Mud | Flows like a thick liquid until it is thin enough to hold itself up (its yield stress, 400 Pa), then stops. |
| Jelly | Springs back from anything that squashes it, keeping its volume: it bounces and wobbles. It is clear and coloured. |
| Clay | Squashes and stays squashed: it holds a shape until it is pushed harder than 20 kPa. |

## What moves it

- **Gravity, the ground and the box.** It rests on the ground and against the box's closed sides. Through an open side, the top, or the bottom of a box without ground, it leaves the simulation.
- **Objects.** It piles against objects that stay put, and keyframed ones plough through it (a crate dragged through mud pushes up a bow wave). Things that fall land on it and it holds them up as its strength and their weight say: a wooden crate rests on top of sand where a steel ball of the same size sinks into it, and jelly throws a dropped ball back up. Matter and the falling objects push each other every step.
- **Blasts.** An explosion (an emitter's *Blast*) throws it away from where it goes off, the surface hardest: a heap of sand near a charge is blown flat.

## How it looks

On the stage, matter is drawn in its material's look and lit like the objects: the key light with soft shadows (it shades itself and the floor, and the objects shade it), the sky, the fire and the lights in the set. Sand and snow glint where a grain catches the sun, snow lets the light into its shadows, mud is wet and glossy, and jelly is clear and coloured: you see the set through it, bent. In your footage it goes in as CG, its shadows on the real ground. It hides the fire and the liquid behind it.

![Ball dropped on jelly](media/gif/jelly_ball.gif "Ball dropped on jelly: a steel ball squashes a block of jelly, is thrown back up and bounces.")

## Detail and speed

- *Matter detail* (Domain, advanced) is how fine it is simulated: grid nodes across the box's longest side, eight particles to a cell. The default, 128, gives a 2 m box 1.6 cm cells. Twice as fine is eight times the particles.
- *Most matter particles* caps how many there are; each takes 128 bytes of GPU memory.
- Its steps are as short as its stiffest material and its fastest particle need: about a hundred a frame for sand, more for packed snow. A heap of 40 litres of sand at the default detail (about 170,000 particles) takes about a tenth of a second a frame.

Presets: *Sand from a hopper*, *Snowballs at a wall*, *Ball dropped on jelly*, *Crate through mud*.

![Crate through mud](media/gif/mud_drag.gif "Crate through mud: a crate dragged through a bed of mud ploughs a trench and leaves ridges.")

## Limits

- The smoke, the water and the cloth do not meet it yet: smoke goes through a sand pile, and water does not wet sand.
- Broken objects' pieces do not push it, and it does not burn.
- A body of matter is a box, a ball, a cylinder or a cone; it cannot fill a mesh yet.
- Up to 15 materials (or colours of them) in a scene at once.
- Snow and mud do not stick to walls: a snowball splats and falls rather than leaving a mark.

Under the hood: the material point method (MLS-MPM, Hu et al. 2018) on the GPU, with sand as Drucker–Prager plasticity (Klár et al. 2016), snow after Stomakhin et al. (2013), mud and clay as von Mises plasticity (mud relaxing toward it over time), and jelly as a neo-Hookean solid. Sand's friction angle is set from the angle of repose asked for, measured on poured heaps. Particle-to-grid sums are fixed-point integers, so a simulation is the same on any GPU, every time.
