# Grass and plants

Grass is a patch of blades, each one simulated: it bends in the wind and in the fire's draught, springs back, parts round what moves through it, and burns. A dry field lit at one edge carries the fire across by itself, faster downwind, and is left as black stubble. It works in every kind of scene but the sky: a meadow on an empty stage, a lawn beside a pond, reeds at the water's edge.

![Meadow fire](media/gif/meadow_fire.gif "Meadow fire: a line of fire lit along the edge of a field of dry long grass, driven across it by the wind, leaving it black.")

## Making some

Add a block from the *Grass & plants* group in Create: a lawn, long grass, dry grass, wheat, reeds, or a grassy hillside (the built-in hill with long grass growing on its slopes). Each is a patch in the scene, with its settings in Properties:

- *Kind*: what grows (below).
- *Shape*: a rectangle or a disc. *Position* is the middle of the patch at the foot of the blades. *Size* is half its width and depth (a disc: its radius, the first value) and how tall the blades grow (the second). *Rotation* turns it.
- *Thickness*: how thickly it grows, times its kind's usual number of blades a square metre.
- *Dryness*: 0 is fresh and green and hard to light; 1 is dry as straw. Drier grass is paler, catches at a spark and burns out faster.
- *Burns*: whether fire can set it alight.
- *Grows on*: *The ground* puts the blades at *Position*'s height, but not where an object stands (no grass grows through a crate). *The ground and objects* drops each blade onto whatever is under it, the ground or an object that stays put (a hillside, a mound, the top of a wall), and stands it up from the slope.
- *Stiffness* and *Blade width* (advanced): times its kind's. Stiffer grass bends less in the wind and springs back faster.
- *Own colour*: a colour of its own instead of its kind's (dried toward straw by its *Dryness*).

Right-click a patch in the viewer for *Set it on fire* (a flame at one end), *Dry it out* or *Make it green*, *Kind*, and *Grow on objects too* or *on the ground only*. Its outline in the viewer shows the patch and how tall it grows.

The blades are spread evenly over the patch, in tufts a little taller or shorter than their neighbours, thinning and shortening raggedly toward its edge.

| Kind | Height | Blades a m² | What it does |
|---|---|---|---|
| Lawn | 8 cm | 4000 | Short and thick: it ripples in the wind and is flattened where things roll over it. |
| Long grass | 45 cm | 900 | Arches over at its tips and bends in the wind in waves; the usual grass for a grass fire. |
| Wheat | 90 cm | 350 | Stiff stalks with their ears, golden when ripe (dry); it sways rather than bends, and burns fast. |
| Reeds | 1.5 m | 120 | Tall, sparse and wide-bladed: they lean and whip in the wind. |

## What moves it

- **The wind.** Outside the simulation box, the scene's wind (Motion › *Wind*; in a liquid scene, Liquid › *Wind*) in gusts that roll across the field downwind, so the grass bends in moving waves. Inside the box it is the gas itself that drags it: the wind as the box carries it, the turbulence, and the fire's draught, so grass leans in toward a fire as the fire draws air in.
- **Objects.** The ground and every object keep the blades out, and things that move push them aside: a ball rolling through long grass parts it, a crate dropped on a lawn flattens what is under it, and the blades spring back up after them. Grass does not push back: it never slows what moves through it.
- **Its own spring.** Each blade bends back toward its shape at rest, from the root up, and keeps its length. It sways a little after the wind drops and settles.

## Fire

In a fire scene (or fire and liquid) grass catches where the gas is hot enough for long enough, the drier the sooner, and burns down to stubble, its char climbing from the foot as it goes, its tip glowing. As it burns it gives its fuel, heat and smoke to the gas, so the flames it makes light the grass next to it: a fire runs through a dry field by itself, faster downwind and slower to the sides, as a head fire does, and leaves a black, smouldering strip behind. Fresh grass smokes more, and takes a hotter flame for longer to light. Only the grass inside the simulation box can burn.

Grass works with the floor's own spreading fire (Spreading fire, *Ground*) or without it: on its own, the grass is the fuel.

Preset: *Meadow fire*, a torch dropped at the edge of a field of dry long grass in the wind.

## How it looks

The blades are drawn as ribbons, full width most of the way and narrowing to a point (wheat's top is its ear), in the same light as the smoke and the cloth: the key light, shadowed by the smoke and the objects, the sky, the fire and the lights in the set. Light comes through a blade lit from behind, greener than its reflection. The deeper into the grass, the less of the sky and the sun reaches a blade, so a field is darker at its foot. Each blade is a slightly different colour, drier toward its tip. Blades narrower than a pixel are drawn a pixel wide and keep only their share of the pixel, a different share each anti-aliasing pass, so a field far off averages out instead of shimmering: use a few anti-aliasing samples for grass.

Cloth and grass are drawn together, so each hides what is behind it of the other, and the fire is marched in front of both. In a liquid scene the water's march sees them, in front of the water and through it.

## Limits

- Up to 8 patches and 400,000 blades in a scene (a thick, wide patch is thinned to fit).
- The blades do not touch each other, sand, snow or cloth; grass does not push back on what moves through it, and does not slow the wind. (Things that fall, broken objects' pieces and people's and cars' parts push it aside; pressed flat by a wheel or a foot it lies down, a track that stands up again over about 40 s.)
- Under water grass is not dragged by the current yet (no kelp), and it does not get wet.
- It grows only on objects that stay put when it is set out, and does not ride on things that move.
- A cached frame keeps each blade to a 127th of its height.

Under the hood: each blade is a chain of five points, one GPU thread to a blade, stepped 300 times a second of simulated time. Drag on each point is the air's speed across the blade, squared, scaled by how much the blade weighs for its area. The blade is bent back toward its rest shape from the root up, each segment's rest direction turned with its parent's ("follow the leader"), and every segment keeps its length. Burning blades put their fuel onto the same coupling grid burning cloth uses.
