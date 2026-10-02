# Fabric and burning cloth

*+ Fabric* (in the Outliner) adds cloth: a curtain, a flag, a banner, a sheet that falls, or any mesh (a garment, a tablecloth modelled over its table, a tarp). It hangs from what holds it (*Held by*: its top edge, one side, its top corners, four corners, or nothing), drapes over colliders and the ground, folds onto itself without passing through, and blows in the fire's own air and the wind, slowing the air it hangs in. Keyframe its *Position* to move what holds it (a curtain being drawn, a flag carried); *Let go at* drops it. A scene holds up to 16 fabrics, in fire, liquid and fire-and-liquid scenes alike. However fine its *Detail*, hanging cloth does not stretch past its length and a stiff fabric drapes as stiffly as it is measured to.

![Burning curtain](media/gif/fabric_curtain.gif "Burning curtain: a lighter at the hem of a cotton curtain; the flame climbs it, chars it and burns through.")

## Gathered panels

*Gathered* hangs a panel in pleats, as a curtain on its rail: 1 is flat, curtains are 1.5 to 2.5 times their width. The pleats are uneven, as hung cloth is, and fall open toward the hem.

## Materials

*Material* is a real fabric with its measured weight, stretch and bending stiffness: cotton, linen, silk, chiffon, wool, denim, canvas, velvet, polyester and nylon. Silk and chiffon flow and crumple finely, canvas and denim fold in big rounded folds. *Weight*, *Stretch* and *Stiffness* scale the material's.

## Burning

*Burnable* fabric heats in the gas around it and in the fire's radiation (thin cloth quickly, heavy cloth slowly; cloth beside a fire, out of its flames, toasts in the heat it radiates), catches at its ignition temperature, and burns: it gives off fuel, heat and smoke that the fire burns, and burns the way real cloth does: a halo of brown toasting ahead of the flame, a thin bright line of glowing fibre where it catches, black char behind with embers smouldering in it, greying to ash, then a dull glowing rim as it crumbles into ragged holes that spread, until it falls apart. The bands are as sharp as the picture, whatever the cloth's *Detail*. Char shrinks and curls. Burnt-through cloth breaks into flakes of ash and embers: glowing as they leave the flame, then black char and grey ash, tumbling up in the fire's plume or fluttering down. Fire climbs a curtain far faster than it creeps down or sideways, as the hot gas rises along it. Wool puts itself out once the flame against it is gone; polyester and nylon shrink away from the heat and melt through before they burn. *Flammability* scales how readily it catches and how much it feeds the fire (flame-retardant cloth is about 0.3).

![Flag in smoky wind](media/gif/flag_wind.gif "Flag in smoky wind: nylon streaming and fluttering in the wind as a fire's smoke rolls past.")

## The look of cloth

It is shaded as cloth, in the same light as the smoke: the soft sheen of its fibres, a highlight along the threads for silk, satin and nylon, light from behind showing through thin cloth (a curtain glows with a fire behind it), the weave where it is big enough to see, and its colour, scorched brown by heat before it chars. It goes behind the fire in front of it and behind the colliders in the shot.

## Shadows

It casts shadows, and is shadowed: it shades the smoke behind it, other cloth and the ground and footage under it from the key light, the sky, the fire and the lights in the set (thin cloth lets some light through), and the smoke, other cloth and the objects around it shade the fire's light on it.

## Sand, snow and mud

Sand, snow, mud, jelly and clay cannot pass through cloth, from either side. A sheet held at its corners catches sand poured onto it and sags under its weight; a sheet dropped onto a heap drapes over it. See [what moves matter](matter.md#what-moves-it).

## In water

In water (liquid and fire-and-liquid scenes) it is carried by the water's flow and drag, and soaks. Dry cloth floats on the air in its weave; cotton and linen soak through in a second or so, wool and synthetics shed water for a few seconds first; soaked, it sinks slowly by what its fibres weigh more than water. It is seen in front of the water, floating on it, and through it under the surface. *Wet at start* hangs it wet in any scene (1 is dripping wet, a towel just out of the water).

## Wet cloth

Wet cloth holds water as the real fabric does: dripping wet, cotton holds about twice its weight, wool and velvet more, polyester and nylon little. Above the waterline the water wicks up into dry cloth, a sharp wet front creeping a few centimetres in a minute. Out of the water, what the fabric cannot keep (about half of it, for cotton) runs down the cloth to the hem and the bottoms of its folds and sags, and drips off them: a towel lifted out streams at first, then drips more and more slowly, and stays damp. The drops fall, carried by the wind, and land on the ground, the objects, or back in the water. Wet cloth is heavier, so it hangs limper, in smaller folds, and it clings to what it lies on. It is darker, glossier (dripping wet, it glistens with the water on it) and more see-through than dry.

## Wet cloth in the heat

Wet cloth in the heat (the gas and the fire's radiation) holds at 100 °C while its water boils off, drying first where it is hottest; it cannot catch until a spot is dry, and a soaked burning spot goes out. The steam goes into the gas: it takes its heat from the air around the cloth, rises and, in cool air, condenses into white steam (Shading › *Air temperature* and *Air humidity*; steam made inside a fire's hot plume stays clear, as it does in life), and steam thick enough starves the flames.

![Wet and dry towels over a fire](media/gif/wet_towels.gif "Wet and dry towels over a fire: the dry one burns away; the soaked one steams and holds at 100 °C.")

## Output

EXRs get a `fabric` layer (the cloth on its own); the beauty holds it under the fire. A render to `.obj` (one per frame) or `.usd` writes the fabric as meshes, with its burnt-through holes left out and a `burn` value per vertex (in a liquid scene, beside the liquid's surface: `name.fabric.####.obj` or `name.fabric.usd`).
