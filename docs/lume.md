# Lume lighting

Lume is Blackbody's path-traced lighting engine. It lights the set drawn in CG (the floor, the objects, broken pieces, sand, snow and mud, ropes) by tracing light as it really travels, instead of adding up a few direct lights. Turn it on in Properties › *Lume* › *Lighting engine: Lume (path traced)*. It works in every kind of scene, with or without footage, and the fire, the smoke and the liquids are drawn over it as before.

<img src="media/img/lume-classic.jpg" width="49%" alt="A wooden shed with a fire inside, lit by the classic engine" title="Classic: direct light and soft shadows; the shed's inner walls flat brown"> <img src="media/img/lume-path.jpg" width="49%" alt="The same shed path traced with Lume" title="Lume: the fire's light on the inner walls and the floor, bounced round inside the shed">

*Shed on fire* at 6 s: the classic engine (left) and Lume (right), the fire's light bouncing round inside the shed.

## What it adds

- **Light that bounces.** Light reaches a surface after bouncing off others: a red brick wall tints the floor in front of it, the shaded side of a crate is lit by the sunny ground beside it, an object in a room is lit by the walls round it.
- **Shadows as soft as each light is big.** The key light's shadow sharpens where an object touches the ground and softens as it falls away, as the sun's disc makes it. Every fire light, glowing patch of hot metal and lamp casts its own shadow, through the objects and through the smoke.
- **The HDRI's own sun.** With an *Environment (HDRI)*, light is aimed at its bright parts, so a small sun in the HDRI lights the set and casts sharp shadows at once (the classic engine only takes the HDRI's average colour).
- **Glass, ice and jelly that bend light.** A ray goes in, bounces inside while it cannot get out, and leaves through the far side, tinted by how far it went through. Clear things refract what is behind them and reflect the sky by the true Fresnel term.
- **Physically based highlights.** A GGX microfacet highlight with height-correlated shadowing and Fresnel at the facet's angle. Rough floors no longer flare toward the horizon.
- **The fire lighting the set.** Each of the fire's point lights is picked by how much light it brings to a surface, and its shadow is traced through the objects and the smoke.

## Settings

| Setting | What it does |
|---|---|
| Lighting engine | *Classic (fast)*: direct light with soft shadows. *Lume (path traced)*: the light traced as it travels. |
| Bounces | How many times light bounces from surface to surface: 3–4 for most shots, more for rooms and glass. |
| Samples (final) | Light paths per pixel in final renders: 128 for a preview, 256–1024 for hero shots. |
| Samples (viewer) | Light paths per pixel the viewer gathers once you stop moving or editing. |
| Denoise | Smooths away the grain that is left, keeping edges, textures and shadow edges. |
| Clamp bright paths | (Advanced) Caps a bounce's light at this many times the sky's brightness, to clear stray sparkles sooner. 0 keeps every path exact. |

## In the viewer

While you drag or play, Lume shows one light path per pixel (denoised). Once you stop, it keeps adding paths, two at a time, until *Samples (viewer)* is reached. The stats overlay shows how far it has got (*Lume · 24/64 light paths per pixel · sharpening*). Any edit starts it afresh.

## Accuracy

Lume is measured against ground truth: the same scenes rendered by Mitsuba 3, a research path tracer, with Lume's own material (`tools/lume_bench`). On those scenes it is within about 1% of the truth in mean brightness (0.25% to 0.85% on the rooms, metal and plastic), except the light a glass ball focuses (a caustic), which is not yet traced: shadow rays pass through glass by its Fresnel losses and tint, straight through, which is exact for a pane but misses a curved piece's focused light. See `tools/lume_bench/README.md` for the scenes, the measures and the latest results.

## Speed

Plain spheres, boxes and cylinders, the floor and broken pieces are hit exactly by each ray; meshes, hollow things and sand, snow and mud are found by marching toward them. On an RTX 3090, a simple set at 1920x1080 takes a few tenths of a second for 64 paths per pixel with 4 bounces; against Mitsuba with the GPU's ray-tracing cores, Lume has 1.2–5 times its error after the same time. A frame with fire light and smoke shadows takes a few times longer. Final renders trace their samples in passes of four paths per pixel (the viewer one), submitted a few at a time, so long renders do not stall the GPU.

## How it works

Every pass traces a path per pixel from the camera (four in a final render). At each surface, Lume:

1. takes the light coming straight from each light (next-event estimation): a direction in the sun's disc; one fire light and one hot-matter light, picked by how much light each brings; every lamp; and a direction in the HDRI picked by its brightness (weighed against finding it by bouncing: multiple importance sampling, power heuristic);
2. bounces on, by the material (a GGX highlight sampled by its visible normals, or diffuse by the cosine), or through glass by its Fresnel term;
3. ends the path by Russian roulette once it carries little light.

The passes are added up, with each pixel's first surface (albedo, normal, distance) and how noisy it still is. The denoiser divides out the albedo so textures stay sharp, filters the light with an edge-avoiding à-trous wavelet filter (five passes, guided by facing, distance and noise), and multiplies the albedo back.

The engine is in `blackbody/engine/lume.py`, `blackbody/engine/wgsl/lume.wgsl` and `blackbody/engine/wgsl/lume_denoise.wgsl`.

## Coming next

Lume grows in stages. Next: a physical sky and sun, area lights and light-profile (IES) files in physical units; then multiple scattering in smoke and clouds, with the fire as a volumetric light source; caustics and refraction with full bounce light in water; per-light passes for compositing and matching the light in your footage; and a spectral mode that traces each wavelength, as a blackbody emits it.
