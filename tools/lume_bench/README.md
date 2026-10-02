# Lume benchmark

Measures Lume, Blackbody's path-traced lighting, against ground truth: the same scenes rendered by **Mitsuba 3**, a research path tracer, on the GPU with OptiX. Mitsuba renders them with **Lume's own material** (written as a Mitsuba BSDF in `mitsuba_side.py`), so any difference is in how Lume carries the light, not in the material.

## Setup

Mitsuba runs in its own Python environment, so Blackbody's stays as it is:

```
python -m venv mitsuba-env
mitsuba-env\Scripts\python -m pip install mitsuba numpy pillow
```

## Run

From the repository, in Blackbody's environment:

```
python tools/lume_bench/run.py --mitsuba-python mitsuba-env\Scripts\python.exe
```

It renders every scene in `scenes.py` with Lume (`lume_side.py`) and with Mitsuba (`mitsuba_side.py`: a 262,144-sample reference, a few minutes, plus timed renders with each of its samplers), then writes `out/lume_bench/DATE/report.md` and a picture per scene: Lume, the reference, and where they differ (green equal, red Lume darker, blue Lume brighter, full colour at 20%). `--skip-render` reports on pictures already there; name scenes to run only those.

## The scenes

| Scene | What it checks |
|---|---|
| furnace | A white ball in a uniform sky: energy kept, the sky's light, multiple bounces |
| cornell | An open box with red and green sides and a lamp inside: light bouncing from wall to wall |
| cornell_direct, cornell_2 | The same with one bounce (direct light only) and two: which bounce a difference is in |
| glossy | Rough and smooth metal and plastic balls under a distant lamp: highlights, reflections, and the lamp a smooth metal ball throws onto the floor |
| glossy_direct, glossy_2 | The same with one bounce and two |
| glass | A glass ball beside a block under a lamp: refraction, its shadow, the light it focuses |
| glass_sky | The glass ball under the sky alone: the sky's caustic |

## The measures

- **Bias**: Lume's mean brightness at its most samples over the reference's. 1.000 is exact.
- **Error**: relative mean squared error, `mean((x - ref)^2 / (ref^2 + 0.01))`, at each sample count, for both renderers.
- **Equal time**: each renderer's error after the same render time (midway, on a log scale, between Lume's times for 4 and 1024 samples), against Mitsuba with each of its samplers: *independent* (plain random numbers, its default), *stratified*, *multijitter* (multi-jittered) and *ldsampler* (low discrepancy). The stratified ones cut Mitsuba's error on a set lit straight by a lamp fifteenfold, so the table gives both: against plain random numbers, and against whichever of Mitsuba's samplers does best on that scene.

Left out of the measures: the lamps themselves and two pixels round them (Lume does not draw lamps for the camera; Mitsuba does), and the row of pixels the horizon crosses (the reference's ground is a square 4 km across, Lume's has no end: a fraction of a pixel of sky, geometry, not light).

The reference is Mitsuba with 262,144 multi-jittered samples per pixel. Its ground is a 100 m square with four long pieces round it out to 2 km: one rectangle 4 km across puts the hits near its middle a fraction of a millimetre off the ground, and shadow rays from there are wrongly blocked (the reference was up to 14% dark under a low lamp). Lume's scenes use flat materials with no pattern at all (a plain surface in Blackbody is otherwise a little uneven, ±6%, which the reference does not have).

## Results, 2026-10-02 (RTX 3090, timings the fastest of three)

| Scene | Bias | Error, Lume at 1024 samples | Equal-time error, Lume / Mitsuba (random) | Lume / Mitsuba (its best sampler) |
|---|---|---|---|---|
| furnace | 0.9999 | 0.000002 | 0.21x | 1.64x (multijitter) |
| cornell | 0.9995 | 0.00080 | 0.54x | 0.68x (multijitter) |
| cornell_direct | 0.9995 | 0.000001 | 0.005x | 0.54x (multijitter) |
| cornell_2 | 0.9995 | 0.00030 | 0.25x | 0.37x (multijitter) |
| glossy | 0.9993 | 0.00034 | 0.68x | 0.68x (independent) |
| glossy_direct | 0.9995 | 0.000003 | 0.07x | 0.47x (multijitter) |
| glossy_2 | 0.9994 | 0.00019 | 0.43x | 0.43x (independent) |
| glass | 0.9985 | 0.00043 | 0.20x | 0.22x (stratified) |
| glass_sky | 0.9994 | 0.000014 | 0.37x | 0.96x (multijitter) |

Below 1x Lume is closer to the truth than Mitsuba after the same time. Against Mitsuba's best sampler for each scene, Lume is ahead on every scene but the furnace, a white ball under an even sky that both render in a few milliseconds: there Lume's error for its samples is Mitsuba's, but setting up a frame (about 3 ms) counts. Where Lume is furthest ahead, it traces the light the lamp focuses through glass or throws off smooth metal from the lamp itself (a camera's path finds it rarely, in fireflies), and its random numbers (Owen-scrambled Sobol) spread its paths as evenly as Mitsuba's stratified samplers do, at less cost.

What is left:

- **Caustics seen through their glass**: light a glass ball focuses onto the floor, seen through the same ball, comes straight through the glass, unfocused (the glass scene's ball is up to a few percent off where its caustic shows through it; 0.15% over the picture).
- **Speed in general**: Mitsuba traces with the GPU's ray-tracing cores (OptiX); Lume traces in compute shaders, plain shapes exactly and meshes and matter by marching.

What the benchmark has caught and fixed so far. In Lume: Russian roulette from the second bounce (twice the noise on shiny metal; now from the fourth), caustics not traced (2.6% dark with a glass ball; now traced from the lights, through glass and off smooth metal), sky caustics leaking up through the ground, the last bounce dropping the sky's share of light (all scenes up to 8x too dark in sky-lit shade), a path clamp that removed most of an HDRI sun's bounced light, lamps treated as soft points (3% dark near them), rays off a ball's underside slipping under the floor (bright contact rings), ray offsets of a pixel or two putting surfaces that much nearer the lamps (up to 3% bright), boxes shaded with normals rounded over a pixel at their edges (a pixel too dark or bright along every edge and corner), and a lamp's highlight in glass left out. In the benchmark itself: Mitsuba's ground one rectangle 4 km across (the reference up to 14% dark under a low lamp: the 0.8% Lume seemed too bright on the glossy scene was this), Blackbody's plain surfaces a little uneven (a fixed blotchy difference no number of samples took away), the reference too noisy at 16,384 samples for errors this small, and comparing only against Mitsuba's plain random sampler (its stratified ones are up to fifteen times closer on direct light).
