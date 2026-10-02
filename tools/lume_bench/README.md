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

It renders every scene in `scenes.py` with Lume (`lume_side.py`) and with Mitsuba (`mitsuba_side.py`, a 16384-sample reference plus timed renders), then writes `out/lume_bench/DATE/report.md` and a picture per scene: Lume, the reference, and where they differ (green equal, red Lume darker, blue Lume brighter, full colour at 20%). `--skip-render` reports on pictures already there; name scenes to run only those.

## The scenes

| Scene | What it checks |
|---|---|
| furnace | A white ball in a uniform sky: energy kept, the sky's light, multiple bounces |
| cornell | An open box with red and green sides and a lamp inside: light bouncing from wall to wall |
| cornell_direct, cornell_2 | The same with one bounce (direct light only) and two: which bounce a difference is in |
| glossy | Rough and smooth metal and plastic balls under a distant lamp: highlights and reflections |
| glass | A glass ball beside a block under a lamp: refraction, its shadow, the light it focuses |
| glass_sky | The glass ball under the sky alone: the sky's caustic |

## The measures

- **Bias**: Lume's mean brightness at its most samples over the reference's. 1.000 is exact.
- **Error**: relative mean squared error, `mean((x - ref)^2 / (ref^2 + 0.01))`, at each sample count, for both renderers.
- **Equal time**: each renderer's error after the same render time, and how long each takes to get under 0.01.

The lamps themselves are left out of the measures (Lume does not draw them for the camera; Mitsuba does).

## Results, 2026-10-02 (RTX 3090, timings the fastest of three)

| Scene | Bias | Error, Lume at 1024 samples | Equal-time error, Lume / Mitsuba |
|---|---|---|---|
| furnace | 1.0002 | 0.00004 | (both under a millisecond) |
| cornell | 0.9976 | 0.0021 | 1.2x |
| cornell_direct | 0.9953 | 0.0010 | 0.6x |
| cornell_2 | 0.9966 | 0.0022 | 0.9x |
| glossy | 1.0081 | 0.0011 | 2.5x |
| glass | 0.9975 | 0.0007 | 0.2x |
| glass_sky | 1.0004 | 0.0005 | (both under a few milliseconds) |

Below 1x Lume is closer to the truth than Mitsuba after the same time: with a glass ball five times closer, because Lume traces the caustic from the lamp (light tracing) where Mitsuba waits for camera paths to find the lamp through the glass.

What is left:

- **Caustics seen through their glass**: light a glass ball focuses onto the floor, seen through the same ball, is not traced (a specular-diffuse-specular path): the ball itself is up to 3% dark where the caustic shows through it.
- **Shiny metal**: at equal time, 2.5x Mitsuba's error on the glossy scene; a lamp seen in a smooth metal ball is the noisiest part.
- **Speed in general**: Mitsuba traces with the GPU's ray-tracing cores (OptiX); Lume traces in compute shaders, plain shapes exactly and meshes and matter by marching.

What the benchmark has caught and fixed so far: Russian roulette from the second bounce (twice the noise on shiny metal; now from the fourth), caustics not traced (2.6% dark with a glass ball; now traced from the lights), sky caustics leaking up through the ground, the last bounce dropping the sky's share of light (all scenes up to 8x too dark in sky-lit shade), a path clamp that removed most of an HDRI sun's bounced light, lamps treated as soft points (3% dark near them), rays off a ball's underside slipping under the floor (bright contact rings), and ray offsets of a pixel or two putting surfaces that much nearer the lamps (up to 3% bright).
