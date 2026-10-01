# Changelog

Notable changes to Blackbody. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Layers: several effects in one shot (a campfire in front, a waterfall behind), each its own simulation with its own
  box and scale, composited back to front in the viewer, renders and the command line.
- Roto: shapes drawn around what is in front of the effect, keyed over the shot, hide the effect behind them.
- Work view: an orbit camera of your own for building the scene; the shot's camera is drawn as a frustum.
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
