# Security

Blackbody opens files that often come from other people: projects and presets,
footage, image sequences, USD scenes, OpenVDB volumes, OBJ and STL meshes, HDR
and EXR images, OCIO configs, `.chan` camera tracks, and simulation caches on
shared render-farm folders. Bugs in how it reads those files, or in where it
writes renders and caches, can be security issues.

## What to report privately

- Any file Blackbody opens (a project, preset, footage, USD, VDB, mesh, image,
  OCIO config, camera track or disk cache) that can make it run code or write
  or delete files outside the output and cache locations you chose.
- A render or `simulate` job that overwrites or removes files it shouldn't. This
  includes a cache folder that a different scene wrote.
- Anything that sends data off the machine. Blackbody shouldn't do that at all,
  because it makes no network connections.
- Memory corruption that a crafted file can trigger in the libraries Blackbody
  uses to read it (FFmpeg through PyAV, OpenEXR, OpenColorIO, OpenUSD).

Crashes on broken files, wrong renders and UI bugs can go in public issues.

## How to report

Use GitHub's private vulnerability reporting: open the **Security** tab of the
repository and choose **Report a vulnerability**. Include your OS, your GPU
(the output of `python -m blackbody info`), the Blackbody version or commit,
steps to reproduce, and the file that triggers it if you can share one. Please
don't open a public issue until a fix is released.

Expect a reply within a week. Fixes go into the next commit on `main`, and the
changelog credits the reporter unless they'd rather not be named.

## Supported versions

Only the latest commit on `main` receives fixes.
