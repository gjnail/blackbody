# Docs media

These scripts make the videos, GIFs and screen recordings in `docs/media`, which the README, the guides and the website use. Everything in them is rendered by Blackbody itself. The working files go to `out/docs_media/`, which is not committed.

## Feature clips

Each clip in `clips.py` is a preset, the moment of it to start at, a gentle camera move, and a procedural backplate (`plates.py`, `plates_fast.py`: night dirt, concrete, pebbles, or sky).

```bash
python tools/docs_media/render_clip.py campfire --test   # three draft frames, to check the framing
python tools/docs_media/render_clip.py campfire          # the 1280x720 frames
python tools/docs_media/encode_media.py clip campfire    # docs/media/video/campfire.mp4 (+ .jpg poster) and docs/media/gif/campfire.gif
```

Use the app's environment (`.venv`) for `render_clip.py` and `record_ui.py`. `encode_media.py` needs FFmpeg on the PATH and Pillow.

## Screen recordings

`record_ui.py` runs the app offscreen and plays real mouse drags, clicks and key presses into it while grabbing the window. It uses throwaway settings, so your own Blackbody settings are left alone. The capture runs at whatever rate the window can be grabbed (often 8 to 15 fps under GPU load), so `compose_ui.py` lays the frames out on a steady 24 fps clock and draws the cursor, clicks and captions in from the logged input.

```bash
python tools/make_test_footage.py out/practice_plate.mp4   # the plate some recordings use
python tools/docs_media/record_ui.py library place scrub views track render
python tools/docs_media/compose_ui.py place
python tools/docs_media/encode_media.py ui place            # docs/media/video/ui-place.mp4 and docs/media/gif/ui-place.gif
```

## Sizes

MP4s are H.264 at CRF 23 (CRF 22 for the UI), and are about 1 MB for 5 seconds. GIFs are 480 px wide at 12 fps (960 px and 10 fps for the UI), and `encode_media.py` shrinks a GIF until it fits in 4 MB. Keep new media about this size, because every file stays in the repository's history.
