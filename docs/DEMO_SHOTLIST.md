# Demo video: shot list and commands

Target: 40–50 seconds, under 25 MB, FPS counter visible in every frame.

The clip is produced by the tool itself (`--record`), not by screen-capture
software. The HUD in the recording is the same overlay the app draws live, so
the frame rate shown in the video is the frame rate that was actually measured.

---

## 1. Before you hit record

```bash
python models/download_weights.py
python -m src.main --source 0 --max-frames 120
```

Use that throwaway run to check three things, because re-shooting is annoying:

- **Framing.** Sit so your head and shoulders fill roughly the left third. Objects
  get held up in the right two-thirds, about 40–60 cm from the lens.
- **Light.** One diffuse source in front of you. Backlighting (a window behind
  you) is the single fastest way to make the detector look bad — the camera
  stops down, everything else goes dark, and recall collapses.
- **Clutter.** Clear the background. Every stray object in frame is one more box
  competing for attention with the thing you are demonstrating.

Have these within reach, in this order: **cup**, **cell phone**, **bottle**,
**book**, **scissors**, **laptop** (already open on the desk), **keyboard**,
**mouse**, **chair** (behind you), **backpack**, **teddy bear** or any soft toy,
**apple** or **banana**.

## 2. Shot list (~45 s)

| Time | Shot | What it demonstrates | Direction |
|---|---|---|---|
| 0:00–0:04 | Empty desk, you in frame | Baseline; HUD legible | Sit still. `person` should be the only box. Let the rolling FPS settle. |
| 0:04–0:10 | Hold up a **cup**, then a **bottle** | Single-object detection, confidence readout | One at a time, steady for ~3 s each. Hold them still enough to read the score. |
| 0:10–0:16 | Put both down next to the **laptop** and **keyboard** | **Multi-object**: 4–6 boxes at once | Lean back so all objects are visible together. This is the money shot. |
| 0:16–0:22 | Pick up **cell phone**, hold it near the cup, then in front of it | **Occlusion**: box shrinks, confidence drops, recovers | Move slowly. The point is that the score visibly reacts, not that it never fails. |
| 0:22–0:28 | Move the phone from arm's length to close to the lens | **Scale**: small objects are harder | Expect the score to climb as it gets bigger. That is the honest behaviour and worth showing. |
| 0:28–0:34 | Wave the cup quickly left-to-right, then stop | **Motion blur**: detection drops during the sweep and returns | Do it twice. Do not cut this — the limitations section in the README claims it, so show it. |
| 0:34–0:40 | Hold up **book**, **scissors**, **teddy bear**, **apple** in quick succession | Class breadth | ~1.5 s each. Enough to read the label. |
| 0:40–0:45 | Sit back, hands down, whole scene in frame | Steady state; final FPS reading | Hold still for a clean last frame. |

Keeping the FPS counter visible needs no effort — it is drawn top-left on every
frame and the recording includes the overlay. Do not cover the top-left corner
with an object.

## 3. Record it

```bash
python -m src.main --source 0 --record results/demo_raw.mp4 --max-frames 1350
```

Notes on that command:

- `--max-frames 1350` is a stop-watch: at the ~30 fps the camera delivers, that
  is roughly 45 seconds. Adjust if your camera runs slower. You can also just
  press `q` when you are done.
- The frame rate written into the file is **measured**, not assumed. The writer
  stays closed for the first 30 frames while the rolling FPS settles, then opens
  at that rate, so the clip plays back at real speed instead of the
  slow-motion-or-chipmunk effect you get from hardcoding 30 fps.
- A red `REC` dot appears top-right once recording actually starts, so there is
  no ambiguity about whether a take was captured.
- The first ~1 second is not in the file (that is the calibration window). Start
  your first shot after the dot appears.

For a version with fewer distractions, add a class filter:

```bash
python -m src.main --source 0 --record results/demo_raw.mp4 \
  --classes person cup bottle laptop keyboard mouse "cell phone" book scissors
```

## 4. Compress it under 25 MB

`--record` writes mp4v, which is what every stock OpenCV wheel can encode. It is
not efficient. One ffmpeg pass to H.264 typically takes a ~45 s 640×480 clip from
roughly 40–60 MB down to 3–6 MB:

```bash
ffmpeg -i results/demo_raw.mp4 -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -an results/demo.mp4
```

- `-crf 26` is the quality knob: lower is better and bigger. 23 is visually
  transparent; 28 starts to smear the text on the labels. If the result is still
  over 25 MB, raise it to 28 before you touch the resolution.
- `-pix_fmt yuv420p` is not optional if you want the file to play in browsers,
  Slack previews and QuickTime.
- `-an` drops the (non-existent) audio track and stops some players complaining.

Check the size:

```bash
ls -lh results/demo.mp4
```

**ffmpeg is not installed by default.** Install it with whichever applies:

```bash
winget install Gyan.FFmpeg          # Windows
brew install ffmpeg                 # macOS
sudo apt install ffmpeg             # Debian/Ubuntu
```

If you would rather not install it, record at a lower resolution instead — a
45 s clip at 480×360 usually lands under 25 MB straight out of `--record`:

```bash
python -m src.main --source 0 --width 480 --height 360 --record results/demo.mp4
```

## 5. Publish

The video is gitignored on purpose (`*.mp4`) — binaries do not belong in a git
history. Upload `results/demo.mp4` to the GitHub release page, a repository
issue, or YouTube/Drive, and put the link in the README where it says
`DEMO_VIDEO_URL`.
