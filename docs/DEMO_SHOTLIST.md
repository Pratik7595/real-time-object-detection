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
python -m src.main --source 0 --record results/demo.mp4 --max-frames 702
```

Notes on that command:

- `--max-frames 702` is a stop-watch, sized from a measured test take on this
  machine: the camera delivered **15.6 fps**, so 45 s ≈ 702 frames. **Check
  your own rate first** — run `python -m src.main --source 0 --max-frames 150`
  and read the `fps mean` line, then use `rate × 45`. A camera's nominal 30 fps
  halves in dim light, and sizing this from the nominal figure gives you a
  22-second clip that stops in the middle of your shot list.
- You can ignore the frame count entirely and just press `q` when you are done.
  That is the safer option for a first take.
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

## 4. Compress it — probably not needed

**Measured on this machine, compression is unnecessary.** A test take wrote
0.308 MB per second of video at 640×480, which projects to:

| Take length | Projected size | Under the 25 MB target? |
|---|---|---|
| 40 s | ~12.3 MB | yes |
| 45 s | ~13.8 MB | yes |
| 50 s | ~15.4 MB | yes |

So `results/demo.mp4` can go straight to the release asset with no ffmpeg step,
which is convenient because ffmpeg is not installed here. Check the size the
tool prints at the end of the run before assuming it holds for your take — a
busier scene compresses worse.

Note also that recording costs frame rate: the `render` stage went from 0.26 ms
to 7.65 ms with the writer active on live camera frames, taking the run from
15.0 to 14.1 FPS. The counter in your clip will read slightly lower than the
same setup not recording. That is honest and worth leaving alone rather than
hiding.

### If you do need to compress

One ffmpeg pass to H.264, which is far more efficient than the mp4v that stock
OpenCV wheels can write:

```bash
ffmpeg -i results/demo.mp4 -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -an results/demo_small.mp4
```

Upload `demo_small.mp4` in that case.

- `-crf 26` is the quality knob: lower is better and bigger. 23 is visually
  transparent; 28 starts to smear the text on the labels. If the result is still
  over 25 MB, raise it to 28 before you touch the resolution.
- `-pix_fmt yuv420p` is not optional if you want the file to play in browsers,
  Slack previews and QuickTime.
- `-an` drops the (non-existent) audio track and stops some players complaining.

**ffmpeg is not installed by default.** Install it with whichever applies:

```bash
winget install Gyan.FFmpeg          # Windows
brew install ffmpeg                 # macOS
sudo apt install ffmpeg             # Debian/Ubuntu
```

If you would rather not install it and your take really did come out oversized,
record the next one smaller instead — `--width 480 --height 360` roughly halves
the byte rate:

```bash
python -m src.main --source 0 --width 480 --height 360 --record results/demo.mp4
```

## 5. Publish it as a release asset

The video is gitignored on purpose (`*.mp4`) — binaries do not belong in a git
history. Attach it to a GitHub release instead. A release asset is tied to a tag
on the repository rather than to a comment thread, which makes it the durable
option: it survives repo transfers, it is obvious where it came from, and the
URL is readable rather than a UUID.

The trade-off, accepted deliberately: the link **downloads** the file rather
than playing it in the page. The screenshot at the top of the README carries the
"what does this look like" job, so the video does not need to autoplay to earn
its place.

### Web UI

1. Push the repo to GitHub if you have not already.
2. Go to **Releases → Draft a new release**.
3. **Choose a tag** → type `v1.0` → *Create new tag on publish*.
4. Title it `v1.0 — real-time detection demo`. A one-line body is plenty:
   the hardware, the model, and the measured frame rate.
5. Drag `results/demo.mp4` into the **Attach binaries** box at the bottom. Wait
   for the upload bar to finish before publishing.
6. **Publish release.**
7. Right-click the attached file → copy link. It looks like:
   `https://github.com/<you>/real-time-object-detection/releases/download/v1.0/demo.mp4`
8. In `README.md`, replace `DEMO_VIDEO_URL` with that link. Commit and push.

### Or, with the `gh` CLI

One command, same result:

```bash
gh release create v1.0 results/demo.mp4 \
  --title "v1.0 - real-time detection demo" \
  --notes "45s webcam demo. YOLOX-Tiny INT8, 416px, CPU-only on an i5-1135G7."
```

The asset URL follows the pattern in step 7 — you do not need to look it up, the
filename you uploaded is the filename in the URL.

### Check it before you submit

- **Open the link in a private window.** A link that only resolves while you are
  signed into your own account is the most common way a submission's demo
  silently fails, and you cannot detect it from your own browser session. Public
  repo release assets are reachable by anyone; private repo assets are not, so
  if the repo is private this is the step that will catch it.
- **Confirm the file plays after downloading**, not just that it downloads. An
  interrupted upload produces a truncated MP4 that still returns HTTP 200.

### Notes

- Release assets have a much larger size ceiling than the 25 MB target in
  section 4, so that target is about the reviewer's patience, not a hard limit.
- Deleting the release deletes its assets and breaks the README link. If you
  re-cut the release, re-check the link.
- If you later want the video to play inline on the repo front page instead,
  the alternative is a GitHub *attachment*: drag the MP4 into an issue comment,
  submit it, and paste the resulting `user-attachments` URL into the README bare
  on its own line. That renders as a player but is tied to the issue.
