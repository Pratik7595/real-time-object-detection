# PRD / Technical Design — Real-Time Webcam Object Detection

**Status:** draft, awaiting approval (Phase 1)
**Date:** 2026-09-04

## 0. Scope and target machine

Build a CPU-first real-time object detector that reads a webcam (or a video file),
draws boxes with confidence scores, holds >15 FPS, and ships with measured numbers
rather than claimed ones.

Development / primary benchmark machine — every number in the final repo that is
labelled "measured" comes from this box unless stated otherwise:

| Item | Value |
|---|---|
| OS | Windows 11 Home 24H2 (build 26200) |
| CPU | Intel Core i5-1135G7, 4 cores / 8 threads, 2.4 GHz base (Tiger Lake) |
| RAM | 7.8 GB |
| GPU | Intel Iris Xe iGPU + NVIDIA MX350 2 GB — **not used**, CPU-only is the default path |
| Python | 3.12.10 (x86_64) |
| Camera | HP TrueVision HD, built-in |
| ffmpeg | not installed — affects the demo-compression step, see R4 |

The MX350 exists on this laptop but I am deliberately not building around it. The
brief asks for platform independence and the reviewer's machine probably has no CUDA,
so a GPU path that only I can run would be a liability rather than a feature. GPU
stays an optional execution provider behind a flag.

---

## 1. Model selection

### Candidates

Accuracy figures below are the **published** COCO val2017 numbers from each project's
own paper/README — not my measurements. I am using them only to rank the options; the
real numbers for this repo come from `benchmark.py` and `evaluate.py` (§5).

| Model | Input | Published mAP@0.5:0.95 | Params | Licence | Runtime dependency | ARM64 story |
|---|---|---|---|---|---|---|
| **YOLOX-Tiny** | 416 | **32.8** | 5.06 M | **Apache-2.0** | ONNX Runtime (~16 MB wheel) | ORT ships aarch64 + macOS-arm64 wheels |
| YOLOX-Nano | 416 | 25.8 | 0.91 M | Apache-2.0 | ONNX Runtime | same |
| YOLOv8n (Ultralytics) | 640 | 37.3 | 3.2 M | **AGPL-3.0** | `ultralytics` + PyTorch (~250 MB+) | torch aarch64 wheels exist, but heavy |
| YOLOv5n (Ultralytics) | 640 | 28.0 | 1.9 M | **AGPL-3.0** | torch, or ONNX export | same |
| MobileNet-SSD v2 | 300 | ~22 | 4.3 M | Apache-2.0 (TF model zoo) | OpenCV DNN only, zero extra deps | anywhere OpenCV runs |

### The licence problem, stated plainly

The constraint says *"use only permissively licensed pre-trained weights"*. YOLOv8n
and YOLOv5n are **AGPL-3.0** — strong copyleft, not permissive. Using those weights
would arguably pull this whole repo under AGPL. That rules out the two models most
people reach for first. I would rather flag this now than ship a README that says
"permissive" next to an AGPL model.

### Recommendation: YOLOX-Tiny, ONNX, 416×416

YOLOX-Tiny is the only candidate that is simultaneously permissive (Apache-2.0,
verified in `Megvii-BaseDetection/YOLOX/LICENSE`), accurate enough to be interesting
(32.8 mAP beats YOLOv5n's 28.0 while running at a smaller 416 input), and free of a
PyTorch dependency, because Megvii publish a pre-exported `.onnx` — a 20.2 MB file at
a stable GitHub release URL I have already confirmed exists. MobileNet-SSD would be
the safest choice for raw speed and zero extra dependencies, but ~22 mAP at 300×300
misses small and mid-distance objects badly enough that the live demo would look
weak, and a reviewer would fairly ask why I picked a 2018 model. If measurement shows
Tiny cannot clear 15 FPS on this i5, `yolox_nano.onnx` (3.7 MB, identical code path,
one config line) is the documented fallback rather than a rewrite.

**Weights:**
`https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx`
— 20.2 MB, Apache-2.0, © Megvii Inc. The SHA-256 is computed on first download and
pinned into `models/download_weights.py` so later runs verify against it.

**Not committed to git.** `models/*.onnx` is gitignored; the download script is the
deliverable.

---

## 2. Runtime choice

**ONNX Runtime 1.x, `CPUExecutionProvider` by default.**

| Requirement | How ORT satisfies it |
|---|---|
| x86_64 + ARM | ORT publishes `manylinux_aarch64`, `macosx_arm64` and `win_amd64` wheels; `pip install onnxruntime` is the same command everywhere. Windows-on-ARM wheels exist for recent releases — I have no ARM machine, so the README marks that row "not personally verified" instead of claiming it. |
| No platform-specific binaries | Pure pip wheels. No CUDA, no TensorRT, no OpenVINO in the default path. |
| Minimal dependencies | `onnxruntime` ≈ 16 MB vs `torch` ≈ 250 MB. A reviewer's `pip install -r requirements.txt` finishes in well under a minute, which matters for the "runs within five minutes of reading the README" bar. |
| Optional acceleration | `--device` maps to a provider list; `onnxruntime-gpu` or `onnxruntime-openvino` can be swapped in without touching detector code. Default stays CPU. |

**Rejected alternatives.** OpenCV DNN means zero extra dependencies and it can read
ONNX — genuinely tempting — but its ONNX importer is version-sensitive and I do not
want the repo to fail on a reviewer's slightly different OpenCV build. It also has no
clean INT8 story. PyTorch/Ultralytics is rejected on licence (§1) and install weight.

**Consequence I am accepting:** ORT hands back raw tensors, so I write letterboxing,
grid decode and NMS myself in NumPy — roughly 120 lines of real work instead of
`model(frame)`. For a role assessment that feels like the right side of the trade.

---

## 3. Architecture

### Module breakdown

| File | Responsibility | Deliberately *not* doing |
|---|---|---|
| `src/video_stream.py` | Threaded capture, single-slot latest-frame buffer, backend selection, fail-fast errors | Any drawing or model work |
| `src/detector.py` | ORT session lifetime, letterbox, decode, NMS, `detect(frame) -> list[Detection]` | Knowing about cameras or windows |
| `src/visualizer.py` | Boxes, labels, confidences, HUD overlay | Timing anything |
| `src/metrics.py` | Rolling per-stage timers, FPS, CSV writer | Printing to screen |
| `src/config.py` | YAML load + CLI override merge + validation | Defaults scattered elsewhere |
| `src/main.py` | CLI, wiring, main loop, `--record` | Business logic |
| `src/benchmark.py` | Fixed-length sweep, resolution matrix, CPU/RAM sampling | Touching the camera by default |
| `src/evaluate.py` | COCO subset → mAP + per-class P/R/F1 | Any live-stream accuracy claim |

`src/metrics.py` and `src/config.py` are additions to the tree in the brief. Reason:
the timing instrumentation is used by `main.py`, `benchmark.py` *and* `evaluate.py`,
and if it lives inside `main.py` the benchmark ends up re-implementing it and the two
sets of numbers drift apart. Same argument for config loading, which `tests/` needs to
import on its own.

### Frame pipeline, end to end

```
[capture thread]                      [main thread]
 cap.read()  ──►  slot (1 frame,  ──►  latest()  ──► letterbox + CHW ──► ORT run
   loop            lock, stale          (never         (preallocated       (session
   forever)        frames dropped)       blocks)        buffers)            .run)
                                                                             │
                            imshow ◄── draw boxes + HUD ◄── NMS ◄── grid decode
                              │
                              └──► VideoWriter (only when --record)
```

Every stage is wrapped in a timer that pushes into a 120-frame deque, so the HUD, the
CSV and the benchmark all read from the same source of truth.

### Threading model

Two threads, not three:

- **Capture thread** — `cap.read()` in a tight loop, writing into a one-slot buffer
  under a lock. Unread frames are *dropped*, not queued. This is the single most
  important decision in the project: with a queue, a slow detector makes the display
  drift further behind reality every second. With a one-slot drop buffer, latency
  stays bounded and you simply see a lower frame rate.
- **Main thread** — inference, drawing, `imshow`, key handling. It stays the main
  thread because OpenCV HighGUI must own the main thread on macOS, and I want one code
  path across all three OSes.

Why no separate inference thread: ORT releases the GIL inside `run()`, so a third
thread would work, but it buys throughput by adding a frame of latency and a second
synchronisation point. I would rather ship two threads I can reason about and add
`--infer-every N` (§4.7) if measurement says I need it. If the benchmark shows a real
gap I will revisit and document the change — but I am not building it on a guess.

**Failure handling:** if `VideoCapture` will not open, or opens but returns no frame
within a timeout, the stream raises `CameraError` naming the backend and index tried,
and points at `--source <video file>`. No silent black windows.

---

## 4. Performance strategy (clearing 15 FPS on this i5)

Each item gets a before/after row in `PERFORMANCE_ANALYSIS.md`, measured with the same
benchmark, one change at a time.

1. **416×416 model input, decoupled from capture resolution.** Capture at 640×480;
   the model never sees more pixels than it needs. Letterbox preserves aspect ratio so
   boxes map back cleanly.
2. **Preallocated buffers.** One `np.empty((1,3,416,416), np.float32)` and one
   letterbox canvas, allocated at construction and reused; `cv2.resize(..., dst=buf)`
   where OpenCV allows it. Goal: zero per-frame heap allocation in the hot path.
3. **No normalisation math.** YOLOX's ONNX export takes raw 0–255 BGR — no `/255`, no
   mean/std, no `cvtColor`. That removes two full-frame passes per frame versus a
   typical YOLOv8 preprocessing chain. *To be confirmed against the actual model in
   Phase 2 — if the export does expect normalisation, I add it and drop this claim.*
4. **Score threshold before NMS.** Filter the ~3549 raw candidates by
   `obj_conf * cls_conf` first, so NMS normally sees fewer than 30 boxes. Vectorised
   NumPy NMS, per-class via a class-offset trick, no Python loop over boxes.
5. **ORT session tuning.** `graph_optimization_level=ORT_ENABLE_ALL`,
   `intra_op_num_threads = 4` (physical cores), `inter_op = 1`, plus
   `cv2.setNumThreads(2)`. OpenCV and ORT both grabbing 8 threads oversubscribes a
   4-core chip and makes things *slower* — a real effect I expect to show up in the
   before/after table.
6. **INT8 dynamic quantisation** via `onnxruntime.quantization`, opt-in through
   `models/quantize.py` producing `yolox_tiny_int8.onnx`. Expected roughly 1.5–2× on
   this CPU with some mAP loss — both numbers measured, not assumed. FP16 is skipped:
   it is not a win on x86 CPU.
7. **`--infer-every N` (default 1).** Detect on every Nth frame and re-draw the
   previous boxes in between. This is honest frame-skipping, not tracking, so the HUD
   shows a "stale detections" marker while it is active and nobody mistakes the
   resulting FPS for real detection throughput. Documented as a weak-hardware fallback.

**Target:** ≥15 FPS end to end at 640×480 capture / 416 inference, CPU only, on the
i5-1135G7. If FP32 misses it: INT8 (6), then Nano, then (7).

---

## 5. Measurement plan

### FPS and latency

- `time.perf_counter()` around five stages: capture-wait, preprocess, inference,
  postprocess/NMS, render. Rolling 120-frame deque feeds instantaneous FPS, rolling
  mean FPS and per-stage milliseconds into the HUD.
- Every run writes `results/run_<timestamp>.csv`, one row per frame, five stage
  timings plus frame index. Nothing is summarised in memory only.
- `benchmark.py` runs a fixed **300-frame** loop, discards the first 30 as warm-up
  (ORT's first `run()` is much slower), and reports mean / median / **p95** FPS with
  the stage breakdown. It defaults to a **video file**, not the camera, so results are
  repeatable and independent of lighting and camera auto-exposure. `--source 0` gives a
  live run, reported separately.
- Resolution sweep, three or more configurations: primary axis is model input
  (320 / 416 / 512), secondary is capture resolution (640×480 / 1280×720). *Risk: the
  released ONNX has a fixed 416 input; other sizes need a one-off graph rewrite
  (`models/make_dynamic.py`, `onnx` package). If that does not hold up, the sweep
  becomes capture-resolution-only at fixed 416, and the doc says so.*
- CPU % and RSS sampled with `psutil` on a background timer during the benchmark.

### Accuracy, and how I keep it honest

**A live webcam stream has no ground truth, so I will not report accuracy on it.**
Two clearly separated things go into the docs:

1. **Measured dataset accuracy.** `scripts/download_coco_subset.py` fetches the
   `instances_val2017` annotations and the first **300 val2017 images** by their
   `coco_url` (≈50 MB of JPEGs, gitignored). `evaluate.py` runs the *same*
   `Detector.detect()` the live app uses, writes COCO-format detections and scores them
   with `pycocotools` → **mAP@0.5** and **mAP@0.5:0.95**. Per-class
   precision/recall/F1 come from a separate greedy IoU ≥ 0.5 match at a stated
   confidence threshold, because `pycocotools` does not expose those directly. Every
   table carries the header *"COCO val2017 subset, N=300 images — not webcam footage."*
2. **Qualitative live observations.** A checklist of classes I actually held in front
   of the camera, with screenshots, described as observations. No percentages attached.

I will also report **confidence-threshold sensitivity** (mAP and mean detections per
frame across conf ∈ {0.15, 0.25, 0.40, 0.60}) and the FP32-vs-INT8 accuracy delta, so
the speed/accuracy trade-off section is backed by two axes of real data.

---

## 6. Repository layout

```
real-time-object-detection/
├── README.md
├── requirements.txt              # runtime deps, pinned
├── requirements-dev.txt          # pytest, pycocotools, onnx (eval/quantise only)
├── LICENSE                       # MIT (my code)
├── NOTICE                        # third-party licences incl. YOLOX Apache-2.0
├── .gitignore
├── config/config.yaml
├── models/
│   ├── download_weights.py       # URL + SHA-256 verification
│   ├── quantize.py               # optional INT8
│   └── .gitkeep                  # *.onnx ignored
├── scripts/
│   └── download_coco_subset.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── video_stream.py
│   ├── detector.py
│   ├── visualizer.py
│   ├── metrics.py
│   ├── benchmark.py
│   ├── evaluate.py
│   └── main.py
├── assets/sample.jpg             # bundled: tests + no-camera demo
├── results/                      # CSVs, tables, plots, screenshots (committed)
├── docs/
│   ├── PRD.md                    # this file
│   ├── PERFORMANCE_ANALYSIS.md
│   └── DEMO_SHOTLIST.md
└── tests/
    ├── test_config.py
    ├── test_detector.py
    ├── test_nms.py
    └── test_file_source.py
```

Additions to the tree in the brief, with reasons: `scripts/` (dataset download is not a
model concern), `assets/` (tests need a real image, and it makes the repo demoable with
no camera and no download), `NOTICE` (Apache-2.0 §4 asks for attribution — cheap to do
properly), `requirements-dev.txt` (`pycocotools` can need a compiler on Windows, and
making the runtime install depend on that would break the five-minute rule).

### Dependency budget

| Package | Why it is here |
|---|---|
| `onnxruntime` | the inference runtime |
| `opencv-python` | capture, resize, draw, VideoWriter |
| `numpy` | decode + NMS |
| `pyyaml` | config |
| `psutil` | **justification:** the brief asks for CPU/RAM in the benchmark and there is no cross-platform stdlib way to get RSS and CPU%. Imported only by `benchmark.py`. |
| `tqdm` | **justification:** progress bars on the two long downloads. Genuinely optional — say the word and I drop it for four runtime deps. |
| *dev:* `pytest`, `pycocotools`, `onnx` | tests, mAP scoring, graph rewrite/quantisation |

Exact pins go in after a clean `pip install` on Python 3.12 records what actually
resolves. I am not writing version pins I have not installed.

### Deliverables checklist (brief → artefact)

| Brief item | Where it lands |
|---|---|
| Real-time detection from webcam | `python -m src.main --source 0` |
| Pre-trained model | YOLOX-Tiny ONNX via `models/download_weights.py` |
| Boxes + confidence scores | `src/visualizer.py`, 2-dp confidence on every box |
| ≥10 object classes | full 80-class COCO set; ≥15 named as verified in the README |
| >15 FPS | `src/benchmark.py` + FPS table in `PERFORMANCE_ANALYSIS.md` |
| 30–60 s demo video | produced by `--record`; `docs/DEMO_SHOTLIST.md`; linked, not committed |
| Performance analysis (FPS + accuracy) | `docs/PERFORMANCE_ANALYSIS.md`, `results/*.csv` |
| Platform independence | `pathlib` throughout, pip-only deps, ARM wheel notes in README |
| GitHub repo + deps + README | repo root |

---

## 7. Risks and open questions

| # | Risk | Mitigation |
|---|---|---|
| R1 | YOLOX-Tiny FP32 misses 15 FPS on this i5 | INT8 → Nano → `--infer-every`. All measured, all documented. |
| R2 | Fixed-416 ONNX blocks the multi-resolution sweep | Dynamic-axis rewrite; fallback is a capture-resolution sweep, labelled as such. |
| R3 | `pycocotools` build pain on Windows | Kept in `requirements-dev.txt`; README documents the fallback; the core app never imports it. |
| R4 | **ffmpeg is not installed on this machine** | `--record` uses `cv2.VideoWriter` (mp4v) and needs no ffmpeg. Only the *compression* step does — README gives `winget install Gyan.FFmpeg` / `brew` / `apt`, and a 45 s 640×480 clip may well land under 25 MB without it. |
| R5 | Preprocessing assumption (§4.3) turns out wrong | Verified against the real model before any claim is written. |
| R6 | Reviewer has no camera | `--source assets/sample.jpg` and `--source <video>` both work; tests cover the file path. |

**One question before I start.** The licence constraint is the real fork in the road.
I recommend YOLOX-Tiny (Apache-2.0). If you would rather have YOLOv8n's higher mAP and
easier tooling, that is a fine call — but the README then has to say **AGPL-3.0** and
the "permissively licensed" constraint drops. Tell me which way and I will build it.

---

---

## 8. Postscript: what the measurements changed

Approved as written, with YOLOX-Tiny. This section records where the plan above
turned out to be wrong, because a design document that gets quietly edited to
match the results is worth nothing. Numbers in
[PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md).

| Plan said | Reality | Outcome |
|---|---|---|
| §4.5 Capping OpenCV threads will help — both libraries grabbing 8 threads must oversubscribe a 4-core chip | Interleaved A/B over 3 rounds: capping **cost ~5%** (20.13 → 19.15 FPS). Preprocessing is ~1.1 ms of a ~50 ms frame, so there was never much to reclaim | **Hypothesis rejected.** Default changed from `2` to `0`. Knob kept for busy machines |
| §4.3 The export takes raw 0–255 BGR, no normalisation *(marked "to be confirmed")* | Confirmed empirically — `/255` input yields **zero** detections | **Held.** Locked down by `test_normalisation_would_break_the_model` |
| R2 The fixed-416 export may block the resolution sweep | The head's Reshape nodes already target `[1, 85, -1]`, so only the declared input dims needed widening. 320/416/512 all produce exactly the predicted anchor counts | **Resolved.** `models/make_dynamic.py`; the sweep runs on model input as originally hoped |
| §5 A fixed frame count is enough for a fair benchmark | Not on this laptop. Four identical runs drifted 24.3 → 19.4 FPS on thermal throttling alone — **larger than most of the effects being measured** | **Plan was inadequate.** Added `--settle`; small differences re-checked with interleaved A/B |
| §4.6 INT8 expected "~1.5–2×, with some mAP loss", opt-in | **1.9×** at 416 (19.8 → 38.0 FPS), −2.5 mAP@0.5:0.95, model 20.2 MB → 5.2 MB | **Held**, at the top of the range — and promoted from opt-in to **the shipped default** (see below) |
| §1 YOLOX-Tiny will clear 15 FPS at 416 | 19.8 FPS sustained, 23.0 live | **Held.** The Nano fallback was never needed |
| Dependency budget included `tqdm` for download progress bars | Both download scripts print their own progress in three lines | **Dropped.** Five runtime dependencies, not six |

The two things I would tell someone starting this again: the pipeline is 93%
inference, so optimising anything else is rearranging deck chairs; and on a thin
laptop, thermal drift is a larger effect than most of the optimisations, so
measure that before you trust any before/after table.

## 9. Amendment: INT8 promoted to the default

The PRD scoped INT8 as an opt-in extra (§4.6). Once measured it was 1.9× for 2.5
mAP@0.5:0.95 points, which is a good enough trade on CPU-only hardware that
leaving it behind a manual step was hard to justify. It is now what
`config/config.yaml` points at. Three consequences the original plan did not
anticipate:

**The default model is built, not downloaded.** There is no permissively
licensed pre-quantised YOLOX ONNX to fetch, so `models/download_weights.py` now
fetches FP32 and quantises locally (~8 s, no network). That means the default
model carries **no checksum** — the one guarantee §1 was proud of. FP32 is still
verified against its pinned SHA-256; the derived model is only as trustworthy as
the ONNX Runtime that built it. `--no-quantize` opts out.

**Static calibration created a test-set contamination bug, briefly.** The first
INT8 build calibrated on the first 64 images of `data/coco_subset` — which *is*
the evaluation set. The model was tuned on data it was then scored against.
Fixed by `scripts/build_calibration_set.py`, which selects 24 redistributable
(CC BY 2.0) images from *outside* the evaluation range and commits them to
`assets/calib/` so the build is offline and reproducible. Measured bias: 0.0004
mAP@0.5:0.95 — negligible, and it happened to favour the corrected version. The
size of the error is not the point; it was only knowable after checking.

**The bottleneck moved off the detector entirely.** With inference at ~23 ms the
webcam became the constraint: measured on its own, with no inference in the
loop, it delivers **15.0 fps** in evening light against a nominal 30. So the
live frame rate went *down* (23.0 → 15.0 FPS) while the detector got nearly
twice as fast. Nothing regressed — the earlier live run simply happened in
better light. The PRD's §5 measurement plan was right to insist on a
deterministic file source for the headline numbers; it just did not anticipate
how quickly the camera would become the thing worth complaining about.
