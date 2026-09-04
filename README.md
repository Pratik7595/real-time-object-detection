# Real-time object detection

Real-time object detection on a webcam, running on the CPU. It uses YOLOX-Tiny
through ONNX Runtime, detects all 80 COCO classes, draws boxes with confidence
scores, and sustains **19.8 FPS at 416×416 on an Intel i5-1135G7 with no GPU** —
or **42.1 FPS** with the INT8 model. There is no PyTorch dependency and no CUDA
requirement: `pip install -r requirements.txt` is about 40 MB of wheels and the
same command on x86_64 and ARM64. Everything also runs from a video file or a
still image, so it is demonstrable on a machine with no camera.

![Detection output](results/sample_detection.png)

*Boxes, class names, two-decimal confidence scores and the live FPS/latency HUD.
This is `assets/sample.jpg`, a CC BY 2.0 image bundled with the repo — reproduce
it with `python -m src.main --source assets/sample.jpg`.*

**Demo video:** `DEMO_VIDEO_URL` — recorded with the tool's own `--record` flag.
See [docs/DEMO_SHOTLIST.md](docs/DEMO_SHOTLIST.md) for the shot list and the
exact commands.

---

## Features

- **Threaded capture with a drop-stale buffer.** The capture thread keeps only
  the newest frame. If inference falls behind, frames are dropped rather than
  queued, so display latency stays bounded instead of drifting further behind
  reality every second. Measured cost of the capture stage on a live 30 fps
  camera: **0.11 ms**.
- **All 80 COCO classes**, with a `--classes` filter that accepts names or
  indices.
- **Confidence on every box**, to two decimals, on a filled chip whose text
  flips between black and white so it stays readable against bright and dark
  scenes alike.
- **Live HUD** showing instantaneous FPS, rolling-average FPS, detection count
  and inference latency.
- **Per-stage instrumentation** (capture / preprocess / inference / postprocess /
  render) with a per-run CSV written to `results/`.
- **`--record`** writes an annotated MP4 at a *measured* frame rate, so the clip
  plays back at real speed instead of being silently sped up or slowed down.
- **Any source:** webcam index, video file, or still image. `--source 0`,
  `--source clip.mp4`, `--source assets/sample.jpg`.
- **Checksummed weight download.** Weights are not committed; the download
  script verifies a pinned SHA-256.
- **Honest measurement.** Every number in this README comes from a script in
  this repository, and each one names the command that produces it.

## Install

Requires **Python 3.9+** (developed and measured on 3.12.10). No GPU, no CUDA,
no compiler.

<details open>
<summary><b>Windows</b></summary>

```bash
git clone https://github.com/YOUR_USERNAME/real-time-object-detection.git
cd real-time-object-detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python models/download_weights.py
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
git clone https://github.com/YOUR_USERNAME/real-time-object-detection.git
cd real-time-object-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python models/download_weights.py
```

On first run macOS will ask for camera permission for your terminal app. If you
never see the prompt, see Troubleshooting.
</details>

<details>
<summary><b>Linux</b></summary>

```bash
git clone https://github.com/YOUR_USERNAME/real-time-object-detection.git
cd real-time-object-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python models/download_weights.py
```

If OpenCV complains about a missing `libGL.so.1`, install it:
`sudo apt install -y libgl1 libglib2.0-0`.
</details>

Then, on any platform:

```bash
python -m src.main --source 0
```

Press `q` to quit, `s` to save a screenshot to `results/`.

**No camera?** Everything works from a file, and the repo ships a sample image:

```bash
python -m src.main --source assets/sample.jpg
```

## Usage

```bash
# Live webcam (default). Try --source 1 if you have more than one camera.
python -m src.main --source 0

# A video file, or the bundled still image
python -m src.main --source clip.mp4
python -m src.main --source assets/sample.jpg

# Only show certain classes; names or indices both work
python -m src.main --classes person laptop "cell phone"
python -m src.main --classes 0 63 67

# Trade accuracy for speed (needs the variable-input model, see below)
python -m src.main --imgsz 320

# Twice the frame rate by detecting on every other frame.
# The HUD marks boxes as reused while this is on.
python -m src.main --infer-every 2

# Be stricter about what counts as a detection
python -m src.main --conf 0.45 --iou 0.5

# Record an annotated MP4 (this is how the demo video is made)
python -m src.main --source 0 --record results/demo.mp4

# Headless, e.g. over SSH or in CI
python -m src.main --source assets/sample.jpg --no-display --max-frames 100

# List the 80 class names with their indices
python -m src.main --list-classes
```

Every flag falls back to [`config/config.yaml`](config/config.yaml), which is
commented with the reasoning behind each default.

### Benchmarking and evaluation

```bash
# FPS / latency / CPU / RAM across three network input sizes
python -m src.benchmark

# Before-and-after table for each optimisation
python -m src.benchmark --ablation

# ONNX Runtime thread scaling
python -m src.benchmark --threads-sweep

# Capture-resolution sweep (needs a real camera)
python -m src.benchmark --capture-sweep --source 0
```

```bash
# Accuracy on labelled data: ~50 MB of images + a one-time 241 MB annotation download
pip install -r requirements-dev.txt
python scripts/download_coco_subset.py --images 300
python -m src.evaluate
```

### Optional: faster and smaller

```bash
# Variable-resolution copy of the model, needed for --imgsz other than 416
python models/make_dynamic.py

# INT8 quantisation: 2.1x faster, 5.2 MB instead of 20.2 MB, -2.6 mAP
python models/quantize.py --mode static
python -m src.main --model models/yolox_tiny_int8.onnx
```

## Model choice

**YOLOX-Tiny, ONNX, 416×416.** The full comparison is in
[docs/PRD.md](docs/PRD.md); the short version:

| Model | Published mAP@0.5:0.95 | Licence | Runtime dependency |
|---|---|---|---|
| **YOLOX-Tiny** | **32.8** | **Apache-2.0** | onnxruntime, ~16 MB |
| YOLOv8n | 37.3 | **AGPL-3.0** | ultralytics + torch, ~250 MB+ |
| YOLOv5n | 28.0 | **AGPL-3.0** | torch |
| MobileNet-SSD v2 | ~22 | Apache-2.0 | OpenCV DNN only |

Three reasons, in order of weight:

1. **Licence.** YOLOv8 and YOLOv5 weights from Ultralytics are AGPL-3.0 — strong
   copyleft, not permissive, and using them would arguably pull this whole
   repository under AGPL. YOLOX is Apache-2.0. This ruled out the two models
   most people reach for first.
2. **Dependency weight.** Megvii publish a pre-exported `.onnx`, so there is no
   PyTorch in the tree at all. That is the difference between a ~40 MB install
   and a ~300 MB one, and it is why a reviewer can go from clone to live
   detection inside five minutes.
3. **Accuracy per millisecond.** 32.8 mAP at a 416 input beats YOLOv5n's 28.0 at
   640. MobileNet-SSD is faster still but ~22 mAP at 300×300 misses small and
   mid-distance objects badly enough to be visible in a live demo.

The cost of this choice is that ONNX Runtime returns raw tensors, so
letterboxing, grid decode and NMS are implemented here in NumPy
([`src/detector.py`](src/detector.py)) rather than being handed over by a
framework.

**Model licence: Apache-2.0, © Megvii Inc.** Weights are downloaded from the
[YOLOX 0.1.1rc0 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)
and verified against a pinned SHA-256. See [NOTICE](NOTICE) for full attribution.

## Performance

Measured on Intel i5-1135G7 (4C/8T, 15 W), 8 GB RAM, Windows 11, **CPU only, no
GPU**. 300 frames per configuration after a 30-frame warm-up and a 25-second
thermal settle. Reproduce with `python -m src.benchmark`.

| Model input | Device | FPS mean | FPS median | FPS p95 | Inference ms | Frame ms | Peak RSS |
|---|---|---|---|---|---|---|---|
| 320×320 | CPU | 32.4 | 33.0 | 42.4 | 28.8 | 30.8 | 129 MB |
| **416×416** (default) | CPU | **19.8** | 19.8 | 25.4 | 47.1 | 50.5 | 142 MB |
| 512×512 | CPU | 13.2 | 13.1 | 17.1 | 71.8 | 76.0 | 166 MB |
| 416×416 **INT8** | CPU | **42.1** | 43.0 | 48.6 | 20.7 | 23.7 | 111 MB |

**The >15 FPS requirement is met at 320, 416 and INT8, and missed at 512.** Live
webcam at 416 measures 23.0 FPS mean.

Where the time goes at 416 — **inference is 93% of the frame**:

| capture | preprocess | inference | postprocess | render |
|---|---|---|---|---|
| 0.03 ms | 1.20 ms | 47.11 ms | 1.72 ms | 0.44 ms |

Accuracy, on **COCO val2017, first 300 labelled images — not webcam footage**:

| Metric | FP32 | INT8 |
|---|---|---|
| mAP@0.5 | 0.5326 | 0.5226 |
| mAP@0.5:0.95 | 0.3568 | 0.3312 |
| Precision / Recall / F1 @ conf 0.30 | 0.694 / 0.490 / 0.574 | 0.680 / 0.480 / 0.562 |

One caveat worth reading before trusting any FPS number from a laptop: **this
machine throttles**. Four identical back-to-back runs measured 24.3, 21.0, 20.4
and 19.4 FPS with nothing changed but time. The figures above are the sustained
ones. Full detail, the per-optimisation before/after table, the per-class
precision/recall breakdown and the limitations are in
**[docs/PERFORMANCE_ANALYSIS.md](docs/PERFORMANCE_ANALYSIS.md)**.

## Supported classes

All **80 COCO classes** are detected. `python -m src.main --list-classes` prints
them with indices.

**Verified against ground truth.** The following 20 classes were confirmed
detecting on the 300-image labelled COCO subset, with the true-positive count at
`conf=0.30`, `IoU=0.50` from `python -m src.evaluate`. Full 60-row table in
[`results/accuracy_fp32.csv`](results/accuracy_fp32.csv):

| Class | TP | | Class | TP | | Class | TP | | Class | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| person | 437 | | chair | 45 | | car | 38 | | cup | 29 |
| dining table | 24 | | bowl | 21 | | bottle | 19 | | laptop | 15 |
| motorcycle | 15 | | bus | 15 | | tv | 14 | | wine glass | 14 |
| zebra | 13 | | horse | 12 | | sheep | 12 | | tie | 12 |
| book | 11 | | couch | 11 | | pizza | 11 | | elephant | 11 |

**Live camera observations, stated separately.** In live testing the only class
present in front of the camera was `person` — 411 detections across 400
processed frames, peak confidence 0.94. That is an observation about one class
in one room, not an accuracy measurement, and there is no ground truth behind
it. I am not claiming live verification of the other 19: those objects simply
were not in front of the camera. [docs/DEMO_SHOTLIST.md](docs/DEMO_SHOTLIST.md) is a
checklist of objects to hold up while recording the demo, which is the honest
way to demonstrate the rest.

## Project structure

```
real-time-object-detection/
├── src/
│   ├── main.py             CLI entry point and the main loop
│   ├── detector.py         ORT session, letterbox, grid decode, NMS
│   ├── video_stream.py     Threaded capture with a drop-stale buffer
│   ├── visualizer.py       Boxes, labels, confidence, HUD
│   ├── metrics.py          Per-stage timers, rolling FPS, CSV output
│   ├── config.py           YAML + CLI override merge, validation
│   ├── benchmark.py        FPS / latency / CPU / RAM sweeps
│   ├── evaluate.py         mAP and per-class P/R/F1 on labelled data
│   └── coco_classes.py     Class names and the 80 -> 91 id mapping
├── config/config.yaml      Every default, with the reasoning
├── models/
│   ├── download_weights.py Checksummed weight download
│   ├── make_dynamic.py     Relax the fixed 416 input
│   └── quantize.py         Optional INT8
├── scripts/
│   └── download_coco_subset.py
├── tests/                  63 tests, no GPU or camera required
├── assets/sample.jpg       CC BY 2.0, bundled for demos and tests
├── results/                Committed measurement output
└── docs/
    ├── PRD.md              Design decisions and the model comparison
    ├── PERFORMANCE_ANALYSIS.md
    └── DEMO_SHOTLIST.md
```

`metrics.py` and `config.py` are not in the original brief's layout. They exist
because timing and config are shared by `main`, `benchmark` and `evaluate`, and
when timing lived inside the main loop the benchmark re-implemented it and the
two FPS numbers quietly disagreed.

## Platform support

Pure pip wheels, no platform-specific binaries, `pathlib` throughout, no shell
calls. Capture backend selection adapts to the OS (DirectShow/MSMF on Windows,
AVFoundation on macOS, V4L2 on Linux) and always falls back to `cv2.CAP_ANY`.

| Platform | Status |
|---|---|
| Windows 11 x86_64 | **Verified** — all measurements in this repo |
| Linux x86_64 | Wheels available for every dependency; not run by me |
| macOS arm64 (Apple Silicon) | Wheels available for every dependency; not run by me |
| Linux aarch64 (64-bit Pi OS, Jetson) | Wheels available for every dependency; not run by me |
| Windows on ARM | onnxruntime publishes recent ARM64 wheels; not run by me |
| 32-bit Raspberry Pi OS | No `opencv-python` wheel — use `sudo apt install python3-opencv` |

I have one x86_64 Windows laptop, so anything other than the first row is "the
wheels exist", not "I ran it". Expect roughly 3–6× lower frame rates on a
Raspberry Pi 4; use `--imgsz 320` and the INT8 model there.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

63 tests, ~5 seconds, no camera and no GPU needed. They cover NMS geometry
(including that per-class suppression keeps a person standing in front of a TV),
detector output shape and bounds, config validation and CLI precedence, file and
image sources, error paths for missing/corrupt files, and end-to-end inference on
the bundled sample. Tests needing weights skip with instructions rather than fail
if you have not run `download_weights.py`.

One test is worth calling out: `test_normalisation_would_break_the_model` asserts
that feeding the model `/255`-normalised input produces **zero** detections. The
YOLOX export takes raw 0–255 BGR, and getting that wrong fails silently — no
error, no warning, just an empty list. That test is the tripwire.

## Troubleshooting

**"Could not read from camera index 0"**
Another application is usually holding the camera — Teams, Zoom, Slack, or the
Windows Camera app. Close it and retry. If you have more than one camera, try
`--source 1`, `--source 2`. The error message lists every backend that was tried.
To carry on without a camera at all: `--source assets/sample.jpg`.

**Camera permissions**
- *Windows:* Settings → Privacy & security → Camera → allow desktop apps.
- *macOS:* System Settings → Privacy & Security → Camera, and tick your terminal
  (or IDE). macOS only prompts once; if you dismissed it you must enable it here
  by hand.
- *Linux:* your user needs to be in the `video` group —
  `sudo usermod -aG video $USER`, then log out and back in.

**"opened but produced no frame within 6s"**
The device opened but delivered nothing. Almost always another app holding it,
or a virtual camera driver (OBS, ManyCam) that is installed but not running.

**Low FPS**
In this order:
1. `python -m src.benchmark` to find out where the time actually goes.
2. `--imgsz 320` — measured 32.4 FPS versus 19.8 at 416.
3. `python models/quantize.py --mode static` then
   `--model models/yolox_tiny_int8.onnx` — measured 42.1 FPS.
4. `--infer-every 2` — roughly doubles throughput by detecting on every other
   frame. The HUD marks boxes as reused while this is on.
5. `python models/download_weights.py --model nano` for a much smaller model
   (published 25.8 mAP against 32.8).

Also check nothing else is using the CPU. A benchmark run against a background
compile measured 15.3 FPS where an idle machine measured 25.6 — I did this to
myself while building this project and had to throw the numbers away.

**`--imgsz` other than 416 fails**
The released YOLOX export has a hard-coded 416×416 input. Run
`python models/make_dynamic.py` once; the error message says so too.

**`pip install pycocotools` fails**
Only needed for `src/evaluate.py`, which is why it is in `requirements-dev.txt`
and not `requirements.txt`. On Windows, version 2.0.11 ships a cp312 wheel and
needs no compiler. If pip falls back to a source build, either install the
Visual C++ Build Tools or skip evaluation — nothing else in the project imports
it.

**`libGL.so.1: cannot open shared object file` (Linux)**
`sudo apt install -y libgl1 libglib2.0-0`.

## Licence

Code in this repository: **MIT** — see [LICENSE](LICENSE).

Model weights: **Apache-2.0**, © Megvii Inc., from the
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) project. Not committed
here; fetched and checksum-verified by `models/download_weights.py`.

Bundled sample image: **CC BY 2.0**, from COCO val2017 (image id 340894).

Full attribution for all third-party material is in [NOTICE](NOTICE).
