# Submission guide

A one-page map of the three deliverables and the exact commands that produce
them. The full documentation is in [README.md](README.md); this page exists so
nobody has to read it to get started.

**You do not copy or paste any code.** Clone the repository, run the commands
below inside it, and the programs do the work.

---

## The three deliverables

| # | Deliverable | Where it is | Regenerate it with |
|---|---|---|---|
| 1 | **Working real-time detection code** | [`src/`](src/) — entry point [`src/main.py`](src/main.py) | `python -m src.main --source 0` |
| 2 | **Demo video (30–60 s)** | [Releases → v1.0 → `demo.mp4`](https://github.com/Pratik7595/real-time-object-detection/releases/download/v1.0/demo.mp4) — 43 s, 4.3 MB | `python -m src.main --source 0 --record results/demo.mp4` |
| 3 | **Performance analysis (FPS + accuracy)** | [`docs/PERFORMANCE_ANALYSIS.md`](docs/PERFORMANCE_ANALYSIS.md), raw data in [`results/`](results/) | `python -m src.benchmark` and `python -m src.evaluate` |

**Deliverable 2 is not in the file tree.** Video files are deliberately not
committed to git — that is standard practice, and the assignment asks for the
video to be linked rather than committed. It lives on the Releases page, at the
link above.

---

## Setup: clone to running detector

Five commands. Takes about three minutes, most of it waiting for `pip`.

```bash
git clone https://github.com/Pratik7595/real-time-object-detection.git
cd real-time-object-detection
python -m venv .venv
```

Activate the environment — **this line differs by operating system**:

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

Then:

```bash
pip install -r requirements.txt
python models/download_weights.py
```

`download_weights.py` fetches the model (20 MB, checksum-verified) and builds
the quantised INT8 version the app uses by default (~8 seconds, no network).

### Deliverable 1: run the detector

```bash
python -m src.main --source 0
```

A window opens showing the webcam with boxes, class names and confidence scores,
plus a live FPS counter. Press `q` to quit, `s` to save a screenshot.

**No webcam?** Everything works on the bundled sample image:

```bash
python -m src.main --source assets/sample.jpg
```

---

## Reproducing deliverable 3 (the measurements)

Nothing here is required to *use* the detector — this is only if you want to
re-run the numbers in the performance analysis rather than take them on trust.

### FPS and latency (~5 minutes)

```bash
python -m src.benchmark
```

Prints a table of FPS, per-stage latency, CPU and RAM at three input
resolutions, and writes it to `results/`. The numbers will differ from the
committed ones — those were measured on an Intel i5-1135G7 with no GPU, and the
hardware is stated in the analysis.

### Accuracy (~15 minutes, downloads ~300 MB)

```bash
pip install -r requirements-dev.txt
python scripts/download_coco_subset.py --images 300
python -m src.evaluate
```

Computes mAP@0.5, mAP@0.5:0.95 and per-class precision/recall/F1 on 300 labelled
COCO val2017 images. These are dataset measurements, not webcam measurements —
a live camera has no ground truth, so no accuracy figure is claimed for it.

### Tests (~7 seconds)

```bash
python -m pytest
```

68 tests. No camera or GPU needed.

---

## Recording a new demo video

```bash
python -m src.main --source 0 --record results/demo_raw.mp4 --max-frames 1338
```

`--max-frames 1338` is roughly 45 seconds at 30 fps — **measure your camera's
actual rate first** with `python -m src.main --source 0 --max-frames 150`, or
just press `q` when you are done. The recording needs compressing before upload;
[docs/DEMO_SHOTLIST.md](docs/DEMO_SHOTLIST.md) has the shot list, the ffmpeg
command, and the steps to attach it to a GitHub release.

---

## What each folder is for

| Folder | Contains |
|---|---|
| `src/` | All the application code — this is deliverable 1 |
| `docs/` | Performance analysis, design document, demo shot list |
| `results/` | Committed measurement output that backs the documented numbers |
| `models/` | Scripts that download and quantise the model. Weights are not committed |
| `scripts/` | Dataset download for the accuracy evaluation |
| `tests/` | Test suite |
| `assets/` | Sample image and the INT8 calibration images |
| `config/` | `config.yaml` — every default, with the reasoning |

Nothing needs to be moved, copied, or edited to run the project.

---

## If something goes wrong

The README's [Troubleshooting](README.md#troubleshooting) section covers the
common cases: camera in use by another app, camera permissions on each OS, low
frame rate, and the missing-`libGL` error on Linux. Most first-run problems are
one of those four.
