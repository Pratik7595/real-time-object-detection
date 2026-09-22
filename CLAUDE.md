# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Real-time CPU webcam object detection: YOLOX-Tiny through ONNX Runtime, no
PyTorch, no CUDA. Python 3.9+ (developed on 3.12).

## Commands

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
python models/download_weights.py                 # FP32 weights + builds the INT8 default (~8s)
```

```bash
python -m src.main --source 0                     # webcam
python -m src.main --source assets/sample.jpg --no-display --max-frames 40   # headless, no camera
python -m src.main --list-classes
```

```bash
pip install -r requirements-dev.txt
python -m pytest                                  # 68 tests, ~7s, no camera or GPU
python -m pytest tests/test_nms.py                # one file
python -m pytest tests/test_detector.py::test_normalisation_would_break_the_model   # one test
python -m pytest -k "nms and not batched"
```

```bash
python -m src.benchmark                           # model-input sweep 320/416/512
python -m src.benchmark --ablation                # cumulative optimisations, FP32 baseline
python -m src.benchmark --threads-sweep
python -m src.benchmark --capture-sweep --source 0   # needs a real camera

python scripts/download_coco_subset.py --images 300   # ~300 MB, one time
python -m src.evaluate                            # mAP + per-class P/R/F1
```

```bash
python models/quantize.py                         # rebuild INT8 (static, assets/calib)
python models/make_dynamic.py --src models/yolox_tiny_int8.onnx   # relax the fixed 416 input
python scripts/make_sample_output.py              # regenerate the committed sample I/O pair
python scripts/build_calibration_set.py           # re-pick assets/calib (needs COCO annotations)
```

## Architecture

**The model is a bare ONNX graph.** ONNX Runtime returns raw tensors, so
`src/detector.py` owns everything a framework would normally do: letterboxing,
grid decode, and NMS in NumPy. `infer()` returns `(1, num_anchors, 85)` of
*undecoded* predictions — xy are grid-cell offsets, wh are log-space, while
objectness and class scores are already sigmoid'd inside the graph.

**Preprocessing takes raw 0-255 BGR.** No `/255`, no mean/std, no `cvtColor`.
Getting this wrong fails *silently* — no error, just an empty detection list.
`tests/test_detector.py::test_normalisation_would_break_the_model` is the
tripwire and asserts that normalised input yields zero detections. Do not
"fix" preprocessing to look like a YOLOv8 pipeline.

**Three model files, each produced differently.** `config/config.yaml` points at
the INT8 one by default:

| File | Origin | Notes |
|---|---|---|
| `yolox_tiny.onnx` | downloaded, SHA-256 pinned | fixed 416×416 input |
| `yolox_tiny_int8.onnx` | built locally by `quantize.py` | the default; no checksum |
| `*_dynamic.onnx` | `make_dynamic.py` rewrites two dim fields | required for any `--imgsz` ≠ 416 |

`Detector.__init__` rejects a size mismatch against a fixed-input graph and
points at `make_dynamic.py`; `benchmark.py` builds the dynamic copy on demand.

**Two threads, deliberately not three.** `src/video_stream.py` runs capture in a
background thread and everything else on the main thread (OpenCV HighGUI must
own the main thread on macOS). The buffer policy switches on source type:

- **camera** → `drop_stale=True`: keep only the newest frame, discard the rest.
  Queueing would make display latency grow without bound whenever inference
  falls behind.
- **video/image file** → `drop_stale=False`: the producer blocks until the
  consumer takes the frame, because a benchmark or evaluation must see every
  frame.

**`src/metrics.py` is shared on purpose.** `main`, `benchmark` and `evaluate`
all time the same five stages (`capture`, `preprocess`, `inference`,
`postprocess`, `render`) through it. When this logic lived inside the main loop,
the benchmark re-implemented it and the two FPS numbers drifted apart. Keep new
timing there.

**Config precedence depends on `None`.** Every argparse default in `main.py` is
`None` so `apply_cli_overrides` can distinguish "user typed the flag" from
"argparse filled it in". Giving a flag a real default silently overrides
`config.yaml`. Unknown YAML keys raise rather than being ignored.

**`Detector` validates its own arguments too, not just `config.py`.** It is
public API and `benchmark.py` and the tests construct it directly, so the YAML
path is not the only way in. Alongside the input-size checks above, `__init__`
rejects a `preprocess_mode` outside `prealloc|naive` — `preprocess` only
special-cases `"naive"` and anything else falls through to the prealloc path
while `describe()` reports the string it was handed, so a typo would otherwise
produce an ablation row labelled naive but measured on the fast preprocessor.
Keep the two validators in agreement.

**COCO has two id spaces.** The model emits contiguous 0-79 indices; the
annotations use sparse 91-category ids. `src/coco_classes.py` holds the mapping
and `evaluate.py` must translate, or every detection scores against the wrong
category.

## Measurement discipline

This project's credibility rests on every published number being reproducible.

- **Never write a number into the docs that a script here did not produce.**
  Where something was not measured, the docs say "not measured" rather than
  interpolating. `docs/PERFORMANCE_ANALYSIS.md` and `docs/PRD.md` §8 also record
  predictions that measurement *contradicted* — keep that habit rather than
  quietly editing a wrong prediction away.
- **The dev laptop throttles ~20% over a run.** `benchmark.py --settle`
  (default 20s) loads the CPU to a steady thermal state first. Differences of a
  few percent are smaller than sweep-order drift and must be re-checked with a
  counterbalanced A/B, not read off a single sweep.
- **Warm-up frames are skipped, never deleted.** `Metrics.summarize(skip_first=)`
  excludes them from the summary while the per-run CSV keeps every frame. Frame 0
  costs ~625 ms against ~26 ms steady state.
- **`--infer-every > 1` makes median/p95 FPS meaningless** (bimodal per-frame
  cost); only the throughput mean stays valid, and the benchmark prints `n/a`.
- **`evaluate.py` uses two thresholds.** Detections are gathered once at
  `MAP_CONF = 0.001` because mAP needs the low-confidence tail of the PR curve;
  precision/recall/F1 are reported separately at `--pr-conf` (default 0.30).
- **Keep `assets/calib/` disjoint from the evaluation subset.** The INT8 model
  calibrates on those 24 images; `download_coco_subset.py` takes the *first* N
  val2017 images by id, and `build_calibration_set.py` selects only from outside
  that range. Calibrating on the eval set inflates the reported accuracy.
- Benchmarks are invalid if anything else is using the CPU. Do not run them in
  parallel with other work.

## Repository conventions

- **Not committed:** model weights (`models/*.onnx`), `data/`, and video
  (`*.mp4`). Download scripts and the Releases page cover them instead.
- **Committed:** the curated tables under `results/` that back the documented
  numbers, plus the sample input/output pair. Per-run `results/run_*.csv` is
  gitignored.
- Weights must stay permissively licensed — YOLOX is Apache-2.0. Ultralytics
  YOLOv5/v8 weights are AGPL-3.0 and were rejected for that reason; see
  `docs/PRD.md` §1 and `NOTICE`.
- `pathlib` throughout; no absolute paths, no shell-outs. Capture backend
  selection is platform-*adaptive* (`_backend_candidates`) and always falls back
  to `cv2.CAP_ANY`.
- Tests that need weights skip with the fetch command rather than failing, so a
  clean clone is never red.
