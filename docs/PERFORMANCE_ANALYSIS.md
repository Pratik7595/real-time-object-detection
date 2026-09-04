# Performance analysis

Every number here was produced by a script in this repository. The command that
generates each table is given above it, and the raw output is committed under
`results/`. Nothing is estimated, extrapolated or copied from a paper — where a
published figure appears it is labelled as published and attributed.

Two things this document is careful about:

* **Speed and accuracy were measured on different inputs.** FPS comes from a
  fixed image loop and a live camera. Accuracy comes from 300 labelled COCO
  val2017 images. No accuracy figure anywhere is derived from webcam footage,
  because webcam footage has no ground truth.
* **This laptop throttles.** It is a thin 15 W chassis and the sustained clock is
  well below the boost clock. That turned out to matter more than most of the
  optimisations, so it is documented first rather than buried.

---

## 1. Test methodology

### Hardware and software

| | |
|---|---|
| Machine | HP laptop, thin chassis, 15 W class |
| CPU | Intel Core i5-1135G7 (Tiger Lake), 4 cores / 8 threads, 2.4 GHz base |
| RAM | 8.3 GB total (7.8 GB usable) |
| GPU | Intel Iris Xe iGPU + NVIDIA MX350 2 GB — **unused**, every number is CPU-only |
| OS | Windows 11 Home 24H2, build 26200, x86_64 |
| Python | 3.12.10 |
| onnxruntime | 1.29.0, `CPUExecutionProvider` |
| opencv-python | 5.0.0.93 |
| numpy | 2.5.2 |
| Camera | HP TrueVision HD, built-in, 640×480 @ 30 fps |

### How a run works

`src/benchmark.py` runs a fixed frame count per configuration, discards the
first 30 frames, and reports the rest. Specifics that matter:

- **Warm-up is discarded, not deleted.** ONNX Runtime's first `run()` allocates
  its arena and spins up its thread pool; OpenCV's first `resize` and first
  `putText` are also one-off costs. On this machine frame 0 costs ~625 ms
  against ~26 ms for every frame after it. Those frames stay in the per-run CSV
  and are excluded from the summary.
- **The default source is `assets/sample.jpg`, looped, not the camera.** A
  camera caps throughput at its own 30 fps and varies with exposure and
  lighting, so it measures the camera rather than the detector. The consequence
  is that the `capture` column is ~0.03 ms and is *not* representative of live
  capture; the live measurement in §3 is reported separately for that reason.
- **Rendering is included.** Boxes and labels are drawn on every benchmarked
  frame; only `imshow` is skipped. A benchmark that skipped drawing would report
  a frame rate the app can never actually reach.
- **FPS mean is throughput**, `frames / wall-clock`, not the mean of per-frame
  FPS values. Averaging reciprocals overstates the result.
- **p95 is the fast tail and p5 is the slow tail.** For a video pipeline the p5
  is the one users notice, so it is reported alongside.

### The thermal problem, and what was done about it

Four *identical* back-to-back 150-frame runs of the same configuration:

```bash
python -m src.benchmark --sizes 416 416 416 416 --frames 150 --warmup 20 --settle 0
```

| Run | FPS mean | Inference ms |
|---|---|---|
| 1 | 24.3 | 38.2 |
| 2 | 21.0 | 44.2 |
| 3 | 20.4 | 45.7 |
| 4 | 19.4 | 48.2 |

Raw output: [`results/throttling_repeat.md`](../results/throttling_repeat.md).

A 20% decline with nothing changed but time. This is the package dropping off
its boost clock, and it is larger than most of the optimisations measured below.
Without accounting for it, whichever configuration runs first in a sweep wins,
and the "effect" measured is really the running order.

So `benchmark.py` takes a `--settle` argument (default 20 s) that loads the CPU
to a steady thermal state *before* the first measurement. Every table below was
produced with `--settle 25`. The numbers are consequently lower than a
cold-start run and are the ones the machine actually sustains.

**Where a difference is only a few percent, settling is not enough**, because the
drift within a sweep is of the same order. Those comparisons were re-run as
interleaved A/B tests (ABABAB) so drift hits both arms equally — see §4.2.

---

## 2. FPS across resolutions

```bash
python -m src.benchmark --frames 300 --warmup 30 --settle 25
```

Raw output: [`results/benchmark_resolution.md`](../results/benchmark_resolution.md)

Capture is 640×480 throughout; the variable is the network input. Sizes other
than 416 use the variable-input graph from `models/make_dynamic.py`.

| Model input | Device | FPS mean | FPS median | FPS p95 | FPS p5 | Inference ms | Frame ms | CPU % | Peak RSS MB | Dets/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| 320×320 | CPU (i5-1135G7) | **32.4** | 33.0 | 42.4 | 24.9 | 28.75 | 30.81 | 398 | 129 | 8.80 |
| **416×416** (shipped) | CPU (i5-1135G7) | **19.8** | 19.8 | 25.4 | 16.1 | 47.11 | 50.49 | 397 | 142 | 9.90 |
| 512×512 | CPU (i5-1135G7) | **13.2** | 13.1 | 17.1 | 11.1 | 71.79 | 75.95 | 397 | 166 | 11.00 |

CPU % is relative to a single core, so 398% means all four cores saturated.

Reading this honestly:

- **The >15 FPS requirement is met at 320 and 416, and missed at 512.** 512 is in
  the table because it is the measurement that shows where the approach runs
  out, not because it is a recommendation.
- **416 clears the bar by ~32% under sustained thermal load.** On a cold machine
  the same configuration measures 24–35 FPS; those runs are real but not
  representative of a machine that has been running for a minute.
- Scaling is worse than pixel count predicts. 512 has 1.5× the pixels of 416 but
  costs 1.52× the time; 416 has 1.69× the pixels of 320 and costs 1.64×. Roughly
  linear in pixels, with no cliff — consistent with being compute-bound rather
  than memory-bound at these sizes.
- Detections per frame rise with input size (8.80 → 11.00) on the identical
  image. Higher resolution genuinely finds more objects, which is the trade the
  §5 table quantifies against accuracy.

---

## 3. Latency breakdown per pipeline stage

Same runs as §2. All values are milliseconds, mean over 300 frames.

| Configuration | capture | preprocess | inference | postprocess | render | **total** |
|---|---|---|---|---|---|---|
| 320×320 | 0.03 | 0.58 | 28.75 | 1.05 | 0.41 | **30.81** |
| 416×416 (shipped) | 0.03 | 1.20 | 47.11 | 1.72 | 0.44 | **50.49** |
| 512×512 | 0.03 | 1.47 | 71.79 | 2.19 | 0.47 | **75.95** |

**Inference is 93% of the frame budget at 416.** Everything else put together is
3.4 ms. That single fact determined which optimisations were worth pursuing and
which were not — see §4.

### Live camera, for comparison

```bash
python -m src.main --source 0 --no-display --max-frames 150
```

| Stage | Image loop (416) | Live camera (416) |
|---|---|---|
| capture | 0.03 ms | **0.11 ms** |
| preprocess | 1.20 ms | 0.80 ms |
| inference | 47.11 ms | 41.44 ms |
| postprocess | 1.72 ms | 0.89 ms |
| render | 0.44 ms | 0.26 ms |
| FPS mean | 19.8 | **23.0** |

Two things to note:

1. **Capture costs 0.11 ms on a live 30 fps camera.** This is the threading model
   working. The camera produces frames faster than the pipeline consumes them,
   so a frame is always already waiting and the main loop never blocks on I/O.
   A single-threaded implementation would pay the camera's frame interval here.
2. **The live run looks faster (23.0 vs 19.8) and that is a thermal artefact,
   not a real difference.** It was a shorter run (6.3 s) on a cooler machine. It
   is reported because it is the honest live measurement, not because live
   capture is somehow quicker than reading a JPEG from RAM.

### Cost of `--record`

Enabling the MP4 writer moves the `render` stage from 0.44 ms to 2.35 ms — about
**1.9 ms per frame, roughly 4%**. There is also a one-off ~20 ms stall on the
frame where the writer opens (codec initialisation), which is why the first
measurement of this looked like a 25% penalty until the writer was forced open
from frame 0 to isolate it.

---

## 4. Effect of each optimisation

```bash
python -m src.benchmark --ablation --frames 300 --warmup 30 --settle 25
```

Raw output: [`results/benchmark_ablation.md`](../results/benchmark_ablation.md)

### 4.1 Before / after

Rows are cumulative; each adds one change to the row above.

| # | Change | FPS mean | Δ FPS | preprocess ms | inference ms | frame ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|
| A | Baseline: allocate preprocessing buffers per frame | 19.2 | — | 3.42 | 46.33 | 51.89 | 142 |
| B | **+ preallocated buffers** *(shipped)* | **20.9** | **+8.9%** | **1.01** | 44.64 | 47.72 | 140 |
| C | + cap OpenCV to 2 threads | 19.4 | −7.2% | 1.24 | 48.01 | 51.53 | 142 |
| D | + `--infer-every 2` | 40.6 | +94% | 0.61 | 22.63 | 24.60 | 141 |
| E | + INT8 quantisation | **42.1** | **+101%** | 1.12 | 20.68 | 23.72 | **111** |

Rows D and E are alternatives to B, not additions to each other in practice —
`--infer-every 2` is a runtime flag and INT8 is a different model file.

### 4.2 What actually worked, and what did not

**Preallocated preprocessing buffers — kept (+8.9%).** Reusing the resize target,
the letterbox canvas and the NCHW blob across frames cut preprocessing from
3.42 ms to 1.01 ms, a **70% reduction in that stage**. But that stage was only
6.6% of the frame, so the end-to-end gain is 8.9%. Worth having, and a useful
reminder that a large relative win on a small stage is a small absolute win.

**Capping OpenCV threads — tested and rejected (−7.2%).** The design document
predicted this would help: OpenCV and ONNX Runtime both default to grabbing
every core, which on a 4-core chip looked like obvious oversubscription. It was
wrong. Because the difference is small enough to be confused with thermal drift,
it was re-run as an interleaved A/B (three rounds, alternating arms):

| Round | OpenCV default | OpenCV capped to 2 |
|---|---|---|
| 1 | 20.00 | 19.21 |
| 2 | 20.23 | 19.51 |
| 3 | 20.16 | 18.74 |
| **mean** | **20.13** | **19.15** |

Capping loses in every round. The default was therefore changed from `2` to `0`
(leave OpenCV alone) in `config/config.yaml`. The knob remains, because it is
the right lever on a machine that is busy doing something else — it is just not
a win here. I have not established the mechanism; the plausible explanation is
that `cv2.setNumThreads` reconfigures a threading runtime shared with ORT, but
that is a hypothesis, not a measurement.

**Frame skipping (`--infer-every 2`) — available, off by default (+94%).** This
doubles throughput by doing half the work, and boxes on skipped frames are stale
by one frame. The HUD says so on screen while it is active. It is documented as
a fallback for hardware that cannot clear 15 FPS otherwise, not as a default,
because it improves the number without improving what the user sees.

**INT8 quantisation — the biggest single win (+101%).** Static quantisation
calibrated on 64 real COCO images (`python models/quantize.py --mode static`).
Inference drops from 44.6 ms to 20.7 ms, the model file from 20.2 MB to 5.2 MB,
and peak RSS from 140 MB to 111 MB. The accuracy cost is measured in §5.3. It is
not the default only because it requires a build step; on this hardware it is
the right choice for anything latency-sensitive.

### 4.3 ONNX Runtime thread scaling

```bash
python -m src.benchmark --threads-sweep --frames 200 --warmup 30 --settle 25
```

Raw output: [`results/benchmark_threads.md`](../results/benchmark_threads.md)

| intra_op_num_threads | FPS mean | Inference ms | CPU % |
|---|---|---|---|
| ORT default *(shipped)* | 19.1 | 48.73 | 397 |
| 1 | 13.1 | 73.92 | 100 |
| 2 | 16.6 | 57.54 | 199 |
| 4 | 20.4 | 45.56 | 398 |
| 8 | 19.3 | 46.93 | 757 |

Scaling from 1 to 4 threads gives 1.56× on 4 cores — normal for convolution
workloads. Going to 8 (hyperthreads) buys nothing and costs 1.9× the CPU, which
matters on battery. ORT's own default already behaves like 4, so the shipped
config leaves `intra_op_threads: 0`; hardcoding `4` would be a marginal win here
and wrong on a reviewer's 8- or 16-core machine.

---

## 5. Accuracy

> **Source: COCO val2017, first 300 images by image id, with human ground-truth
> labels. These are not webcam measurements.** No accuracy figure in this
> repository is derived from live camera footage.

```bash
python scripts/download_coco_subset.py --images 300
python -m src.evaluate
```

Raw output: [`results/accuracy_fp32.md`](../results/accuracy_fp32.md) ·
[`results/accuracy_int8.md`](../results/accuracy_int8.md)

**Method note.** Detections are gathered once at `conf=0.001` and thresholded
afterwards. mAP is the area under the precision-recall curve and needs the
low-confidence tail to trace that curve; evaluating at the app's 0.30 default
would truncate it and understate mAP substantially. Precision, recall and F1 are
operating-point metrics and are reported at a stated threshold (0.30).

### 5.1 Headline

| Metric | YOLOX-Tiny FP32 | YOLOX-Tiny INT8 |
|---|---|---|
| **mAP@0.5** | **0.5326** | 0.5226 |
| **mAP@0.5:0.95** | **0.3568** | 0.3312 |
| mAP@0.75 | 0.3827 | 0.3604 |
| mAP small | 0.1847 | 0.1731 |
| mAP medium | 0.3582 | 0.3239 |
| mAP large | 0.5464 | 0.5044 |
| AR@100 | 0.4786 | — |

For context, the YOLOX authors publish **32.8** mAP@0.5:0.95 for YOLOX-Tiny on
the *full* 5000-image val2017 set. This measurement is 35.7 on a 300-image
subset. The gap is subset sampling error, not a better model — a 300-image slice
is not a substitute for full val2017, which is why every table says "300-image
subset" rather than quoting these as COCO val2017 results.

### 5.2 Per-class precision / recall / F1

At `conf=0.30`, `IoU=0.50`. Full 60-row table in
[`results/accuracy_fp32.csv`](../results/accuracy_fp32.csv); the classes with the
most ground-truth instances in the subset:

| Class | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| person | 685 | 437 | 93 | 248 | 0.825 | 0.638 | 0.719 |
| chair | 119 | 45 | 30 | 74 | 0.600 | 0.378 | 0.464 |
| book | 75 | 11 | 7 | 64 | 0.611 | 0.147 | 0.237 |
| cup | 69 | 29 | 22 | 40 | 0.569 | 0.420 | 0.483 |
| car | 67 | 38 | 19 | 29 | 0.667 | 0.567 | 0.613 |
| bottle | 66 | 19 | 25 | 47 | 0.432 | 0.288 | 0.345 |
| dining table | 48 | 24 | 20 | 24 | 0.545 | 0.500 | 0.522 |
| tie | 45 | 12 | 3 | 33 | 0.800 | 0.267 | 0.400 |
| wine glass | 44 | 14 | 3 | 30 | 0.824 | 0.318 | 0.459 |
| bowl | 38 | 21 | 9 | 17 | 0.700 | 0.553 | 0.618 |
| laptop | 19 | 15 | 9 | 4 | 0.625 | 0.789 | 0.698 |
| tv | 19 | 14 | 4 | 5 | 0.778 | 0.737 | 0.757 |
| motorcycle | 22 | 15 | 4 | 7 | 0.789 | 0.682 | 0.732 |
| **micro-average** | **2145** | **1050** | **463** | **1095** | **0.694** | **0.490** | **0.574** |

The shape of this is consistent and worth stating plainly: **precision (0.69) is
much better than recall (0.49)**. At a 0.30 threshold the model is conservative —
what it reports is usually right, but it misses about half of everything
labelled. `book` (recall 0.147) and `handbag` (0.138) are the clearest failures,
and both are dominated by small, cluttered, overlapping instances.

Some classes score 0.000 on a subset this small — `cow` (9 GT), `donut` (8),
`orange` (4), `scissors` (2). With that little support these are anecdotes, not
measurements, and should not be read as "the model cannot detect oranges".

### 5.3 Speed / accuracy trade-off

| Configuration | FPS mean | mAP@0.5:0.95 | mAP@0.5 | Model size |
|---|---|---|---|---|
| 320×320 FP32 | 32.4 | not measured | not measured | 20.2 MB |
| 416×416 FP32 *(shipped)* | 19.8 | 0.3568 | 0.5326 | 20.2 MB |
| 416×416 INT8 | 42.1 | 0.3312 | 0.5226 | **5.2 MB** |
| 512×512 FP32 | 13.2 | not measured | not measured | 20.2 MB |

**INT8 is the standout: 2.1× the throughput for 2.6 mAP points (a 7.2% relative
drop), in a quarter of the disk space.** On mAP@0.5 the cost is only 1.0 point.
If this were going into production on this class of hardware, INT8 at 416 would
be the configuration to ship.

The 320 and 512 rows are honestly marked "not measured" — `evaluate.py` can
produce them (`--imgsz 320` with the dynamic model) but the runs were not
performed, and inventing plausible numbers is exactly what this document is
trying to avoid.

### 5.4 Confidence-threshold sensitivity

FP32, 416, same 300 images. `dets/image` is what a viewer actually sees on screen.

| conf | mAP@0.5 | mAP@0.5:0.95 | dets/image |
|---|---|---|---|
| 0.15 | 0.4842 | 0.3351 | 7.49 |
| 0.25 | 0.4532 | 0.3205 | 5.60 |
| **0.30** *(shipped)* | 0.4426 | 0.3149 | 5.04 |
| 0.40 | 0.4181 | 0.3024 | 4.19 |
| 0.60 | 0.3593 | 0.2714 | 2.89 |

(These mAP values are lower than §5.1 by construction: each row scores only the
detections surviving that threshold, so the PR curve is truncated. The row that
matters for comparison is the *shape*, not the absolute value.)

Raising the threshold from 0.15 to 0.60 removes 61% of the boxes and costs 26%
of the mAP@0.5. **0.30 was chosen from this table**: it keeps roughly five
detections per frame — enough to look responsive without the screen filling with
low-confidence clutter — while giving up 8.6% of the mAP@0.5 available at 0.15.
Below 0.25 the live view becomes noticeably noisy; above 0.45 objects held at
arm's length start dropping out.

---

## 6. Limitations

Stated from the measurements above, not from general knowledge about detectors.

**Small objects.** mAP small is **0.1847** against 0.5464 for large — a 3× gap.
This is the model's single biggest weakness and it is structural: at 416×416 a
distant object may occupy a handful of pixels after letterboxing. Concretely,
`book` and `handbag` recall sit near 0.14. Holding an object closer to the
camera genuinely improves detection, which is why the demo shot list includes a
distance shot.

**Motion blur.** Not quantified — there is no labelled motion-blur set here, and
measuring it properly would need one. Qualitatively, on the live camera, fast
hand movement visibly drops detections until the object settles. The demo shot
list deliberately includes this rather than avoiding it. Treating this as a
measured limitation would be overclaiming; it is an observation.

**Low light.** Also unquantified, same reason. The camera's auto-exposure raises
gain, which adds noise, and noise is not what the model was trained on.

**Crowded scenes.** Visible in the numbers: `person` has 93 false positives and
248 false negatives against 685 instances. NMS at IoU 0.45 cannot separate
heavily overlapping instances of the same class — one of two overlapping people
gets suppressed. Raising the IoU threshold trades this for duplicate boxes.

**Recall generally.** 0.490 micro-average recall at conf 0.30. Roughly half of
all labelled objects are missed at the shipped operating point. Lowering the
threshold recovers some of that at the cost of on-screen clutter (§5.4).

**Sustained vs burst performance.** The 19.8 FPS headline is a warm, sustained
figure. The same configuration measures 24–35 FPS on a cold machine. Any FPS
number from this class of hardware that does not say which one it is should be
treated with suspicion — including the ones I measured before adding `--settle`.

**Subset size.** 300 images is enough for stable aggregate mAP but leaves many
classes with single-digit support (§5.2). Per-class figures for rare classes are
not reliable.

**Colour distinctness.** Each of the 80 classes gets a stable unique colour, but
80 hues on one wheel means some pairs are close — `cup` and `tv` land in similar
greens. Alternating brightness helps; it does not fully solve it.

---

## 7. Further optimisation

Three concrete options, with the reasoning for the expected gain. These are
**projections, not measurements**, and are labelled as such.

**1. Ship INT8 as the default (measured: 2.1×).** This is not a projection —
it is in §4.1. The only work is deciding whether the build step belongs in the
setup path or whether a pre-quantised model should be published as a release
asset with its own checksum. Given the accuracy cost is 2.6 mAP points for
double the frame rate, this is the highest-value change available.

**2. OpenVINO execution provider (expected 1.5–2.5× on top of FP32).** This CPU
is Tiger Lake with an Iris Xe iGPU sitting completely idle. `onnxruntime-openvino`
targets both, and Intel's own published figures for this class of CNN on this
generation are in that range. It is a one-line provider change in
`Detector._build_session` — the architecture already isolates it. Not done here
because it is an Intel-specific dependency and this project's default has to
stay platform-neutral, so it would be an opt-in extra rather than the default.
**Unverified — I have not run it.**

**3. Move rendering and encoding off the main thread (expected 4–8%).** Render
plus the MP4 writer is 0.44–2.35 ms of a ~48 ms frame. Moving it to a third
thread would overlap it with the next frame's inference. The gain is small
because the pipeline is 93% inference — which is precisely why this is third on
the list rather than first, and why the two-thread design in the PRD was the
right starting point. Worth doing only if inference gets much cheaper first
(i.e. after option 1 or 2), at which point rendering becomes a larger share.

A fourth option deliberately not pursued: a smaller model. YOLOX-Nano would be
roughly 4× cheaper by parameter count, but its published mAP is 25.8 against
32.8 for Tiny. Given Tiny already clears the 15 FPS requirement by 32% while
warm, spending 7 mAP points to buy frame rate nobody asked for is a bad trade.
`models/download_weights.py --model nano` fetches it for anyone whose hardware
disagrees.
