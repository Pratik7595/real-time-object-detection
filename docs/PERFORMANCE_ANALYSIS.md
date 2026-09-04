# Performance analysis

Every number here was produced by a script in this repository. The command that
generates each table is given above it, and the raw output is committed under
`results/`. Nothing is estimated, extrapolated or copied from a paper — where a
published figure appears it is labelled as published and attributed.

**The shipped default is the INT8 model**, so it is what the headline tables
measure. FP32 is reported alongside wherever the comparison is the point.

Three things this document is careful about:

* **Speed and accuracy were measured on different inputs.** FPS comes from a
  fixed image loop and a live camera. Accuracy comes from 300 labelled COCO
  val2017 images. No accuracy figure anywhere is derived from webcam footage,
  because webcam footage has no ground truth.
* **This laptop throttles.** It is a thin 15 W chassis and the sustained clock is
  well below the boost clock. That turned out to matter more than most of the
  optimisations, so it is documented first rather than buried.
* **The INT8 model is not calibrated on the evaluation set.** That distinction is
  easy to get wrong and it is why §5 can be trusted; see §5.0.

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
| Camera | HP TrueVision HD, built-in, 640×480, nominal 30 fps |

### How a run works

`src/benchmark.py` runs a fixed frame count per configuration, discards the
first 30 frames, and reports the rest. Specifics that matter:

- **Warm-up is discarded, not deleted.** ONNX Runtime's first `run()` allocates
  its arena and spins up its thread pool; OpenCV's first `resize` and first
  `putText` are also one-off costs. On this machine frame 0 costs ~625 ms
  against ~26 ms for every frame after it. Those frames stay in the per-run CSV
  and are excluded from the summary.
- **The default source is `assets/sample.jpg`, looped, not the camera.** This is
  more important than it first looks — see §3.1, where the camera turns out to
  be the binding constraint on live frame rate. The consequence is that the
  `capture` column here is ~0.03 ms and is *not* representative of live capture.
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

Raw output: [`results/throttling_repeat.md`](../results/throttling_repeat.md)
(measured on FP32, before INT8 became the default).

A 20% decline with nothing changed but time. This is the package dropping off
its boost clock, and it is larger than most of the optimisations measured below.
Without accounting for it, whichever configuration runs first in a sweep wins,
and the "effect" measured is really the running order.

So `benchmark.py` takes a `--settle` argument (default 20 s) that loads the CPU
to a steady thermal state *before* the first measurement. Every table below was
produced with `--settle 25`.

**Where a difference is only a few percent, settling is not enough**, because
drift within a sweep is the same size as the effect. Those comparisons were
re-run as counterbalanced A/B tests — see §4.2, which is also a worked example
of a sequential ablation row pointing the *opposite* way to the paired test.

---

## 2. FPS across resolutions

```bash
python -m src.benchmark --frames 300 --warmup 30 --settle 25
python -m src.benchmark --frames 300 --warmup 30 --settle 25 --model models/yolox_tiny.onnx
```

Raw output: [`results/benchmark_resolution.md`](../results/benchmark_resolution.md)

Capture is 640×480 throughout; the variable is the network input. Sizes other
than 416 use the variable-input graph from `models/make_dynamic.py`.

### INT8 (shipped default)

| Model input | Device | FPS mean | FPS median | FPS p95 | FPS p5 | Inference ms | Frame ms | CPU % | Peak RSS MB | Dets/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| 320×320 | CPU (i5-1135G7) | **53.7** | 54.4 | 66.3 | 44.1 | 16.27 | 18.57 | 396 | 105 | 8.80 |
| **416×416** (shipped) | CPU (i5-1135G7) | **38.0** | 38.4 | 46.0 | 31.4 | 22.85 | 26.26 | 399 | 112 | 9.90 |
| 512×512 | CPU (i5-1135G7) | **27.6** | 27.9 | 32.7 | 23.2 | 31.58 | 36.20 | 396 | 121 | 11.00 |

### FP32, same sweep

| Model input | FPS mean | FPS median | FPS p95 | FPS p5 | Inference ms | Frame ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|
| 320×320 | 32.4 | 33.0 | 42.4 | 24.9 | 28.75 | 30.81 | 129 |
| 416×416 | 19.8 | 19.8 | 25.4 | 16.1 | 47.11 | 50.49 | 142 |
| 512×512 | 13.2 | 13.1 | 17.1 | 11.1 | 71.79 | 75.95 | 166 |

CPU % is relative to a single core, so 398% means all four cores saturated.

Reading this honestly:

- **The >15 FPS requirement is met at every INT8 resolution**, with 2.5×
  headroom at the shipped 416. FP32 clears it at 320 and 416 and **misses at
  512** (13.2) — which is why 512 was in the original table at all: it showed
  where the FP32 approach ran out. INT8 moves that boundary past 512.
- Even the worst INT8 p5 (23.2 FPS at 512) clears 15, so the requirement holds
  on the slow tail and not just on the average.
- **Quantisation helps more at large inputs than small ones**: 1.66× at 320,
  1.92× at 416, 2.09× at 512. Bigger tensors spend proportionally more time in
  convolution, which is exactly what INT8 makes cheaper.
- Detections per frame are *identical* between FP32 and INT8 at each resolution
  (8.80 / 9.90 / 11.00 on this image). Quantisation shifts confidences, not the
  gross number of objects found — the accuracy cost shows up in localisation and
  ranking, which is what §5 measures.

---

## 3. Latency breakdown per pipeline stage

Same runs as §2. All values are milliseconds, mean over 300 frames.

### INT8 (shipped)

| Configuration | capture | preprocess | inference | postprocess | render | **total** |
|---|---|---|---|---|---|---|
| 320×320 | 0.12 | 0.60 | 16.27 | 1.14 | 0.44 | **18.57** |
| 416×416 (shipped) | 0.03 | 1.22 | 22.85 | 1.71 | 0.46 | **26.26** |
| 512×512 | 0.03 | 1.53 | 31.58 | 2.58 | 0.48 | **36.20** |

### FP32

| Configuration | capture | preprocess | inference | postprocess | render | **total** |
|---|---|---|---|---|---|---|
| 320×320 | 0.03 | 0.58 | 28.75 | 1.05 | 0.41 | **30.81** |
| 416×416 | 0.03 | 1.20 | 47.11 | 1.72 | 0.44 | **50.49** |
| 512×512 | 0.03 | 1.47 | 71.79 | 2.19 | 0.47 | **75.95** |

**Inference is 87% of the frame budget under INT8 at 416, down from 93% under
FP32.** Everything else together is 3.4 ms in both cases — quantisation did not
make preprocessing or NMS cheaper, it just made inference stop dominating quite
so completely. That shift is what makes §7's third option worth reconsidering.

### 3.1 Live camera — and why it is now the bottleneck

```bash
python -m src.main --source 0 --no-display --max-frames 200
```

| Stage | Image loop (416 INT8) | Live camera (416 INT8) | Live camera (416 FP32, earlier) |
|---|---|---|---|
| capture | 0.03 ms | **48.20 ms** | 0.11 ms |
| preprocess | 1.22 ms | 0.75 ms | 0.80 ms |
| inference | 22.85 ms | 16.59 ms | 41.44 ms |
| postprocess | 1.71 ms | 0.86 ms | 0.89 ms |
| render | 0.46 ms | 0.26 ms | 0.26 ms |
| **FPS mean** | **38.0** | **15.0** | **23.0** |

The live INT8 run is *slower* than the live FP32 run. That is not a regression,
and it is worth spelling out because it is the most counter-intuitive number in
this document.

I measured the camera on its own, with no inference in the loop at all:

| Trial | Frames | Wall clock | Delivered fps | Frames dropped |
|---|---|---|---|---|
| 1 | 120 | 8.00 s | **15.0** | 0 |
| 2 | 120 | 8.00 s | **15.0** | 0 |

The webcam was delivering exactly 15.0 fps. Its nominal rate is 30, but it had
dropped to 15 as the room darkened over the evening — a standard auto-exposure
behaviour, where longer exposures halve the sensor's frame rate. The earlier
FP32 live run happened in better light, when the camera was still supplying
~30 fps and the detector (24 FPS capable) was the limit; the `capture` cost of
0.11 ms in that column is what "the camera is outrunning us" looks like.

Three conclusions:

1. **Live FPS is now camera-bound, not compute-bound.** 48.2 ms of a 66.7 ms
   frame is spent waiting for the sensor. Making inference faster than the
   camera's frame interval cannot raise live frame rate any further.
2. **The drop-stale buffer is doing its job**: zero frames dropped, because the
   consumer is now faster than the producer. When the camera was the faster of
   the two, the same code kept latency bounded by discarding stale frames.
3. **A live FPS figure measures your camera and your lighting** as much as it
   measures this code. That is precisely why `benchmark.py` defaults to a
   deterministic file source, and why the headline numbers in §2 come from there.

### 3.2 Cost of `--record`

Enabling the MP4 writer moves the `render` stage from 0.44 ms to 2.35 ms — about
**1.9 ms per frame, roughly 4%** on FP32 and ~7% on the faster INT8 pipeline.
There is also a one-off ~20 ms stall on the frame where the writer opens (codec
initialisation), which is why the first measurement of this looked like a 25%
penalty until the writer was forced open from frame 0 to isolate it.

---

## 4. Effect of each optimisation

```bash
python -m src.benchmark --ablation --frames 300 --warmup 30 --settle 25
```

Raw output: [`results/benchmark_ablation.md`](../results/benchmark_ablation.md)

### 4.1 Before / after

Rows are cumulative and start from unoptimised FP32, so row A is a genuine
"before" rather than the shipped article.

| # | Change | Model | FPS mean | Δ vs A | preprocess ms | inference ms | frame ms | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| A | Baseline: allocate preprocessing buffers per frame | FP32 | 20.1 | — | 3.53 | 43.97 | 49.71 | 144 |
| B | + preallocated buffers | FP32 | 21.6 | +7.5% | **1.04** | 43.17 | 46.28 | 141 |
| C | + cap OpenCV to 2 threads | FP32 | 22.1 | +10.0% | 1.19 | 41.90 | 45.22 | 141 |
| D | **+ INT8 quantisation** *(shipped)* | INT8 | **43.2** | **+115%** | 1.14 | **20.06** | 23.10 | **112** |
| E | + `--infer-every 2` | INT8 | 85.3 | +324% | 0.56 | 9.94 | 11.69 | 112 |

### 4.2 What actually worked, and what did not

**INT8 quantisation — the change worth making (+115% over baseline, 1.9–2.1×
over like-for-like FP32).** Static quantisation calibrated on the 24 images in
`assets/calib/`. Inference falls from 43.2 ms to 20.1 ms, the model file from
20.2 MB to 5.2 MB, and peak RSS from 141 MB to 112 MB. The accuracy cost is 2.5
mAP@0.5:0.95 points (§5). It is now the default, and the ~8-second build runs as
part of `models/download_weights.py`.

**Preallocated preprocessing buffers — kept (+7.5%).** Reusing the resize
target, letterbox canvas and NCHW blob across frames cut preprocessing from
3.53 ms to 1.04 ms, a **71% reduction in that stage**, for a 7.5% end-to-end
gain. A large relative win on a small stage is a small absolute win. Worth
having; not worth mistaking for the main event.

**Capping OpenCV threads — still rejected, and a useful lesson in how not to
read an ablation table.** Row C above says capping *helped* by 2.3% over row B.
The earlier FP32 ablation said it *hurt* by 7.2%. Both are single sequential
runs a few percent apart, which is exactly the resolution limit established in
§1. So it was re-run properly, counterbalanced (ABBA ordering, four pairs), on
the INT8 model that now ships:

| Round | Order | OpenCV default | OpenCV capped to 2 |
|---|---|---|---|
| 1 | A→B | 35.95 | 37.50 |
| 2 | B→A | 35.42 | 34.05 |
| 3 | A→B | 39.59 | 34.66 |
| 4 | B→A | 39.59 | 37.80 |
| **mean** | | **37.64** | **36.00** |

Capping measures **−4.3%**, reproducing the direction of the earlier FP32 test
(−5.0%). But note the spread *within* each arm: 35.4–39.6 for the default. The
effect is the same order as the run-to-run noise, so the honest conclusion is
**"no evidence that capping helps"**, not "capping is definitely harmful". The
default stays `0` (leave OpenCV alone) on that basis. The knob remains for
machines that are busy doing something else. I have not established a mechanism;
the plausible story is that `cv2.setNumThreads` reconfigures a threading runtime
shared with ORT, but that is a hypothesis, not a measurement.

**Frame skipping (`--infer-every 2`) — available, off by default (+97% over
row D).** Doubles throughput by doing half the work; boxes on skipped frames are
one frame stale and the HUD says so on screen. Documented as a fallback for
hardware that cannot clear 15 FPS, not as a default, because it improves the
number without improving what the user sees. At 85 FPS on this machine it is
well past the point of usefulness anyway.

### 4.3 ONNX Runtime thread scaling

```bash
python -m src.benchmark --threads-sweep --frames 200 --warmup 30 --settle 25
```

Raw output: [`results/benchmark_threads.md`](../results/benchmark_threads.md)

Measured on the shipped INT8 model:

| intra_op_num_threads | FPS mean | Inference ms | CPU % | FPS per 100% CPU |
|---|---|---|---|---|
| ORT default *(shipped)* | 42.6 | 20.32 | 399 | 10.7 |
| 1 | 33.3 | 27.95 | 101 | 33.0 |
| 2 | 42.1 | 21.32 | 199 | 21.2 |
| 4 | 42.8 | 20.26 | 404 | 10.6 |
| 8 | 33.1 | 26.01 | 767 | 4.3 |

Quantisation changed the shape of this curve, and in a useful direction:

- **INT8 scales worse with threads than FP32 did** — 1 → 4 threads gives 1.29×
  here against 1.56× on FP32. The model is now cheap enough per layer that
  thread-dispatch overhead is a visible share of the work.
- **`intra_op=2` reaches 98.4% of the best result on half the cores.** On a
  laptop running on battery, or a machine doing anything else, that is the
  configuration I would actually pick.
- **8 threads is now clearly harmful** (−22.7% against 4, at 1.9× the CPU),
  where on FP32 it was merely pointless. Hyperthread contention costs more when
  each thread's slice of work is smaller.

The shipped config still leaves `intra_op_threads: 0`, because ORT's own default
lands within noise of the best here and hardcoding `4` would be wrong on a
reviewer's 8- or 16-core machine. `--threads 2` is a one-flag change for anyone
who cares more about CPU headroom than the last 2%.

---

## 5. Accuracy

> **Source: COCO val2017, first 300 images by image id, with human ground-truth
> labels. These are not webcam measurements.** No accuracy figure in this
> repository is derived from live camera footage.

```bash
python scripts/download_coco_subset.py --images 300
python -m src.evaluate                                        # INT8 (default)
python -m src.evaluate --model models/yolox_tiny.onnx         # FP32
```

Raw output: [`results/accuracy_int8.md`](../results/accuracy_int8.md) ·
[`results/accuracy_fp32.md`](../results/accuracy_fp32.md)

### 5.0 Two methodology points

**Calibration data is disjoint from evaluation data.** Static quantisation
derives its activation ranges from real images. The first version of this model
was calibrated on the first 64 images of `data/coco_subset` — which is the
evaluation set. That is test-set contamination: the quantised model had been
tuned on the data it was then scored against. The shipped model calibrates on
the 24 images in `assets/calib/`, selected by
`scripts/build_calibration_set.py` from val2017 images *outside* the first 300.

The measured size of that bias, for the record:

| | mAP@0.5:0.95 | mAP@0.5 |
|---|---|---|
| Calibrated on the eval set (wrong) | 0.3312 | 0.5226 |
| Calibrated on held-out images (shipped) | **0.3316** | **0.5183** |

0.0004 on the primary metric — negligible, and it happened to land in the
*unfavourable* direction on mAP@0.5:0.95. The number barely moved; the method
still had to be fixed, because "it turned out not to matter" is only knowable
after you check.

**Two different confidence thresholds are in play.** Detections are gathered
once at `conf=0.001` and thresholded afterwards. mAP is the area under the
precision-recall curve and needs the low-confidence tail to trace that curve;
evaluating at the app's 0.30 default would truncate it and understate mAP
substantially. Precision, recall and F1 are operating-point metrics and are
reported at a stated threshold (0.30).

### 5.1 Headline

| Metric | **INT8 (shipped)** | FP32 | Δ |
|---|---|---|---|
| **mAP@0.5** | **0.5183** | 0.5326 | −0.0143 (−2.7%) |
| **mAP@0.5:0.95** | **0.3316** | 0.3568 | −0.0252 (−7.1%) |
| mAP@0.75 | 0.3544 | 0.3827 | −0.0283 |
| mAP small | 0.1717 | 0.1847 | −0.0130 |
| mAP medium | 0.3247 | 0.3582 | −0.0335 |
| mAP large | 0.5077 | 0.5464 | −0.0387 |

The pattern is worth noting: **the loss is much larger at strict IoU than loose**
(−7.4% at IoU 0.75 vs −2.7% at IoU 0.5). Quantisation mostly costs *localisation
precision* — boxes end up slightly less well fitted — rather than causing the
model to miss objects outright. That matches §2, where detections-per-frame were
identical between the two models.

For context, the YOLOX authors publish **32.8** mAP@0.5:0.95 for YOLOX-Tiny on
the *full* 5000-image val2017 set. FP32 here measures 35.7 on a 300-image
subset. The gap is subset sampling error, not a better model — which is why
every table says "300-image subset" rather than quoting these as COCO val2017
results.

### 5.2 Per-class precision / recall / F1

Shipped INT8, at `conf=0.30`, `IoU=0.50`. Full table in
[`results/accuracy_int8.csv`](../results/accuracy_int8.csv):

| Class | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| person | 685 | 430 | 104 | 255 | 0.805 | 0.628 | 0.706 |
| chair | 119 | 43 | 35 | 76 | 0.551 | 0.361 | 0.437 |
| book | 75 | 12 | 9 | 63 | 0.571 | 0.160 | 0.250 |
| cup | 69 | 28 | 19 | 41 | 0.596 | 0.406 | 0.483 |
| car | 67 | 37 | 15 | 30 | 0.712 | 0.552 | 0.622 |
| bottle | 66 | 19 | 24 | 47 | 0.442 | 0.288 | 0.349 |
| dining table | 48 | 26 | 28 | 22 | 0.482 | 0.542 | 0.510 |
| **micro-average** | **2145** | **1035** | **504** | **1110** | **0.673** | **0.483** | **0.562** |

FP32 micro-average for comparison: **0.694 / 0.490 / 0.574**. Quantisation costs
about 2 points of precision and 0.7 of recall at this operating point.

The shape is consistent and worth stating plainly: **precision (0.67) is much
better than recall (0.48)**. At a 0.30 threshold the model is conservative —
what it reports is usually right, but it misses about half of everything
labelled. `book` (recall 0.160) is the clearest failure, and it is dominated by
small, cluttered, overlapping instances.

Some classes score 0.000 on a subset this small — `cow` (9 GT), `donut` (8),
`orange` (4), `scissors` (2). With that little support these are anecdotes, not
measurements, and should not be read as "the model cannot detect oranges".

### 5.3 Speed / accuracy trade-off

| Configuration | FPS mean | mAP@0.5:0.95 | mAP@0.5 | Model size |
|---|---|---|---|---|
| 320×320 INT8 | 53.7 | not measured | not measured | 5.2 MB |
| **416×416 INT8** *(shipped)* | **38.0** | **0.3316** | **0.5183** | **5.2 MB** |
| 512×512 INT8 | 27.6 | not measured | not measured | 5.2 MB |
| 320×320 FP32 | 32.4 | not measured | not measured | 20.2 MB |
| 416×416 FP32 | 19.8 | 0.3568 | 0.5326 | 20.2 MB |
| 512×512 FP32 | 13.2 | not measured | not measured | 20.2 MB |

**The decision this table drove: INT8 at 416 gives 1.9× the frame rate of FP32
at 416 for 2.5 mAP@0.5:0.95 points, in a quarter of the disk space.** On a
CPU-only target that is a clearly favourable trade, so it became the default.

Note also that **INT8 at 512 (27.6 FPS, higher input resolution) beats FP32 at
416 (19.8 FPS) outright** — more pixels *and* more speed. Whether it beats it on
accuracy is untested, and is the single most interesting unmeasured question
left here.

The rows marked "not measured" are honest: `evaluate.py` can produce them
(`--imgsz 320` with the dynamic model) but the runs were not performed, and
inventing plausible numbers is exactly what this document is trying to avoid.

### 5.4 Confidence-threshold sensitivity

Shipped INT8, 416, same 300 images. `dets/image` is what a viewer sees on screen.

| conf | mAP@0.5 | mAP@0.5:0.95 | dets/image |
|---|---|---|---|
| 0.15 | 0.4723 | 0.3100 | 7.62 |
| 0.25 | 0.4472 | 0.2992 | 5.78 |
| **0.30** *(shipped)* | 0.4353 | 0.2934 | 5.13 |
| 0.40 | 0.4078 | 0.2808 | 4.17 |
| 0.60 | 0.3560 | 0.2540 | 2.94 |

(These mAP values are lower than §5.1 by construction: each row scores only the
detections surviving that threshold, so the PR curve is truncated. What matters
for comparison is the shape, not the absolute value.)

Raising the threshold from 0.15 to 0.60 removes 61% of the boxes and costs 25%
of the mAP@0.5. **0.30 was chosen from this table**: it keeps roughly five
detections per frame — enough to look responsive without the screen filling with
low-confidence clutter — while giving up 7.8% of the mAP@0.5 available at 0.15.
Below 0.25 the live view becomes noticeably noisy; above 0.45 objects held at
arm's length start dropping out. The FP32 curve has the same shape.

---

## 6. Limitations

Stated from the measurements above, not from general knowledge about detectors.

**Small objects.** INT8 mAP small is **0.1717** against 0.5077 for large — a
nearly 3× gap. This is the model's single biggest weakness and it is structural:
at 416×416 a distant object may occupy a handful of pixels after letterboxing.
Concretely, `book` recall sits at 0.160. Holding an object closer to the camera
genuinely improves detection, which is why the demo shot list includes a
distance shot.

**Live frame rate is limited by the camera, not this code.** Measured directly:
15.0 fps delivered with no inference running (§3.1). Any claim that this system
"runs at N FPS live" is a claim about the camera and the lighting.

**Localisation under quantisation.** INT8 loses 7.4% of mAP@0.75 against 2.7% of
mAP@0.5. Boxes are slightly looser. If precise box geometry matters more than
frame rate for your use, run FP32.

**Motion blur.** Not quantified — there is no labelled motion-blur set here, and
measuring it properly would need one. Qualitatively, on the live camera, fast
hand movement visibly drops detections until the object settles. Treating this
as a measured limitation would be overclaiming; it is an observation. Note that
a camera at 15 fps in low light also has a longer exposure per frame, so blur
and low light are not independent problems.

**Low light.** Also unquantified, same reason — but it is now known to have a
second-order effect that *is* measured: it halved the camera's frame rate.

**Crowded scenes.** Visible in the numbers: `person` has 104 false positives and
255 false negatives against 685 instances. NMS at IoU 0.45 cannot separate
heavily overlapping instances of the same class — one of two overlapping people
gets suppressed. Raising the IoU threshold trades this for duplicate boxes.

**Recall generally.** 0.483 micro-average recall at conf 0.30. Roughly half of
all labelled objects are missed at the shipped operating point. Lowering the
threshold recovers some of that at the cost of on-screen clutter (§5.4).

**Sustained vs burst performance.** All headline figures are warm and sustained.
The same configuration measures materially higher on a cold machine. Any FPS
number from this class of hardware that does not say which one it is should be
treated with suspicion — including the ones I measured before adding `--settle`.

**Subset size.** 300 images is enough for stable aggregate mAP but leaves many
classes with single-digit support (§5.2). Per-class figures for rare classes are
not reliable.

**INT8 reproducibility.** The quantised model is built locally, not downloaded,
so it carries no checksum and is not guaranteed bit-identical across ONNX
Runtime versions. The calibration inputs are pinned (24 committed images, fixed
order); the quantisation implementation is not.

**Colour distinctness.** Each of the 80 classes gets a stable unique colour, but
80 hues on one wheel means some pairs are close — `cup` and `tv` land in similar
greens. Alternating brightness helps; it does not fully solve it.

---

## 7. Further optimisation

Three concrete options. These are **projections, not measurements**, and are
labelled as such. The obvious one — INT8 — has already been done and moved into
the default, so what remains is genuinely speculative.

**1. A better camera, or more light (measured constraint, not a projection).**
The single largest limit on live frame rate right now is the sensor delivering
15 fps. The detector has 38 FPS of capability and 23 of it is going unused. This
is first on the list because it is the only item here backed by a measurement
(§3.1), and because it costs nothing to try.

**2. OpenVINO execution provider (expected 1.3–2× on top of INT8).** This CPU is
Tiger Lake with an Iris Xe iGPU sitting completely idle. `onnxruntime-openvino`
targets both and has its own INT8 path. It is a one-line provider change in
`Detector._build_session` — the architecture already isolates it. Not done here
because it is an Intel-specific dependency and this project's default has to
stay platform-neutral, so it would be an opt-in extra. I would expect a smaller
multiplier than on FP32, since INT8 has already collected much of the available
win. **Unverified — I have not run it.**

**3. Move rendering and encoding off the main thread (expected 2–9%).** Render
plus the MP4 writer is 0.46–2.35 ms of a ~26 ms frame. Moving it to a third
thread would overlap it with the next frame's inference. This was third on the
list when the pipeline was 93% inference and FP32; at 87% and INT8 the case is
slightly stronger, and it becomes stronger still if option 2 lands. Still not
first, because the numbers say the camera is.

A fourth option deliberately not pursued: a smaller model. YOLOX-Nano would be
roughly 4× cheaper by parameter count, but its published mAP is 25.8 against
32.8 for Tiny. Given INT8 Tiny already clears the 15 FPS requirement by 2.5× at
the default resolution — and clears it at 512 as well — spending 7 mAP points to
buy frame rate the camera cannot supply would be a bad trade.
`models/download_weights.py --model nano` fetches it for anyone whose hardware
disagrees.
