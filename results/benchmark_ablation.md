# Benchmark: ablation

```
platform : Windows-11-10.0.26200-SP0
python   : 3.12.10 (AMD64)
cpu      : Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
cores    : 4 physical / 8 logical
ram      : 8.3 GB
ort      : 1.29.0
```

| Config | Model | imgsz | Capture | FPS mean | FPS median | FPS p95 | FPS p5 | Infer ms | Total ms | CPU % | Peak RSS MB | Dets/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A baseline (naive preprocess, per-frame allocation) | yolox_tiny.onnx | 416 | 640x480 | 19.2 | 19.1 | 24.3 | 15.7 | 46.3 | 51.9 | 399 | 142 | 9.90 |
| B + preallocated preprocess buffers  [shipped] | yolox_tiny.onnx | 416 | 640x480 | 20.9 | 21.5 | 25.8 | 16.6 | 44.6 | 47.7 | 400 | 140 | 9.90 |
| C + OpenCV capped to 2 threads (rejected) | yolox_tiny.onnx | 416 | 640x480 | 19.4 | 19.5 | 25.0 | 15.3 | 48.0 | 51.5 | 396 | 142 | 9.90 |
| D + --infer-every 2 | yolox_tiny.onnx | 416 | 640x480 | 40.6 | n/a | n/a | n/a | 22.6 | 24.6 | 400 | 141 | 9.90 |
| E + INT8 quantisation (models/quantize.py) | yolox_tiny_int8.onnx | 416 | 640x480 | 42.1 | 43.0 | 48.6 | 34.5 | 20.7 | 23.7 | 400 | 111 | 9.90 |

> `n/a`: with `--infer-every > 1` the per-frame time is bimodal, so median/p95 of per-frame FPS are meaningless. Mean is throughput (frames / wall clock) and is still valid. Note also that inference ms is the average over *all* frames, including the skipped ones.

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | total ms |
|---|---|---|---|---|---|---|
| A baseline (naive preprocess, per-frame allocation) | 0.03 | 3.42 | 46.33 | 1.69 | 0.42 | 51.89 |
| B + preallocated preprocess buffers  [shipped] | 0.03 | 1.01 | 44.64 | 1.62 | 0.42 | 47.72 |
| C + OpenCV capped to 2 threads (rejected) | 0.04 | 1.24 | 48.01 | 1.79 | 0.46 | 51.53 |
| D + --infer-every 2 | 0.02 | 0.61 | 22.63 | 0.86 | 0.47 | 24.60 |
| E + INT8 quantisation (models/quantize.py) | 0.03 | 1.12 | 20.68 | 1.48 | 0.41 | 23.72 |
