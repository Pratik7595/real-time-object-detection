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
| A baseline FP32 (naive preprocess, per-frame alloc) | yolox_tiny.onnx | 416 | 640x480 | 20.1 | 20.6 | 23.7 | 15.9 | 44.0 | 49.7 | 399 | 144 | 9.90 |
| B + preallocated preprocess buffers | yolox_tiny.onnx | 416 | 640x480 | 21.6 | 22.8 | 24.8 | 16.4 | 43.2 | 46.3 | 399 | 141 | 9.90 |
| C + OpenCV capped to 2 threads (rejected) | yolox_tiny.onnx | 416 | 640x480 | 22.1 | 22.9 | 25.9 | 17.2 | 41.9 | 45.2 | 399 | 141 | 9.90 |
| D + INT8 quantisation  [shipped] | yolox_tiny_int8.onnx | 416 | 640x480 | 43.2 | 45.0 | 47.6 | 35.4 | 20.1 | 23.1 | 401 | 112 | 9.90 |
| E + --infer-every 2 | yolox_tiny_int8.onnx | 416 | 640x480 | 85.3 | n/a | n/a | n/a | 9.9 | 11.7 | 401 | 112 | 9.90 |

> `n/a`: with `--infer-every > 1` the per-frame time is bimodal, so median/p95 of per-frame FPS are meaningless. Mean is throughput (frames / wall clock) and is still valid. Note also that inference ms is the average over *all* frames, including the skipped ones.

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | total ms |
|---|---|---|---|---|---|---|
| A baseline FP32 (naive preprocess, per-frame alloc) | 0.03 | 3.53 | 43.97 | 1.75 | 0.44 | 49.71 |
| B + preallocated preprocess buffers | 0.03 | 1.04 | 43.17 | 1.61 | 0.44 | 46.28 |
| C + OpenCV capped to 2 threads (rejected) | 0.03 | 1.19 | 41.90 | 1.69 | 0.43 | 45.22 |
| D + INT8 quantisation  [shipped] | 0.03 | 1.14 | 20.06 | 1.48 | 0.40 | 23.10 |
| E + --infer-every 2 | 0.02 | 0.56 | 9.94 | 0.73 | 0.43 | 11.69 |
