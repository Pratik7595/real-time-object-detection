# Benchmark: resolution

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
| imgsz 416 | yolox_tiny.onnx | 416 | 640x480 | 24.3 | 25.6 | 38.3 | 15.9 | 38.2 | 41.0 | 399 | 139 | 10.20 |
| imgsz 416 | yolox_tiny.onnx | 416 | 640x480 | 21.0 | 22.3 | 25.1 | 16.3 | 44.2 | 47.5 | 399 | 141 | 10.20 |
| imgsz 416 | yolox_tiny.onnx | 416 | 640x480 | 20.4 | 21.0 | 25.2 | 16.4 | 45.7 | 48.9 | 399 | 141 | 10.20 |
| imgsz 416 | yolox_tiny.onnx | 416 | 640x480 | 19.4 | 20.0 | 25.2 | 14.7 | 48.2 | 51.5 | 396 | 142 | 10.20 |

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | total ms |
|---|---|---|---|---|---|---|
| imgsz 416 | 0.02 | 1.05 | 38.15 | 1.42 | 0.38 | 41.03 |
| imgsz 416 | 0.03 | 1.21 | 44.19 | 1.60 | 0.45 | 47.47 |
| imgsz 416 | 0.03 | 1.17 | 45.69 | 1.54 | 0.42 | 48.85 |
| imgsz 416 | 0.03 | 1.24 | 48.21 | 1.61 | 0.44 | 51.53 |
