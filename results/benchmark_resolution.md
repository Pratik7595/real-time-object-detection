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
| imgsz 320 | yolox_tiny_dynamic.onnx | 320 | 640x480 | 32.4 | 33.0 | 42.4 | 24.9 | 28.8 | 30.8 | 398 | 129 | 8.80 |
| imgsz 416 | yolox_tiny.onnx | 416 | 640x480 | 19.8 | 19.8 | 25.4 | 16.1 | 47.1 | 50.5 | 397 | 142 | 9.90 |
| imgsz 512 | yolox_tiny_dynamic.onnx | 512 | 640x480 | 13.2 | 13.1 | 17.1 | 11.1 | 71.8 | 76.0 | 397 | 166 | 11.00 |

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | total ms |
|---|---|---|---|---|---|---|
| imgsz 320 | 0.03 | 0.58 | 28.75 | 1.05 | 0.41 | 30.81 |
| imgsz 416 | 0.03 | 1.20 | 47.11 | 1.72 | 0.44 | 50.49 |
| imgsz 512 | 0.03 | 1.47 | 71.79 | 2.19 | 0.47 | 75.95 |
