# Benchmark: capture

```
platform : Windows-11-10.0.26200-SP0
python   : 3.12.10 (AMD64)
cpu      : Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
cores    : 4 physical / 8 logical
ram      : 8.3 GB
ort      : 1.29.0
```

| Config | Model | imgsz | Capture | FPS mean | FPS median | FPS p95 | FPS p5 | Infer ms | Compute ms | CPU % | Peak RSS MB | Dets/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capture 640x480 | yolox_tiny_int8.onnx | 416 | 640x480 | 29.9 | 30.9 | 37.0 | 21.2 | 21.2 | 33.4 | 373 | 124 | 3.57 |
| capture 960x540 (camera gave 640x480) | yolox_tiny_int8.onnx | 416 | 640x480 | 29.9 | 30.8 | 38.2 | 21.8 | 20.6 | 33.4 | 380 | 128 | 4.23 |
| capture 1280x720 | yolox_tiny_int8.onnx | 416 | 1280x720 | 29.9 | 30.7 | 39.0 | 21.6 | 22.2 | 33.4 | 423 | 136 | 3.60 |

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | compute ms |
|---|---|---|---|---|---|---|
| capture 640x480 | 9.52 | 1.09 | 21.24 | 1.33 | 0.20 | 33.38 |
| capture 960x540 (camera gave 640x480) | 10.06 | 1.13 | 20.56 | 1.37 | 0.25 | 33.37 |
| capture 1280x720 | 8.35 | 1.02 | 22.24 | 1.50 | 0.26 | 33.37 |
