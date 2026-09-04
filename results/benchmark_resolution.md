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
| imgsz 320 | yolox_tiny_int8_dynamic.onnx | 320 | 640x480 | 53.7 | 54.4 | 66.3 | 44.1 | 16.3 | 18.6 | 396 | 105 | 8.80 |
| imgsz 416 | yolox_tiny_int8.onnx | 416 | 640x480 | 38.0 | 38.4 | 46.0 | 31.4 | 22.8 | 26.3 | 399 | 112 | 9.90 |
| imgsz 512 | yolox_tiny_int8_dynamic.onnx | 512 | 640x480 | 27.6 | 27.9 | 32.7 | 23.2 | 31.6 | 36.2 | 396 | 121 | 11.00 |

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | total ms |
|---|---|---|---|---|---|---|
| imgsz 320 | 0.12 | 0.60 | 16.27 | 1.14 | 0.44 | 18.57 |
| imgsz 416 | 0.03 | 1.22 | 22.85 | 1.71 | 0.46 | 26.26 |
| imgsz 512 | 0.03 | 1.53 | 31.58 | 2.58 | 0.48 | 36.20 |
