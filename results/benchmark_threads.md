# Benchmark: threads

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
| ORT default | yolox_tiny.onnx | 416 | 640x480 | 19.1 | 18.9 | 25.4 | 16.0 | 48.7 | 52.2 | 397 | 139 | 10.35 |
| intra_op=1 | yolox_tiny.onnx | 416 | 640x480 | 13.1 | 13.1 | 14.4 | 11.7 | 73.9 | 76.2 | 100 | 140 | 10.35 |
| intra_op=2 | yolox_tiny.onnx | 416 | 640x480 | 16.6 | 17.0 | 18.9 | 13.7 | 57.5 | 60.3 | 199 | 140 | 10.35 |
| intra_op=4 | yolox_tiny.onnx | 416 | 640x480 | 20.4 | 20.5 | 25.1 | 16.7 | 45.6 | 49.0 | 398 | 141 | 10.35 |
| intra_op=8 | yolox_tiny.onnx | 416 | 640x480 | 19.3 | 20.5 | 23.4 | 13.8 | 46.9 | 51.7 | 757 | 141 | 10.35 |

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | total ms |
|---|---|---|---|---|---|---|
| ORT default | 0.03 | 1.34 | 48.73 | 1.70 | 0.42 | 52.22 |
| intra_op=1 | 0.02 | 0.86 | 73.92 | 1.13 | 0.30 | 76.24 |
| intra_op=2 | 0.03 | 0.95 | 57.54 | 1.40 | 0.35 | 60.27 |
| intra_op=4 | 0.04 | 1.20 | 45.56 | 1.73 | 0.44 | 48.97 |
| intra_op=8 | 0.04 | 1.60 | 46.93 | 2.48 | 0.59 | 51.65 |
