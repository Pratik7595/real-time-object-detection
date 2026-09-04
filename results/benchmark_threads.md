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
| ORT default | yolox_tiny_int8.onnx | 416 | 640x480 | 42.6 | 44.6 | 47.3 | 34.8 | 20.3 | 23.4 | 399 | 109 | 10.35 |
| intra_op=1 | yolox_tiny_int8.onnx | 416 | 640x480 | 33.3 | 34.0 | 35.1 | 30.1 | 27.9 | 30.0 | 101 | 110 | 10.35 |
| intra_op=2 | yolox_tiny_int8.onnx | 416 | 640x480 | 42.1 | 42.8 | 43.4 | 38.8 | 21.3 | 23.7 | 199 | 110 | 10.35 |
| intra_op=4 | yolox_tiny_int8.onnx | 416 | 640x480 | 42.8 | 44.6 | 47.6 | 34.7 | 20.3 | 23.3 | 404 | 109 | 10.35 |
| intra_op=8 | yolox_tiny_int8.onnx | 416 | 640x480 | 33.1 | 34.8 | 37.2 | 25.5 | 26.0 | 30.2 | 767 | 111 | 10.35 |

| Config | capture ms | preprocess ms | inference ms | postprocess ms | render ms | total ms |
|---|---|---|---|---|---|---|
| ORT default | 0.03 | 1.14 | 20.32 | 1.54 | 0.41 | 23.44 |
| intra_op=1 | 0.02 | 0.73 | 27.95 | 1.02 | 0.27 | 29.99 |
| intra_op=2 | 0.02 | 0.85 | 21.32 | 1.20 | 0.32 | 23.71 |
| intra_op=4 | 0.03 | 1.12 | 20.26 | 1.51 | 0.41 | 23.32 |
| intra_op=8 | 0.03 | 1.46 | 26.01 | 2.12 | 0.55 | 30.18 |
