# Experimental stereo mug observer

This separately trained perception model passes its declared image and saved-state gates. It is **not integrated into the selected dinner controller** and does not yet demonstrate corrective movement.

Two calibrated 320×240 oblique RGB views enter a new 393,772-parameter CNN. It predicts two mug-axis image points, a foreground mask and visibility. Fixed camera triangulation returns the known mug's axis midpoint. Inference uses RGB and calibration; simulator state and segmentation are training/scoring labels only. This is a bounded upright held-mug observer, not general object perception.

- Selected weights: `step-008000/model.safetensors`, SHA-256 `e6db2942f2510cbb84bbe250e7e4fbb51aab4db95cbc601a70ffaa84845a898b`.
- Earlier checkpoint: `step-004000/model.safetensors`; preserved with all its development scores.
- Evaluated deployment: `openvino/observer.xml` and `.bin`, FP32 CPU, two images per call.
- Actual host: AMD Ryzen 7 5800X; development median neural inference 29.55 ms. This is not Intel execution evidence or full robot throughput.

Fresh synthetic perception accepts 448/448 present mugs and 0/64 absent mugs: p95 position error 0.270 mm, maximum 0.422 mm. All 144 already-exposed physical states are accepted: p95 0.617 mm, maximum 0.893 mm. These are different test sets, and the latter is a saved-state regression with no new physics.

The [protocol](protocol.json), [complete evidence](../../docs/robotics/evidence/mug-keypoint-observer-v1/README.md) and [exact compact inputs](../../training/mug_keypoint_observer_v1/README.md) explain the fixed confidence/geometry gates and limits. Original Talos weights/source use MIT; no pretrained weights were incorporated.
