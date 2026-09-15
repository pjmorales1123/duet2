# RGB bottle correction experiment v1

**Experimental, not promoted.** The frozen 48-trial test passed 2/12 nominal live-image trials and 3/12 pushed live-image trials. Initial-image-frozen control passed 1/12 nominal and 0/12 pushed trials. Seven seeds were refused before movement. The selected submission baseline remains [the six-skill dinner suite](../dinner_suite/README.md).

The custom 393,772-parameter CNN estimates two keypoints from three calibrated RGB cameras. At least two confident views, triangulation and a known 103 mm keypoint separation produce the bottle grasp point. A separate neural Cartesian motor map predicts five arm joints. Robot forward kinematics checks its error and refuses uncertain predictions; there is no inference-time IK repair. The controller uses current RGB during approach/descent, measured joint feedback, explicit phases and proprioceptive carry after closure. A separate privileged-state monitor only stops and scores execution.

`observer.safetensors` and `motor.safetensors` are the frozen selected weights, with FP32 OpenVINO exports in `openvino/`. The model hashes, calibration settings and status are recorded in `experiment.json`. Original Talos code, weights and generated supervision are MIT-licensed; existing robot-asset notices still apply. No external pretrained weights or private data were used.

The fresh synthetic perception test accepted 233/236 bottle-present states, rejected all 20 absent states, and measured 1.938 mm p95 / 5.182 mm maximum accepted keypoint error. That perception result does not establish physical grasp coverage. All four final conditions and the seven pre-motion refusals are retained in [the evidence](../../docs/robotics/evidence/rgb-servo-v1/README.md).

With the documented training/OpenVINO environment, reproduce the exposed final seed that passed both undisturbed controls:

```powershell
.\.venv-training\Scripts\python scripts/evaluate_rgb_servo_bottle.py --split evaluation --seed 2026095309 --observer models/bottle_servo_v1/observer.safetensors --motor models/bottle_servo_v1/motor.safetensors --minimum-views 2 --rigid-geometry --openvino models/bottle_servo_v1/openvino --push docs/robotics/experiments/rgb-servo-push-v1.json --output .run/reproduce-rgb-live
.\.venv-training\Scripts\python scripts/evaluate_rgb_servo_bottle.py --split evaluation --seed 2026095309 --observer models/bottle_servo_v1/observer.safetensors --motor models/bottle_servo_v1/motor.safetensors --minimum-views 2 --rigid-geometry --openvino models/bottle_servo_v1/openvino --push docs/robotics/experiments/rgb-servo-push-v1.json --mode frozen --output .run/reproduce-rgb-frozen
```

These are reproduction cases, not new holdouts. The scripts refuse an existing output folder and check the 10 GiB reserve. The experimental controller is available through this CLI; it is not substituted into the browser's dinner workflow. See [compact training reproduction](../../training/bottle_servo_v1/README.md).
