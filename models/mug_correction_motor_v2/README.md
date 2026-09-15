# Experimental joint-limited mug motor map

This 19,215-parameter neural map passes bounded offline geometry and CPU checks. **It is not integrated into the selected dinner controller.** Contact physics and live-camera correction require a separate experiment.

Inputs are five measured right-arm joints and an XY correction of at most 2 mm. A neural differential matrix predicts joint movement. Explicit postprocessing scales that movement and the requested translation together to keep every joint step at or below 0.02 rad. Robot-only forward geometry may refuse, never repair, a prediction. No object state, teacher, Jacobian or inverse solver enters deployment.

Selected weights: `step-006000/model.safetensors`, SHA-256 `e9b0cedd850c000620ddebe0061a21ed0eabb49858a651562dfad2d5a1dcc36a`. Step 3,000 and all its scores are also retained. `openvino/motor.xml` / `.bin` are the evaluated FP32 export. `runtime.json` records hashes, support and step/guard settings; `simulation_lab.mug_correction_runtime.LocalMugMotor` implements the checked callable interface.

All 1,024 fresh configurations pass: p95 endpoint error 0.061 mm, maximum 0.131 mm. Errors are against the **scaled effective translation**, not a guaranteed 2 mm movement. Minimum fresh step fraction is 0.0407; median is 1.0. Difficult poses may need more correction steps. Physical convergence remains untested.

Median neural inference is 0.100 ms on AMD Ryzen 7 5800X, excluding cameras, guards and physics. This is not Intel evidence. See [complete evidence](../../docs/robotics/evidence/mug-correction-motor-v2/README.md) and [numeric inputs/reproduction](../../training/mug_correction_motor_v2/README.md). Original Talos source/weights use MIT; no pretrained weights were incorporated.
