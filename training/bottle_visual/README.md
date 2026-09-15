# Frozen upright bottle training input

This 271 KB package contains the exact 22-episode compact input used for the selected upright bottle network: initial RGB centroid features, offline stage-aligned 20 Hz joint targets, episode boundaries and timing. Original procedural simulation data and derived arrays are MIT licensed. There are no personal recordings or credentials.

The filenames retain an earlier loader convention: `retrieval.npz` and `retrieval.json`. They are **training inputs only**. The deployed model loads safetensors/OpenVINO weights and `visual.npz`; it never loads these action arrays. The RGB detector is classical image processing and is not trained from this package.

With a CUDA GPU and the training environment, reproduce the selected optimization schedule using fresh output directories:

```powershell
.\.venv-training\Scripts\python scripts/train_primitive_policy.py --source training/bottle_visual --visual-scaling global --output .run/reproduce-adam --steps 16000 --save-every 16000 --lr 0.0003 --decay
.\.venv-training\Scripts\python scripts/train_primitive_policy.py --source training/bottle_visual --visual-scaling global --output .run/reproduce-refine --steps 400 --save-every 400 --lr 1 --optimizer lbfgs --warm-start .run/reproduce-adam/step-016000
.\.venv-training\Scripts\python scripts/export_primitive_openvino.py --checkpoint .run/reproduce-refine/step-000400 --output .run/reproduce-openvino
```

Floating-point kernels on another GPU/runtime can change optimization outcomes. Evaluate a retrained checkpoint physically before substituting it. The packaged model is the evaluated result, not an assurance that retraining will reproduce identical success rates. Every script checks disk space with a 10 GiB reserve before output.
