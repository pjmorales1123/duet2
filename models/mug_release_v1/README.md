# Experimental mug release candidate — not promoted

This weighted warm fit passes its 41/41 revised demonstration replay gate and FP32 OpenVINO parity, but **regresses in complete physical workflows**. The baseline passes 6/6 standard and 6/6 left-reach development starts; the candidate passes 4/6 in each preset. All 24 outcomes are preserved, and the reserved final scenes remain unexposed.

- [Full comparison and reproduction](../../docs/robotics/evidence/mug-release-v1/README.md).
- [Exact training inputs](../../training/mug_release_v1/README.md), [training report](training.json) and [export parity/timing](openvino/benchmark.json).
- `checkpoint/` preserves the sole 18,000-update candidate; `openvino/` preserves its actual evaluated export. `suite.json` references unchanged original models for every other skill.

Training takes 70.569 seconds on the RTX 4070. Reported inference devices are the actual AMD CPU and NVIDIA GPU, not Intel hardware. The selected demo continues using `models/dinner_suite/suite.json`.

```powershell
.venv-training/Scripts/python scripts/train_release_primitive.py --protocol docs/robotics/experiments/mug-release-v1.json --output .run/mug-release-fit-reproduction
```

Use a fresh output directory, retain the 10 GiB reserve and preserve every earlier artifact. A reproduced fit does not authorize reselection on exposed scenes. Original Talos code and weights use MIT; upstream robot notices remain separate.
