# Experimental spoon release model — not promoted

One warm-start spoon fit completes 12,000 updates after all 39 modified training action replays pass. Its OpenVINO export passes numerical parity, but its paired physical development test does **not** improve complete workflows: candidate and baseline each pass 4/6 standard and 4/6 left-reach starts. Both pass every reached spoon attempt; two scenes fail earlier at the unchanged mug skill. The candidate misses the 5/6-per-preset gate and stops before final testing.

- [All outcomes, scope and reproduction](../../docs/robotics/evidence/spoon-release-v1/README.md).
- [Exact training inputs](../../training/spoon_release_v1/README.md), [training report](training.json) and [export parity/timing](openvino/benchmark.json).
- `checkpoint/` preserves the sole trained candidate; `openvino/` preserves the actual evaluated FP32 export. `suite.json` references unchanged original policies for every other skill.

Training uses the RTX 4070 for 43.674 seconds. Export measurements identify the actual AMD CPU and NVIDIA GPU; they are not Intel benchmarks. Original Talos weights and code are MIT; upstream robot notices remain separate. The selected demo continues using `models/dinner_suite/suite.json`.

To reproduce the fit on the documented CUDA environment, use a new output directory:

```powershell
.venv-training/Scripts/python scripts/train_spoon_release.py --output .run/spoon-release-fit-reproduction
```

Keep the original models, all outcomes, unused output/log paths and the 10 GiB disk reserve. A reproduced fit does not authorize a new checkpoint selection on the exposed development scenes.
