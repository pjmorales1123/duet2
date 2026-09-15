# Exact stereo mug perception inputs

The package retains all 5,120 posed states: 4,096 training, 512 development and 512 fresh evaluation. Each split contains the full joint/object pose recipe, episode index, exposure, presence, projected image points, world labels, visibility/counts and RGB hashes. All absent examples and every outcome are retained. The raw RGB/segmentation shards stay local; the compact recipes regenerate them from the existing repository scenes/assets.

Training reuses only the failed direct-XYZ observer's original training poses and verifies all 8,192 reused RGB hashes. New development and evaluation RNG seeds are 2026099852 and 2026099853. These are synthetic posed renders with joint/object perturbations, not physically valid grasp demonstrations. The older protocol's reserved 2026099803 split remains unexposed.

One RTX 4070 FP32 fit completes 8,000 AdamW updates, batch 12, in 173.022 seconds, reserving at most 2.914 GiB of Torch GPU memory. Both checkpoints (4,000 and 8,000) complete; development selects step 8,000 by the frozen accuracy rule. There is no pretrained warm start. Fixed calibration and object dimensions remain explicit geometric assumptions.

Install the root README's Python 3.12 training/OpenVINO environment. To verify the selected packaged model and all 512 fresh recipes without reading raw `.run` data, choose a new output path:

```powershell
.\.venv-training\Scripts\python scripts/reproduce_mug_keypoints.py --output .run/reproduce-mug-keypoints-v1.json
```

The recorded reproduction verifies 117 core package files, exactly regenerates all 1,024 RGB views and reproduces every keypoint and acceptance result exactly on this PC. It is an exposed reproduction, not an additional independent evaluation. Other rendering devices may differ; an RGB hash mismatch is reported rather than silently accepted.

For a full retraining reproduction, restore the frozen sources in `source/`, use an isolated checkout with no existing `.run/mug-keypoint-observer-v1`, then run `collect_mug_keypoints.py --split training`, `--split development`, and `train_mug_keypoints.py`. Audit both splits before `export_mug_keypoints.py`. Generate/audit/score `--split evaluation` only after passing development/export, then `evaluate_mug_keypoints.py --mode physical`. Follow the unchanged [protocol](protocol.json); do not refit against exposed evaluation results. Original JSON paths in the evidence identify original-run artifacts; `reproduction-inputs.json` maps them to their packaged copies.

Check disk before each split/export/package and retain the 10 GiB reserve plus expected writes. The declared combined raw/package cap is 2 GiB. The source package includes the corrected audit and separately retains the stopped audit attempt in the evidence. Never overwrite earlier results, models or raw shards.
