# All configurations retained with limited joint steps

[V2](protocol.json) reuses every 8,192 training and 1,024 development configuration, direction and matrix from the stopped fixed-step V1, including every rejection. Joint deltas and requested translations are scaled together to a 0.02 rad joint bound. No sampled position is removed; every prior rejected point now has an accepted local label.

The fresh set contains 1,024 new configurations (seed 2026100203). V1 seed 2026100103 remains unexposed. Development reuses exposed configurations and is labeled accordingly. Each `data.npz` retains original requests, effective translations, step fractions, matrices, labels and all joint/recipe fields. Manifests preserve every score and source hash.

One RTX 4070 FP32 fit completes 6,000 AdamW updates in 18.160 seconds, with 0.0234 GiB peak Torch-reserved memory. Both checkpoints pass development; the frozen rule selects step 6,000. Independent geometry audits check every label and 64 finite-difference Jacobians per split.

Install the root README's Python 3.12 training/OpenVINO environment. This reproduction checks all 101 core manifest files and exactly reproduces all 1,024 fresh accepted joint outputs without raw `.run` data:

```powershell
.\.venv-training\Scripts\python scripts/package_mug_correction_v2.py --verify-only
```

It repeats exposed data, not a new independent experiment. For full reproduction in a new isolated checkout, restore `source/` and use unused outputs. Run `mug_correction_motor_v2.py --mode prepare --split training` and `--split development`; audit both with `audit_mug_correction_motor.py --protocol` pointing to the absolute V2 protocol path. Then run `--mode train` and `--mode export`. After passing those gates, use **`collect_mug_correction_fresh.py`**, audit the evaluation split, then run `mug_correction_motor_v2.py --mode evaluate`.

The fitted V2 script retains a fresh-dispatch naming bug, so its `--mode prepare --split evaluation` is not the reproduction entrypoint. The separate adapter calls the unchanged sampler and step-sizing functions. Both launch failures preceded affected data creation; source snapshots, redacted console copies and original hashes remain in the [evidence](../../docs/robotics/evidence/mug-correction-motor-v2/README.md).

Check disk before every dataset/export/package and preserve the 10 GiB reserve plus expected writes. The combined raw/package cap is 256 MiB. This package verifies offline motor preparation, without physical integration or promotion.
