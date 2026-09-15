# Compact relay training inputs

Each of the four folders contains `retrieval.npz` (the exact compact neural
training input), `retrieval.json` (training split, encoding and label lineage)
and `input-replay.json` (the complete physical replay gate tied to the input
SHA-256). The historical filename does not mean action retrieval at inference.

There are 13 examples per leg. All 52 aligned trajectories passed physical replay
before model fitting. Raw RGB episodes and private experiment checkpoints are
excluded. Published inputs are sufficient to repeat fitting; the collector and
preparation scripts regenerate physical demonstrations when a new dataset is
needed. Check free disk space before each new dataset/export and keep 10 GiB free
beyond estimated writes.

Example for one leg, in the documented CUDA training environment:

```powershell
./.venv-training/Scripts/python.exe scripts/train_primitive_policy.py --source training/bottle_relays/relay_bottle_left --output .run/relay-left-retrain-adam --steps 16000 --save-every 16000 --lr 0.0003 --decay --visual-scaling standard
./.venv-training/Scripts/python.exe scripts/train_primitive_policy.py --source training/bottle_relays/relay_bottle_left --output .run/relay-left-retrain-lbfgs --steps 400 --save-every 400 --lr 0.8 --optimizer lbfgs --visual-scaling standard --warm-start .run/relay-left-retrain-adam/step-016000
```

Repeat with each leg's own inputs; a checkpoint is specific to its arm, start and
fixed destination. Floating-point training results can vary by device/library;
physical evaluation is required before promoting a retrained result. The tested
release remains preserved in `models/bottle_relays`.

Protocols: `docs/robotics/experiments/relay-learning-v1.json` and
`relay-grip-repair-v3.json`. Frozen evaluation seeds and complete outcomes live in
`docs/robotics/evidence/learned-relays-v3`. Those evaluation seeds must not be
treated as unseen data in subsequent model selection.
