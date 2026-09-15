# Exact learned dinner training inputs, v6

Each skill contains a compact `retrieval.npz`, `retrieval.json` provenance and
`input-replay.json` physical gate. The filename is historical: runtime executes
neural predictions, not action retrieval. Plate, mug and drawer have 41 eligible
examples each; fork and spoon have 39 each (201 total). Two rejected fork
attempts and the two uncollected following spoon episodes remain disclosed in
the original collection reports. No failed attempt entered fitting.

Only training scenes supplied images and labels. The full 32-dimensional RGB
feature input, stage-aligned 20 Hz action endpoints and scaling information
are sufficient to retrain without private raw images. Carry labels hold the
gripper closed; all 201 exact aligned-input trajectories passed physical replay
before fitting. Runtime has no privileged stage labels or object-position input.

Train each of plate, mug, drawer, fork and spoon to one fixed Adam endpoint:

```powershell
./.venv-training/Scripts/python.exe scripts/train_primitive_policy.py --source training/dinner_suite/mug --output .run/dinner-mug-retrain --steps 40000 --save-every 40000 --lr 0.0003 --decay --visual-scaling standard
```

Use the recorded `arguments` in each packaged `primitive.json` as the exact
training configuration. V6 reuses those v5 weights without further fitting and
sets `inactive_arm_control` to `hold_previous_targets` in a separate candidate
metadata copy before exporting with `scripts/export_primitive_openvino.py`.
This explicit motor-target continuity guard is separate from neural inference.
The original bottle input remains under `training/bottle_visual`.

The six development seeds and all ten frozen evaluation scenes are now exposed.
Their outcomes are in `docs/robotics/evidence/learned-dinner-v6`; do not reuse them
as unseen holdouts. RTX training and AMD CPU export measurements are not Intel
deployment evidence. All outputs require a new path and the 10 GiB disk reserve.
