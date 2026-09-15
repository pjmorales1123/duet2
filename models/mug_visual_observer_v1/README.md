# Experimental mug RGB pose checkpoints — not selected

Both checkpoints from the [stopped v1 protocol](../../docs/robotics/evidence/mug-visual-observer-v1/README.md) are preserved. Neither passes development accuracy. They are not used by the browser or hosted dinner demo and have not been exported or tested as motor-control inputs.

| File | SHA-256 |
| --- | --- |
| `step-006000.safetensors` | `0e00ada4eb00574a68094d58e7498d551626bb8aac350f1b05d50889f38e746b` |
| `step-012000.safetensors` | `b26fca746bc54b65fb91f4660ee03d7fabb57a680c80b3acd85814800d3a0923` |

Architecture: `MugShapePoseNet` in `simulation_lab/mug_visual_observer.py`, 792→256→256→128→3 with SiLU, trained from scratch. Input is classical RGB silhouette geometry from three fixed cameras. Convert network output to the body midpoint in metres with `output * 0.05 + [0.255, -0.025, 0.8]`. At least two image components need 60 pixels; this availability check does not establish accurate pose inference.

All weights, source and original contributions remain under the repository MIT license; existing third-party robot asset notices remain applicable. The selected dinner suite remains `models/dinner_suite/suite.json`.
