# Upright visual bottle policy

The selected checkpoint completed **10/10 new randomized dinner scenes** (seeds 2026091301–2026091310), with **0.36–1.66 mm** final placement errors and **28.4–29.3 simulated seconds** per task. The checkpoint and runtime were frozen before this evaluation, and every trial is retained. These seeds are now exposed. This covers the task preset's small bottle-position perturbations, not arbitrary positions across the reachable workspace.

## Architecture and observations

A classical color-based detector locates the amber bottle in the initial **overhead RGB image**. It uses a fixed calibrated region and rejects missing or ambiguous detections. The normalized image centroid, padded to 32 features, conditions a neural motion primitive with three 256-unit SiLU layers and internal progression features. The network predicts 12 nominal joint targets. A motor-feedback guard delays progression if the left arm has not tracked the previous target.

The neural network runs through OpenVINO FP32 on the Intel CPU. Image processing and feedback regulation remain ordinary Python/NumPy. Wrist views remain available in the application but are not used by this encoder. This is an initial-image-conditioned learned trajectory, not a learned detector, general VLA, or continuously corrected visual controller. No demonstration action lookup, simulator object position, simulator clock or teacher task-stage signal selects its actions.

An independent simulator-state monitor checks physical support, lift, release, placement and disturbances. It can stop the policy but cannot generate corrective actions. The gripper uses a 0.25 Nm torque cap, with actual MuJoCo finger contacts and no object attachment.

## Training and limits

Training used 22 eligible upright physical demonstrations from a broader compact batch. Offline teacher-stage labels align demonstration timing; those labels are absent at inference. Adam ran 16,000 steps, followed by LBFGS refinement. The selected refinement checkpoint is step 400. A further step-800 checkpoint was not substituted into the frozen evaluation.

Development checks passed the default scene and six new full-scene seeds. Of three additional wider-position training cases sampled for development, one passed and two failed. Failure handling is part of the controller, but larger pose coverage remains unresolved. Sideways bottles, different camera placement, changed bottle appearance, occlusion at startup and other object categories are outside this model's scope. The original three-view checkpoint remains available separately for familiar sideways practice.

The package uses safetensors and OpenVINO XML/BIN, not pickle. Weights and inference settings match the evaluated export; only the architecture description in `primitive.json` was corrected to describe the RGB centroid encoder. Code and weights are MIT licensed. See `submission/evidence/` and `docs/robotics/VISUAL_BOTTLE_POLICY.md` for the protocol and reproduction commands.
