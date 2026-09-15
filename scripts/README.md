# Reproduction tools

Most users only need the root launcher. These retained scripts reproduce or inspect the demonstrated system:

| Task | Entry point |
| --- | --- |
| Evaluate dinner skills or a bottle relay | `evaluate_learned_dinner.py` |
| Verify production live-mug execution and device evidence | `verify_visual_mug_submission.py` |
| Verify Intel baseline/device eligibility | `verify_intel_submission.py` |
| Check local Speechmatics configuration with a supplied audio file | `verify_speechmatics.py` |
| Build a credential-free hosted bundle | `build_hf_space.py` |
| Train/export a trajectory model | `train_primitive_policy.py`, `export_primitive_openvino.py` |
| Reproduce the selected stereo observer | `reproduce_mug_keypoints.py` and the frozen training package |
| Reproduce the corrective motor network | `package_mug_correction_v2.py` and the frozen training package |

Supporting modules are retained where those entry points import them. The [setup guide](../docs/SETUP.md) gives runnable commands. Some full historical package verifiers require the [development archive](../docs/ARCHIVE.md), which preserves all members of their immutable manifests. One-off slide exporters, stopped manipulation probes and abandoned development drivers are in that archive.

Always choose unused output and console filenames, retain failures and check the storage reserve before writes. No script should upload secrets or silently replace a published model.
