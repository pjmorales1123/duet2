# Experimental bottle RGB refinement

This model **failed its fresh perception gate and is not deployed**. It uses current RGB, a frozen coarse CNN, a new semantic-point crop CNN and fixed stereo geometry. No simulator object pose, depth, segmentation or teacher enters inference. Physical generalization is unverified.

See [all results and limitations](../../docs/robotics/evidence/bottle-refinement-v1/README.md) and [exact training input](../../training/bottle_refinement_v1/README.md). Both candidates and the selected CPU export are preserved. License: MIT.
