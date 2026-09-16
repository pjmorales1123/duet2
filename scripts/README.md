# Duet 2 pipeline scripts

The scripts most relevant to the current SmolVLA pipeline:

| Task | Entry point |
| --- | --- |
| Collect verified physical demonstrations for a seed | `collect_dinner_learning.py` |
| Convert collected demonstrations into a LeRobot v3 dataset | `prepare_lerobot_dataset.py` |
| Render the expert-teacher video gallery for the demo dashboard | `build_expert_gallery.py` |
| Capture the target-arrangement reference image | `capture_target_arrangement.py` |
| Design/inspect a dinner table layout | `design_dinner_layout.py` |

Other scripts in this folder support earlier, separate experiments from the
project's baseline and aren't part of the current SmolVLA pipeline.
