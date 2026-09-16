# Duet 2 pipeline scripts

The scripts most relevant to the current SmolVLA pipeline:

| Task | Entry point |
| --- | --- |
| Collect verified physical demonstrations for a seed | `collect_dinner_learning.py` |
| Convert collected demonstrations into a LeRobot v3 dataset | `prepare_lerobot_dataset.py` |
| Render the expert-teacher video gallery for the demo dashboard | `build_expert_gallery.py` |
| Capture the target-arrangement reference image | `capture_target_arrangement.py` |
| Design/inspect a dinner table layout | `design_dinner_layout.py` |
| Export the SmolVLA vision tower to OpenVINO IR | `export_smolvla_vision.py` |

Other scripts in this folder support earlier, separate experiments from the
project's baseline and aren't part of the current SmolVLA pipeline.

`benchmark_live_vla.py` loads the live SmolVLA checkpoint
(`simulation_lab/vla_task.py`) against a real dinner scene, headless, and
reports checkpoint load time, camera-render and inference latency, and
whether it can keep up with the physics loop in real time.

`export_smolvla_vision.py` writes the OpenVINO IR the live VLA runs its vision
tower on, into `models/<checkpoint>/openvino/`, with a SHA-pinned `parity.json`
and a hard parity gate. Run it once per checkpoint; without it the policy falls
back to PyTorch.

`quantize_smolvla_vision.py` is the INT8/NNCF variant of that export. It is
**not** part of the shipped path: measured 41% *slower* on the demo laptop,
which is AVX2-only and so has no VNNI. Kept because it is expected to win on
any Ice Lake or newer Intel CPU.

See [`OPTIMIZATIONS.md`](../OPTIMIZATIONS.md) for the measured results and
[`simulation_lab/PERF_NOTES.md`](../simulation_lab/PERF_NOTES.md) for the
engineering trail, including superseded conclusions.
