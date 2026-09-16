# Duet 2

Two SO-101 arms coordinate to set a dinner table in MuJoCo — a scripted physical
teacher policy, and a SmolVLA vision-language-action model fine-tuned on its
demonstrations. Built for the lablab.ai AI Infra Summit Hackathon, Intel
Bimanual VLA track.

## What's here

- **Expert teacher policy** (`simulation_lab/dinner_autonomy.py`, `autonomy.py`):
  a contact-only, exact-state physical controller that picks and places a
  bottle, plate, mug, fork and spoon from a randomized cabinet zone.
- **Live SmolVLA policy** (`simulation_lab/vla_task.py`): the actual fine-tuned
  checkpoint, run end-to-end from raw camera pixels and free-text
  instructions — no scripted motion, no privileged state.
- **Demo dashboard** (`simulation_lab/web/demo.html`): separates the expert
  teacher's capability showcase from the live VLA, with a target-arrangement
  reference image, a 10-seed video gallery, live "try it now" buttons against
  the real simulator, and a generalization check against unseen seeds.
- **Data + model pipeline** (`scripts/collect_dinner_learning.py`,
  `scripts/prepare_lerobot_dataset.py`, `scripts/build_expert_gallery.py`):
  collect verified physical demonstrations, convert them to a LeRobot v3
  dataset, and render the gallery videos.

## Running it

```powershell
pip install -r requirements.txt
python -m simulation_lab.server
```

Then open `http://127.0.0.1:8765/` for the full engineering console, or
`http://127.0.0.1:8765/demo` for the judge-facing dashboard.

## Data and model checkpoints

Training data and the fine-tuned checkpoint are hosted on Hugging Face, not
committed to this repo:

- Dataset: [`pjmorales04/duet-micro-v1`](https://huggingface.co/datasets/pjmorales04/duet-micro-v1)
- Model (current, 7000 steps): [`pjmorales04/duet-smolvla-v2`](https://huggingface.co/pjmorales04/duet-smolvla-v2)
- Model (earlier): [`pjmorales04/duet-smolvla-v1`](https://huggingface.co/pjmorales04/duet-smolvla-v1)

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('pjmorales04/duet-smolvla-v2', local_dir='models/duet-smolvla-v2')"
```

`simulation_lab/vla_task.py` prefers `models/duet-smolvla-v2` and falls back to
`v1`; set `DUET_VLA_CHECKPOINT` to point at any other checkpoint directory.

### OpenVINO inference

The live VLA runs its vision tower through **OpenVINO CPU FP32**. Build the IR
once per checkpoint (it lands in `models/<name>/openvino/`, which is gitignored):

```powershell
python scripts/export_smolvla_vision.py
```

The export writes a `parity.json` recording SHA-256 of the IR, checkpoint and
exporter, and fails if numerical parity against the PyTorch reference regresses.
If the IR is absent the policy falls back to PyTorch automatically — slower,
never silently different. Environment overrides: `DUET_VLA_OPENVINO=0` forces
PyTorch, `DUET_VLA_OV_DEVICE` selects the plugin, `DUET_VLA_BF16=1` restores the
checkpoint's original dtype for A/B comparison.

To check that a checkpoint loads and to measure whether it can keep up with the
physics loop on your hardware:

```powershell
python scripts/benchmark_live_vla.py --sim-seconds 6
```

On the CPU-only demo laptop (i5-8265U, no discrete GPU) this work took the live
policy from a real-time factor of **0.05 to 0.49** — a **10.3× end-to-end
speedup** with no loss of accuracy. What was measured, what was changed, and
what was deliberately rejected is written up in
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md).

## Attribution

This project started from an MIT-licensed baseline; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE) for
full upstream attribution and licenses of bundled assets.
