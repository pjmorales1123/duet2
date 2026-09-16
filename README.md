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
- Model: [`pjmorales04/duet-smolvla-v1`](https://huggingface.co/pjmorales04/duet-smolvla-v1)

Download a checkpoint into `models/<name>/` and `simulation_lab/vla_task.py`'s
`CHECKPOINT` constant will pick it up.

## Attribution

This project started from an MIT-licensed baseline; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE) for
full upstream attribution and licenses of bundled assets.
