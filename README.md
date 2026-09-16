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

### Local setup (Windows / CPU)

```powershell
git clone https://github.com/pjmorales1123/duet2.git
cd duet2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m simulation_lab.server
```

Then open `http://127.0.0.1:8765/` for the full engineering console, or
`http://127.0.0.1:8765/demo` for the judge-facing dashboard.

The server starts the MuJoCo physics loop and camera process locally. The
Expert tab is the verified scripted teacher: it uses exact simulator state and
physical contact/placement checks. The VLA tab is different: it is the real
fine-tuned SmolVLA checkpoint receiving camera pixels, language and robot state
and producing action targets. It is not a scripted fallback, but it is still an
unfinished research result and should not be described as reliably completing
the table.

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

The live VLA runs its vision tower through **OpenVINO CPU FP32** when its
parity-checked IR is present. Build the IR once per checkpoint (it lands in
`models/<name>/openvino/`, which is gitignored):

```powershell
python scripts/export_smolvla_vision.py
```

Run the export after downloading the checkpoint and before starting the server.
Do not commit the generated `.xml`/`.bin` files to GitHub; they are large build
artifacts. If a valid export already exists, the exporter refuses to overwrite
it unless `--force` is passed. The export is tied to the checkpoint and records
the IR, source and runtime hashes in `parity.json`.

For the normal optimized CPU path, use these settings (the defaults already do
this, but making them explicit is useful for a judge machine):

```powershell
$env:DUET_VLA_OPENVINO = "1"
$env:DUET_VLA_OV_DEVICE = "CPU"
Remove-Item Env:DUET_VLA_BF16 -ErrorAction SilentlyContinue
python -m simulation_lab.server
```

At startup, the log should report `OpenVINO CPU FP32 vision tower + PyTorch
CPU FP32 expert`. The VLA task snapshot also exposes the active runtime in
`policy_details.neural_runtime`. If the IR is missing, invalid, or fails its
parity gate, the code falls back to PyTorch and reports the slower runtime; it
never silently uses an unverified graph. Set `DUET_VLA_OPENVINO=0` only for an
A/B comparison. Do not set `DUET_VLA_BF16=1` on CPUs without native bf16
instructions: that restores the checkpoint's original format and is expected
to be much slower.

To check that a checkpoint loads and to measure whether it can keep up with the
physics loop on your hardware:

```powershell
python scripts/benchmark_live_vla.py --sim-seconds 6
```

The committed benchmark baseline, captured on an i5-8265U, took the live policy
from a real-time factor of **0.05 to 0.49** — a **10.3× end-to-end speedup**
with no loss of accuracy. The deployment target is an **Intel Core i5-12400**;
rebuild the IR and rerun the benchmark there before publishing target-hardware
latency claims. What was measured, what was changed, and what was deliberately
rejected is written up in
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md).

### What was optimized for Intel

The optimization is deliberately split between model math and camera work:

- The checkpoint's bf16 weights are cast to FP32 because the reference CPU
  cannot execute bf16 natively; this removed the emulation penalty without
  changing the weights.
- The SigLIP vision tower is exported to OpenVINO CPU FP32 with
  `PERFORMANCE_HINT=LATENCY`, explicit FP32 precision and a parity gate.
- Flow-matching denoising was reduced from 10 to 5 steps, PyTorch threads are
  bounded so physics keeps CPU capacity, and the checkpoint is preloaded.
- SmolVLA predicts 50-action chunks. Camera frames are rendered only when a
  fresh model query consumes them; cached action pops do not redo vision work.
- The dashboard's Fast Preview changes only camera rendering (resolution,
  shadows and anti-aliasing), never collision geometry or task physics.

On the i5-8265U baseline, the measured end-to-end real-time factor improved
from **0.05 to 0.49** (10.3×). Those figures are historical baseline numbers;
the stated deployment target is the i5-12400, so rerun the benchmark there.
The iGPU and INT8 experiments were rejected when they failed the FP32 accuracy
or latency contract. See [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) for the full
measurement trail and rejected alternatives.

### Making OpenVINO available to judges

GitHub contains the exporter and runtime, but the generated IR is gitignored.
For a hosted judge demo, download the checkpoint and either build the IR during
image setup or upload the three `openvino/` artifacts (`vision_tower.xml`,
`vision_tower.bin`, `parity.json`) to the Hugging Face checkpoint repository.
The deployment must run the long-lived FastAPI/MuJoCo server with
`DUET_VLA_OPENVINO=1` and `DUET_VLA_OV_DEVICE=CPU`; a serverless function is not
appropriate for the persistent simulator and camera context. Always run the
benchmark on the deployment CPU before publishing its latency.

The parity-checked OpenVINO files are published in the model repository at
[`openvino/`](https://huggingface.co/pjmorales04/duet-smolvla-v2/tree/main/openvino)
([upload commit](https://huggingface.co/pjmorales04/duet-smolvla-v2/commit/e308bceb6f6f66ebfd540fb15bea85cec068ceae)). A fresh
`snapshot_download` therefore includes the optimized runtime automatically.

### Submission checklist

1. Download `duet-smolvla-v2` with `snapshot_download`.
2. Confirm `models/duet-smolvla-v2/openvino/parity.json` reports
   `parity_passed: true`.
3. Start the server with the OpenVINO CPU settings above and verify the startup
   runtime line.
4. Open `/demo`: use the Expert tab for verified table-setting and pour
   completion; use the VLA tab to demonstrate the real, multimodal checkpoint
   while describing it honestly as experimental.
5. For a hosted demo, expose the long-lived server from a persistent container
   and run the six-second benchmark on that deployment CPU before claiming its
   speed.

## Attribution

This project started from an MIT-licensed baseline; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE) for
full upstream attribution and licenses of bundled assets.
