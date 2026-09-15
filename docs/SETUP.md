# Setup and reproduction

Start with the root README. Python 3.12 and `requirements.txt` provide the complete local CPU demo, including OpenVINO and Torch. The model parameters and scene geometry are already included. Keep 10 GiB free **in addition to expected writes** before each dataset, capture or export.

## Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\start-lab.ps1 -Learned -Port 8765
```

The launcher also recognizes an existing `.venv-training` environment from development. `-NoBrowser` runs without opening a tab. Stop the selected server with `.\stop-lab.ps1 -Port 8765`.

The recommended recorded workflow uses Dinner challenge, Task start, Closed drawer, seed 42 and **Learned dinner · live mug vision**. Reset, then enter `Set the table`. Other instructions include `Place the bottle` and the bottle relay commands listed in the app. Unsupported combinations are refused. Each hosted trial resets automatically; reset locally before a new full workflow.

## Optional voice

Copy `.env.example` to `.env` and privately set `SPEECHMATICS_API_KEY`. The server reads that local file; it is ignored by Git. Use `Speak instruction` in the local browser and grant microphone access yourself. No voice service or key is included in the public Space. Typed commands require no key.

## Linux and hosted interface

Install Python 3.12 and the graphics libraries listed in `hosting/packages.txt`. Use `python3.12 -m venv .venv`, install the root requirements, then run `python -m simulation_lab.server --learned --port 8765` on a machine with a working OpenGL renderer. For headless hosting, the Gradio app configures MuJoCo with OSMesa.

The separate `hosting/requirements.txt` adds Gradio and records the deployed Space environment. For a local Linux CPU run of this interface, install the graphics libraries above and run these commands from the full GitHub checkout:

```bash
python3.12 -m venv .venv-hosting
source .venv-hosting/bin/activate
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r hosting/requirements.txt
python hosting/gradio_app.py
```

Installing CPU Torch first avoids downloading a CUDA build for this local CPU interface. The `torch==2.8.0` requirement accepts the installed `2.8.0+cpu` build. The public Space's platform manages its optional shared-GPU support separately.

Build an explicit allowlisted Space bundle with:

```powershell
.\.venv\Scripts\python scripts/build_hf_space.py --output .run/space-export
```

The builder refuses existing outputs and does not upload. It excludes credentials, recordings, development history and training datasets, and includes the third-party notices with the license files. Within the exported bundle, `hosting/requirements.txt` is copied to the root as `requirements.txt`. The Space's OpenVINO CPU path performs local inference; its optional shared-GPU path depends on Hugging Face allocation and can report allocation failures separately.

## Verification

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python scripts/evaluate_learned_dinner.py --suite models/dinner_suite/suite.json --skills bottle,plate,mug,drawer,fork,spoon --seed 2026091901 --output .run/dinner-reproduction.json
.\.venv\Scripts\python scripts/verify_visual_mug_submission.py --diagnostic --output .run/live-mug-reproduction
```

The last check runs two exposed production-entry workflows. `--diagnostic` reports physical execution on the current machine while retaining hardware eligibility flags. It does not turn a nonqualifying CPU into an Intel result. Every output path must be new. These commands reproduce exposed cases; they are not fresh generalization tests. Some historical offline package reproducers require the complete development archive, as their immutable manifests include archived files.

The complete training-contract suite also exercises the historical ACT implementation. After installing the optional training environment below, run `.\.venv-training\Scripts\python -m unittest discover -s training_tests`. LeRobot is needed for those training tests; it is not needed to run the selected CPU demo.

## Optional RTX training

The selected trajectories, stereo observer and corrective motor network were trained from Talos data. They do not require a pretrained robotics foundation model. Compact inputs are in `training/`; each package documents its protocol and frozen source. Historical full-package reproduction uses the [archive](ARCHIVE.md).

Create a separate Python 3.12 environment, install the root CPU requirements, replace Torch with the CUDA 12.8 build, then install `training/requirements.txt`:

```powershell
py -3.12 -m venv .venv-training
.\.venv-training\Scripts\python -m pip install -r requirements.txt
.\.venv-training\Scripts\python -m pip install --upgrade torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv-training\Scripts\python -m pip install -r training/requirements.txt
.\.venv-training\Scripts\python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use new output folders and preserve the published checkpoints. An RTX 4070 improves iteration time but does not establish broader physical coverage. The archived CLIPort adaptation ran with finite gradients; it was not trained into a successful Talos policy.
