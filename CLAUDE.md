# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Duet 2: two SO-101 arms coordinate to set a dinner table in MuJoCo. There are two
distinct controllers for the same task, both reachable from one FastAPI server:

- A **scripted expert teacher** (`simulation_lab/dinner_autonomy.py` + `autonomy.py`):
  a contact-only, exact-state IK controller. It reads privileged simulator state
  (ground-truth object poses) and is used to generate verified demonstrations.
- A **live SmolVLA policy** (`simulation_lab/vla_task.py`): the actual fine-tuned
  checkpoint, run end-to-end from raw camera pixels and free-text instructions only
  — no privileged state, no scripted motion.

Built for the lablab.ai AI Infra Summit Hackathon (Intel Bimanual VLA track). Data
and the fine-tuned checkpoint live on Hugging Face (`pjmorales04/duet-micro-v1`,
`pjmorales04/duet-smolvla-v1`), not in this repo — see the root README for how to
fetch a checkpoint into `models/<name>/`.

**This is a public cut of a larger private project.** Several modules the server
lazily imports (`learned_dinner`, `mug_visual_profile`, `wide_bottle_task`) are
*not* in this repo; calls into those code paths are guarded with
`try/except ImportError` and raise a clear `ValueError` instead of crashing. Don't
"fix" those imports by stubbing the modules — that gating is intentional.

## Running it

```powershell
pip install -r requirements.txt
python -m simulation_lab.server
```

Then open `http://127.0.0.1:8765/` (full engineering console, chemistry + dinner
scenes, manual joint control, recording) or `http://127.0.0.1:8765/demo` (judge-
facing dashboard: expert-teacher video gallery + live "try it now" against the
real simulator).

The server is local-only by design: `TrustedHostMiddleware` restricts to
`127.0.0.1`/`localhost`/`testserver`, and a same-origin check rejects cross-origin
mutating requests. Preserve both if you touch `server.py`.

## Tests

Tests are `unittest.TestCase` classes (not pytest-specific), run under pytest or
unittest interchangeably:

```powershell
python -m pytest tests/
python -m pytest tests/test_dinner_autonomy.py -k test_name
python -m unittest tests.test_autonomy
```

`pytest` is not in `requirements.txt` — install it separately if it's missing from
your environment. Most tests build a real MuJoCo model/data pair (via
`simulation_lab.scene.build_scene`) and step actual physics; there is no mocked
physics layer. A common assertion pattern (see `tests/test_autonomy.py`) advances
the sim tick-by-tick and asserts a controller never mutates `qpos` directly
("Controller teleported a physical coordinate") — only `mj_step` may move state.

## Architecture

### Process model (`simulation_lab/engine.py`)

`LabEngine` owns physics on a dedicated thread and runs a fixed-timestep loop
(target 200 Hz control) driven by a `queue.Queue[Request]`. HTTP handlers in
`server.py` never touch MuJoCo state directly — they call `engine.submit(op,
payload)`, which enqueues a `Request`, blocks (with a 15s timeout) on a
`threading.Event`, and returns a deep-copied state snapshot. Camera rendering runs
in a **separate `multiprocessing` process** (`rendering.py`) with its own MjModel
compiled from the same XML, communicating over bounded queues — this keeps
JPEG encoding off the physics thread and means the physics state and the frame
you see are never the same Python object.

Four request kinds: `reset`, `layout`, `task`, `language` (plus bare joint/camera
control). All mutate `LabEngine` state only from inside `_run()`.

### Scene + controller hierarchy

- `scene.py` composes the licensed SO-101 arms (`assets/so101/`) with a seeded
  scene; `dinner.py` builds the dinner-specific objects (bottle, plate, mug, fork,
  spoon) with two geom layers (visual vs. collision) and seeded cabinet starts vs.
  declared final poses.
- `autonomy.py`'s `LiftReturn` is the base contact-only teacher (IK via a
  **separate scratch `MjData`**, never the live one — perturbing IK guesses must
  not leak into physics). `dinner_autonomy.py`'s `DinnerSequence`/`DinnerTask`
  subclasses it into a per-object state machine (`planning → approach → descend →
  close → retry → lift → hold → transit → rotate → align → lower → release →
  retract → park → verify`), one skill per object in `SKILLS = ('bottle', 'plate',
  'mug', 'fork', 'spoon')`.
- `vla_task.py`'s `SmolVLATask` also subclasses `LiftReturn` but overrides control
  entirely with real SmolVLA inference (loads the checkpoint, tokenizes the
  instruction, runs pre/post-processing by hand — see the module docstring for
  why it doesn't use `lerobot`'s own config loader). It picks a wrist camera by
  keyword match against the instruction (`_WRIST_BY_KEYWORD`), not real NLU.
- `command_task.py` / `language.py` implement a bounded, explicit grammar for
  typed table-setting instructions (**not** an LLM) — `parse_command` returns a
  typed plan, `CommandSequence` executes it against the same scripted skills.

Most other `simulation_lab/*.py` files (`rgb_servo_*`, `mug_*`, `retrieval_policy`,
`sequence_policy`, `primitive_policy`, `bottle_refinement*`) are earlier,
self-contained learned-control **experiments** predating the SmolVLA pipeline —
each file's docstring states its exact input/output contract (e.g. "RGB only, no
simulator-state input"; "never generates motor targets, may only stop a trial").
Preserve that boundary language when touching them: it's enforcing a specific
"what this component is/isn't allowed to see" invariant used across the codebase,
not incidental phrasing.

### Data protocol (`duet_protocol.py`)

Seeds are partitioned and enforcement is real, not advisory:
- `1000–1099` training (may generate demonstrations),
- `2000–2019` validation (may select a model, may not update weights),
- `3000–3009` evaluation (held out; untouched until a candidate is frozen).

`require_partition()` / `is_training_seed()` are called from `engine.py` before
any recording starts — e.g. `_task_command` refuses to record on a non-training
seed. When adding new recording/training entry points, route them through
`duet_protocol`, not ad hoc seed checks.

### Data + model pipeline (`scripts/`)

Only five scripts are part of the current SmolVLA pipeline (see
`scripts/README.md` for the full table): `collect_dinner_learning.py` (verified
demos) → `prepare_lerobot_dataset.py` (LeRobot v3 dataset) →
`build_expert_gallery.py` (gallery videos), plus `capture_target_arrangement.py`
and `design_dinner_layout.py`. The rest of `scripts/` supports earlier, separate
experiments and is not wired into the current pipeline — don't assume a script
there is still load-bearing without checking `scripts/README.md` first.

### Frontend

Two independent single-page apps under `simulation_lab/web/`, both served as
static files by `server.py`: `index.html`/`app.js` (full engineering console) and
`demo.html`/`demo.js` (judge-facing dashboard, tabbed expert-gallery vs. live-VLA
view). They intentionally do not share JS — `test_server_contract.py` asserts
specific controls exist/don't exist in each.

### Licensing / attribution

`assets/so101/` is derived from MuJoCo Menagerie (Apache 2.0) with modified
visual meshes only — see `simulation_lab/NOTICE.md` and
`THIRD_PARTY_NOTICES.md` before changing anything under `assets/`. Dinner assets
are original to this project (MIT).
