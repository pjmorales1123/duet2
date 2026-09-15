# Cabinet-Source Spawn Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the drawer workflow with a cabinet-zone source layout and pick rotated objects from their live simulator spawn frames.

**Architecture:** A versioned dinner-layout configuration supplies approved FINAL poses. cabinet_source.py supplies separate accessible seeded starts. The scene renders a visual-only cabinet zone with physical cutlery ledges. The dinner teacher transforms local cutlery grasp axes through live rotation, validates grasp/placement/release reach, and executes five skills without a drawer.

## Execution record (2026-09-15)

- [x] Tasks 1–4 implemented: final configuration, cabinet source, pose-based grasps, five-skill controller and commands.
- [x] Task 5 interface/schema checks: no drawer controls, incompatible learned models disabled, browser drawer refusal and set-table acceptance verified.
- [x] Physical checks: complete five-skill success on seeds 42, 1000, 1001, 1002; 28 runtime regression tests passed, plus 3 server/web contract tests.
- [x] Final live-browser completion: seed 42, text instruction, five skills succeeded; saved-action replay seed 1000 passed 5/5; 465-frame video exported and visually inspected. Exact commands and artifacts are recorded in docs/DUET_PROGRESS.md.

The detailed original steps below are retained as the pre-implementation plan, not literal implementation claims. Necessary corrections: the user's JSON is a FINAL layout, not starts; exact carry-center IK replaces iterative wrist-offset guessing; pickup candidates must also permit placement/retreat; thin-cutlery jaw opening avoids neighboring objects without an unstable far-handle grip. No training or final ten-seed evaluation claimed.

**Tech Stack:** Python 3.12, NumPy, MuJoCo, `unittest`, existing static web UI.

**Spec:** `docs/superpowers/specs/2026-09-15-cabinet-spawn-grasp-design.md`

## Global Constraints

- Use real grasp and contact physics; never teleport, weld, or externally force an item in a learned or teacher trial.
- The canonical layout is project-versioned, not loaded from `.run/`.
- Spawn position and yaw are exact simulator state for this teacher; do not describe it as camera inference.
- Preserve old drawer recordings/results as historic artifacts; do not relabel them as five-skill results.
- Retain failed physical trials and record verification evidence in `docs/DUET_PROGRESS.md`.

---

### Task 1: Make the approved source layout canonical

**Files:**
- Create: `config/dinner-layout.json`
- Modify: `simulation_lab/dinner_layout.py`
- Modify: `tests/test_dinner_layout.py`

**Interfaces:**
- Consumes: user-supplied `.run/dinner-layout.json` source poses.
- Produces: `canonical_dinner_layout() -> dict`, returning validated object poses only.

- [ ] **Step 1: Write the failing test**

```python
from simulation_lab.dinner_layout import canonical_dinner_layout

def test_canonical_layout_has_each_dinner_object_and_no_robot_state():
    layout = canonical_dinner_layout()
    assert set(layout['objects']) == {'bottle', 'fork', 'glass', 'mug', 'plate', 'side_plate', 'spoon'}
    assert 'robots' not in layout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_dinner_layout -v`

Expected: FAIL because `canonical_dinner_layout` does not exist.

- [ ] **Step 3: Write the minimal implementation**

Copy only `coordinate_frame`, `objects`, `scene`, and `schema_version` from the supplied JSON into `config/dinner-layout.json`. Add:

```python
CANONICAL_LAYOUT_PATH = Path(__file__).parents[1] / 'config' / 'dinner-layout.json'

def canonical_dinner_layout() -> dict:
    return load_dinner_layout(CANONICAL_LAYOUT_PATH, ('plate', 'side_plate', 'mug', 'glass', 'bottle', 'fork', 'spoon'))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_dinner_layout -v`

Expected: PASS and canonical item poses load without robot-state fields.

- [ ] **Step 5: Commit**

```bash
git add config/dinner-layout.json simulation_lab/dinner_layout.py tests/test_dinner_layout.py
git commit -m "add canonical cabinet source layout"
```

### Task 2: Add a visual cabinet zone and use canonical spawns

**Files:**
- Modify: `simulation_lab/dinner.py`
- Modify: `tests/test_dinner.py`

**Interfaces:**
- Consumes: `canonical_dinner_layout()`.
- Produces: `layout['source_zone']` with `kind == 'cabinet_zone'`; all configured free-body source poses are used by scene construction.

- [ ] **Step 1: Write the failing test**

```python
def test_cabinet_zone_groups_canonical_sources_without_a_drawer_joint(self):
    model, data, layout = load(seed=1000)
    self.assertEqual(layout['source_zone']['kind'], 'cabinet_zone')
    with self.assertRaises(KeyError):
        model.joint('drawer_slide')
    for object_id, pose in canonical_dinner_layout()['objects'].items():
        np.testing.assert_allclose(data.body(object_id).xpos[:2], pose['position_m'][:2], atol=.004)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_dinner.DinnerTests.test_cabinet_zone_groups_canonical_sources_without_a_drawer_joint -v`

Expected: FAIL because the scene still adds `drawer_slide` and ignores the canonical layout by default.

- [ ] **Step 3: Write the minimal implementation**

In `add_dinner_scene`, default `exact_layout` to `canonical_dinner_layout()` when no explicit layout is supplied. Replace `add_drawer` with `add_cabinet_zone(world, bounds)` that creates only `contype="0"`, `conaffinity="0"` frame geoms around the bounding box of canonical XY positions. Return:

```python
{'kind': 'cabinet_zone', 'bounds_m': [min_x, max_x, min_y, max_y], 'physical_support': 'table'}
```

Do not create joints, moving bodies, or collision geoms for the cabinet zone.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_dinner -v`

Expected: PASS; source objects settle on the real table and no drawer joint exists.

- [ ] **Step 5: Commit**

```bash
git add simulation_lab/dinner.py tests/test_dinner.py
git commit -m "replace dinner drawer with cabinet zone"
```

### Task 3: Transform cutlery grasp orientation from live spawn rotation

**Files:**
- Create: `simulation_lab/spawn_grasp.py`
- Modify: `simulation_lab/dinner_autonomy.py`
- Create: `tests/test_spawn_grasp.py`

**Interfaces:**
- Produces: `world_grasp_axis(spawn_rotation: np.ndarray, local_axis: np.ndarray) -> np.ndarray`.
- Consumes: `data.xmat[object_body].reshape(3, 3)` and the local cutlery axis `[-1., 0., 0.]`.

- [ ] **Step 1: Write the failing test**

```python
def test_rotates_a_local_fork_grasp_axis_with_the_spawn_yaw(self):
    rotation = yaw_rotation(math.pi / 2)
    np.testing.assert_allclose(world_grasp_axis(rotation, [-1., 0., 0.]), [0., -1., 0.])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_spawn_grasp -v`

Expected: FAIL because `simulation_lab.spawn_grasp` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
def world_grasp_axis(spawn_rotation, local_axis) -> np.ndarray:
    axis = np.asarray(spawn_rotation, dtype=float) @ np.asarray(local_axis, dtype=float)
    length = float(np.linalg.norm(axis))
    if length == 0.:
        raise ValueError('A grasp axis must be non-zero.')
    return axis / length
```

In `DinnerTask._plan`, for `fork` and `spoon`, set `self.ik.x_target` to `world_grasp_axis(self.data.xmat[self.body].reshape(3, 3), [-1., 0., 0.])` after selecting the live object body. Remove the fixed `[-1., 0., 0.]` assignment.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_spawn_grasp -v`

Expected: PASS; a 90-degree spawn yaw yields a 90-degree world gripper constraint.

- [ ] **Step 5: Commit**

```bash
git add simulation_lab/spawn_grasp.py simulation_lab/dinner_autonomy.py tests/test_spawn_grasp.py
git commit -m "derive cutlery grasps from spawn rotation"
```

### Task 4: Remove drawer dependencies from teacher and command dispatch

**Files:**
- Modify: `simulation_lab/dinner_autonomy.py`
- Modify: `simulation_lab/command_task.py`
- Modify: `tests/test_dinner_autonomy.py`
- Modify: `tests/test_command_task.py`

**Interfaces:**
- Produces: `SKILLS == ('bottle', 'plate', 'mug', 'fork', 'spoon')`.
- Accepts: `DinnerSequence.start(kind='set_table')` and `DinnerSequence.start(kind='dinner_place', object_id='fork')` without drawer state.
- Rejects: `kind='drawer_open'` with an explicit unsupported-goal error.

- [ ] **Step 1: Write the failing tests**

```python
def test_five_skill_sequence_has_no_drawer_precondition(self):
    model, data, task, targets = setup()
    task.start(kind='dinner_place', object_id='fork')
    run_task(model, data, task, targets)
    self.assertEqual(task.status, 'succeeded')

def test_drawer_goal_is_rejected(self):
    model, data, task, targets = setup()
    with self.assertRaises(ValueError):
        task.start(kind='drawer_open')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_dinner_autonomy tests.test_command_task -v`

Expected: FAIL because fork requires `drawer_slide` and the drawer command is accepted.

- [ ] **Step 3: Write the minimal implementation**

Remove `DrawerTask`, `drawer_open` handling, `drawer_ready` expansion, and all `drawer_slide` checks. Keep `DinnerTask` only for five object IDs. Update completed-step assertions and message text to the five-skill contract. Do not change placement validation, hold validation, or the fork plate-detour recovery.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_dinner_autonomy tests.test_command_task -v`

Expected: PASS; fork is a cabinet-zone pick and drawer goals are rejected.

- [ ] **Step 5: Commit**

```bash
git add simulation_lab/dinner_autonomy.py simulation_lab/command_task.py tests/test_dinner_autonomy.py tests/test_command_task.py
git commit -m "remove drawer from dinner workflow"
```

### Task 5: Update UI, evaluation evidence, and physical verification

**Files:**
- Modify: `simulation_lab/server.py`
- Modify: `simulation_lab/web/index.html`
- Modify: `simulation_lab/web/app.js`
- Modify: `scripts/evaluate_dinner_autonomy.py`
- Modify: `docs/DUET_PROGRESS.md`
- Create: `tests/test_server_contract.py`
- Create: `tests/test_web_contract.py`

**Interfaces:**
- UI exposes only five physical pickup goals plus `set_table`.
- Evaluation report labels the run as a five-skill exact-state physical baseline.

- [ ] **Step 1: Write the failing UI/API contract tests**

```python
def test_dinner_goal_schema_excludes_drawer_open(self):
    self.assertNotIn('drawer_open', str(GoalRequest.model_json_schema()))
```

Add `tests/test_web_contract.py` with a text assertion that `simulation_lab/web/index.html` contains no `option value="drawer"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_server_contract -v`

Expected: FAIL because the API and UI still expose drawer controls.

- [ ] **Step 3: Write the minimal implementation**

Remove `drawer_open` reset/request fields and `drawer_open` goal enum value; remove drawer select/status markup and JavaScript branches. Update display copy to say “cabinet source zone” and “Run all five skills.” Change evaluation defaults/report schema text to five-skill wording.

- [ ] **Step 4: Run automated and physical verification**

Run:

```bash
./.venv/Scripts/python.exe -m unittest tests.test_dinner tests.test_dinner_autonomy tests.test_command_task tests.test_spawn_grasp -v
./.venv/Scripts/python.exe scripts/evaluate_dinner_autonomy.py --seeds 1000 --output .run/cabinet-spawn-grasp-seed-1000.json
```

Expected: focused tests pass; report contains exactly `bottle, plate, mug, fork, spoon`, zero controller state writes, zero external forces, zero equality constraints, and retained outcome data.

- [ ] **Step 5: Verify the browser golden path**

Start the existing local server, reset the dinner scene, confirm no drawer controls appear, select fork, and confirm the status identifies a cabinet-zone pickup. Run the five-skill sequence once and retain the report/video whether it succeeds or fails.

- [ ] **Step 6: Record evidence and commit**

Add exact command output, report paths, status, seed, and any failure to `docs/DUET_PROGRESS.md`.

```bash
git add simulation_lab/server.py simulation_lab/web/index.html simulation_lab/web/app.js scripts/evaluate_dinner_autonomy.py docs/DUET_PROGRESS.md tests/test_server_contract.py tests/test_web_contract.py
git commit -m "present cabinet source workflow"
```

## Plan self-review

- Spec coverage: canonical configuration (Task 1), non-blocking cabinet source (Task 2), live spawn rotation (Task 3), five-skill contract (Task 4), UI/evidence/browser verification (Task 5).
- Placeholder scan: no deferred implementation labels or unspecified error handling remain.
- Type consistency: `canonical_dinner_layout`, `world_grasp_axis`, and five-skill `SKILLS` are defined before their consumers.
