# Duet 2 progress tracker

Updated: 2026-09-15. Canonical plan: `DUET_IMPLEMENTATION_PLAN.md`.

## Current status

**Planning complete; Talos source imported into Documents/Duet 2. Runtime verification and feature implementation pending.**

Approach: modify the Talos ZIP into Duet with attribution. Retain working control infrastructure, add original scene/interface, collect varied demonstrations, improve learned behavior, and evaluate on unseen seeds.

Existing `Documents/Duet` work is preserved. Its README results are unverified and are not counted below.

## Milestones

| ID | Deliverable | State | Dependency | Evidence required |
| --- | --- | --- | --- | --- |
| P0 | Inspect reference and competition brief | Done, targeted inspection | None | Relevant source files and all five PDF pages read |
| P1 | Implementation plan and tracker | Done | P0 | These two Markdown documents |
| M0 | Extract/import Talos and reproduce baseline | In progress | P1 | ZIP/import recorded below; environment and trial logs pending |
| M1 | Original Duet assets and seeded scene | In progress | M0 | Scene tests, parameter manifests, visual check, physical rollout |
| M2 | Training dataset and frozen splits | In progress | M1 | Validated episodes, coverage and provenance report |
| M3 | Improved learned placement candidate | Not started | M2 | Training log, checkpoint, closed-loop validation comparison |
| M4 | Bounded recovery | Not started, optional | M0, failure evidence | Fault tests and matched recovery comparison |
| M5 | Original Duet interface | Not started | M0; finalize with M1/M3 | Browser golden path, stop/error and layout checks |
| M6 | Held-out 10-seed evaluation | Not started | Frozen scene/model | Complete results and original recordings |
| M7 | Intel/OpenVINO verification | Not started | Model export and Intel access | Actual device report, parity and benchmark logs |
| M8 | Submission package | Not started | M5–M7 | Reproduction instructions, video, architecture and provenance |

Allowed states: Not started, In progress, Blocked, Done, Deferred. Mark Done only with recorded evidence; add a reason when Blocked or Deferred. No numerical percentage while work scope remains conditional.

## Decision log

| Decision | Current agreement |
| --- | --- |
| Starting point | Talos ZIP, modified in place within a new safe working copy |
| Identity | Duet; original appearance and measured improvements |
| Reuse | Retain useful source/models with clear attribution and provenance |
| Task | Dinner-table workflow with two simulated SO-101 arms |
| Hardware | User reports organizer acceptance of 8th-generation Intel; exact target unconfirmed |
| Independent rebuild | Superseded by user's explicit direction to modify Talos |
| Kit packing | Superseded by dinner-table competition scope |
| Existing Documents/Duet | Preserve; contains uncommitted changes, not verified as current baseline |

## Open dependencies and risks

- Exact submission cutoff, timezone and portal URL are unknown.
- Intel target access and hardware details have not been verified. Current workstation was identified as AMD Ryzen 5 5600GT.
- Organizer reuse/AI-assistance rules are not specified in the supplied technical PDF; preserve any confirmation.
- Baseline runtime and installed dependencies have not been tested here.
- Visual/geometry changes can invalidate perception and learned grasp trajectories; stage variation and recollect matching observations.
- Learned placement improvement is a hypothesis until evaluated. Data collection/training duration is unmeasured.

## Dataset ledger

| Split | Planned seeds | Attempted | Valid successful episodes | Failures retained | State |
| --- | --- | --- | --- | --- | --- |
| Training | 1000–1099 | 0 | 0 | 0 | Not collected |
| Validation | 2000–2019 | 0 | 0 | 0 | Not collected |
| Final evaluation | 3000–3009 | 0 | 0 | 0 | Locked by plan; manifest pending |

Counts describe this task's new pipeline, not the separate existing project's files. Do not use final evaluation data for training or calibration. Track configuration hashes and actual scene overlap as well as seeds.

## Transformation log

### 2026-09-15 - Duet identity and seed protocol

- Changed the dinner scene's display-only palette, work zones, scene model name and challenge metadata. Collision shapes, grasp sites, objects, target locations, and control interfaces remain unchanged in this first visual transformation.
- Renamed the browser entry point to Duet 2 and applied the indigo/coral service theme.
- Added simulation_lab/duet_protocol.py and scripts/collect_duet_demos.py. Collection can now use only the declared training partition; it rejects final-evaluation seeds before it creates a run.
- Added declared seed ranges: training 1000-1099, validation 2000-2019, evaluation 3000-3009. No data has been collected yet.
- Verification: compiled the added/changed Python files; scene generation loaded as Duet dinner service with the indigo-coral service metadata; training seeds 1000 and 1001 were accepted; evaluation seed 3000 was correctly rejected by a training run.
- Not yet verified: full grasp/place sequence after the visual change, browser rendering, new demonstrations, learned-policy performance, or Intel runs.

### 2026-09-15 - First randomized physical Duet run

- Ran the inherited physical teacher on Duet 2 seed 1000 after the scene-variation change.
- Outcome: succeeded across all six skills. The report recorded 250.21 simulated seconds, 12 actuators, zero external forces, zero equality constraints and zero controller state writes.
- Artifact: .run/duet-2-teacher-seed-1000.json. This is a physical baseline only, not learned-policy, Intel, or final ten-seed evidence.
- Next: collect a small training-only pilot on seeds 1000-1009, then select the placement behavior to improve.

### 2026-09-15 - Duet evening scene assets

- Added original generated backdrop asset at simulation_lab/assets/duet/duet-evening-wall-v1.png and mounted it as a non-colliding MuJoCo wall.
- Added Duet work mats, service runner, centerpiece, wall sill, object labels and an indigo/coral material direction.
- Verified the generated image visually and verified that seed 1000 scene compilation succeeds with the backdrop asset present.
- Physical object geometry is intentionally unchanged in this revision. New camera-trained data is required before claiming that inherited learned policies handle the Duet appearance.

### 2026-09-15 - Duet procedural tableware pass

- Added visual-only sunray plate accents, a banded/fluted cup, carafe service bands, cutlery inlays and a decorated drawer facade.
- Verified that all new visual geoms compile into the seed 1001 MuJoCo scene. The model retained 12 actuators and zero equality constraints.
- The next project milestone is fresh Duet demonstration data. Existing learned models must not be described as validated against the new camera appearance.

### 2026-09-15 - First Duet training-data batch

- Collected one new Duet 2 training batch from seed 1000 only. The collector used the Duet scene, new backdrop and procedural tableware visuals.
- All six skills - bottle, plate, mug, drawer, fork and spoon - completed and passed the independent physical replay gate.
- Artifact: datasets/duet-2-pilot-seed-1000. It contains 40 files totaling 17.91 MiB, including the protocol manifest, scene XML, initial state, RGB observations, joint state, target actions, task stages and per-skill outcome manifests.
- The batch is training-eligible but is only one scene. It is a pipeline proof, not sufficient data to train or evaluate a policy.

### 2026-09-15 - Seed 1001 physical test and failure video

- Ran the inherited exact-state physical teacher on Duet 2 seed 1001 with `.\.venv\Scripts\python.exe scripts\evaluate_dinner_autonomy.py --seeds 1001 --output .run\duet-2-teacher-seed-1001.json`.
- Outcome: failed after 51.005 simulated seconds because both fingers did not establish a grasp after the allowed attempts. The report records zero physical state writes, zero external forces and the failed task state.
- Exported `.run/duet-2-seed-1001-test.mp4` from the same physical rollout path. The 572-frame overhead video includes a small seed/result caption and retains the failure; it is not learned-policy, Intel, or final-evaluation evidence.
- Video report: `.run/duet-2-seed-1001-video.json`; SHA-256 `445def3545410d2d11974274fdc5b73d6e5e441dddae8e5a60d78644ac3b3379`.

### 2026-09-15 - Dinner mesh presentation cleanup

- Fixed the authored plate mesh so both plate variants have a continuous visual centre instead of rendering as hollow rings.
- Kept all vessel, plate and cutlery collision proxies unchanged; only visual mesh data, dinner table presentation colour and camera-hidden target-guide groups changed.
- Made the mesh authoring script Windows-console safe by removing non-ASCII progress glyphs.
- Verification: `.\.venv\Scripts\python.exe scripts\author_dinner_meshes.py`, `.\.venv\Scripts\python.exe -m unittest tests.test_dinner -v`, and `.\.venv\Scripts\python.exe scripts\render_preview.py --seed 1001 --width 1280 --height 720`.
- Result: five mesh assets regenerated, all four focused dinner physics tests passed, and six seed-1001 preview images were written under `outputs/previews/`.
- The full runtime suite still reports the two existing baseline failures in relay placement and the complete contact-only sequence; both reproduce in a clean `HEAD` worktree and are unrelated to this visual-only cleanup.

### 2026-09-15 - Bottle base and visible tableware cleanup

- Changed the authored carafe mesh from a narrow circular foot to a full-width flat base and regenerated `simulation_lab/assets/dinner/meshes/carafe.obj`.
- Hid the supporting side plate and tumbler from presentation camera groups so the rendered task scene shows exactly one plate and one cup. Their physical bodies remain camera-hidden for existing validation contracts.
- Verification: `.\.venv\Scripts\python.exe scripts\author_dinner_meshes.py`, `.\.venv\Scripts\python.exe scripts\render_preview.py --seed 1001 --width 1280 --height 720`, and `.\.venv\Scripts\python.exe -m unittest tests.test_dinner -v`.
- Result: carafe base vertices span approximately 21 mm radius at z=0, the rendered seed-1001 scene shows one plate/cup, and all four focused dinner physics tests pass.

### 2026-09-15 - Base scene identity fix (floor/sky/table, not just tableware)

- Root cause found: earlier "Duet identity" passes only recolored objects in `dinner.py`; the shared `simulation_lab/scene.py` used by every scenario still carried Talos's unmodified stock MuJoCo checker floor texture and generic blue-grey sky gradient. This, not the tableware, was why the scene still read as the upstream template.
- Changed `scene.py`: floor texture from `builtin="checker"` to a flat solid dark plum tone; sky gradient shifted to indigo/plum; `table_mat`/`edge_mat` shifted to a warm tan/plum palette; each arm's base pad now gets a distinct coral (left) / cobalt (right) color instead of the shared grey `edge_mat`.
- Also found the generated `simulation_lab/assets/duet/duet-evening-wall-v1.png` backdrop described in `docs/DUET_ASSETS.md` is never referenced by any texture in code — it is an orphaned, unused asset. Not wired in; a plain-color material was used instead to avoid texture-mapping risk under the deadline.
- Change is visual-only: same geom count/sizes/materials-by-name structure, no collision or actuator changes.
- Verification: `.\.venv\Scripts\python.exe scripts\render_preview.py --seed 1001 --width 640 --height 360` (visually confirmed floor/sky/table/pads changed, no checkerboard) and `.\.venv\Scripts\python.exe -m unittest tests.test_dinner -v` (4/4 pass, unchanged).
- Open note: organizer feedback says the drawer is only an example fixture, not mandatory — the environment can be redesigned (e.g. a shelf to pick crockery from instead of a drawer). This is a geometry-level decision for a later stage, not part of this visual-only pass; needs its own scoping before touching `add_drawer`/pickup logic.

### 2026-09-15 - Layout-mirror attempt: reverted, findings recorded

- User flagged that object/drawer coordinates in `dinner.py` are the exact same absolute numbers as the Talos baseline (colors changed, composition did not) — a real fingerprint risk for published training data, not just a visual one.
- Attempted a full left-right mirror of drawer/target/object coordinates plus the corresponding arm-side hardcoding in `dinner_autonomy.py`/`command_task.py`. This broke physical reachability: several skills (plate, mug, the spoon retrieval routine) have hardcoded world-frame approach-direction vectors (`dinner_autonomy.py` lines ~66, 118, 286, 294-295) tuned for the original hemisphere; mirroring positions without re-deriving each of these caused "outside this arm's useful reach" failures.
- Also tried a smaller rigid translation (~1.2cm) of the same coordinates, keeping the original hemisphere/arm assignment. This too pushed `test_relay_expands_and_physically_finishes` and `test_complete_contact_only_sequence` outside tolerance, confirming the existing layout is tightly tuned with little slack (consistent with the previously-recorded 8/10 baseline success rate).
- Reverted all position/coordinate edits back to the exact original values. Verified via `git stash` that the two remaining test failures (`test_complete_contact_only_sequence`, `test_relay_expands_and_physically_finishes`) reproduce byte-for-byte identically on a clean, untouched checkout — confirmed pre-existing and unrelated to any change made today.
- Decision: defer real layout/composition redesign (mirror, reordering, or the organizer-suggested shelf-instead-of-drawer swap) to its own properly scoped and physically-verified task. Do not rush a coordinate change without re-deriving every hardcoded approach vector it touches and re-validating each affected skill individually.
- Kept: the `scene.py` floor/sky/table palette fix and per-arm base-pad colors (visual-only, zero physics risk, already verified).

### 2026-09-15 - Fresh seed-1000 demonstration and video on the new scene identity

- Collected a new batch: `.\.venv\Scripts\python.exe scripts\collect_duet_demos.py --output datasets/duet-2-demo-seed-1000 --seeds 1000`. All six skills (bottle, plate, mug, drawer, fork, spoon) succeeded physically and passed the independent replay gate under the new floor/sky/table/base-pad palette (original object positions, per the revert above).
- Exported a review video: `.\.venv\Scripts\python.exe scripts\export_duet_pilot_video.py --batch datasets/duet-2-demo-seed-1000/seed-1000 --output .run/duet-2-seed-1000-demo.mp4` (546 frames). Visually confirmed coral/cobalt arm base pads and recolored tableware are visible in the recorded overhead camera.

### 2026-09-15 - Standalone exact dinner layout editor

- Added `scripts/design_dinner_layout.py`, a desktop Tkinter editor with no server or browser dependency. It draws the actual `0.96 m x 0.78 m` table and loads the current Task-start object positions for a selected seed, defaulting to seed 42.
- Added `simulation_lab/dinner_layout.py` JSON serialization and scene consumption so exported world-space body poses can be passed to `build_scene(..., dinner_layout=...)`.
- Verification: `.\.venv\Scripts\python.exe scripts\design_dinner_layout.py --help`, `.\.venv\Scripts\python.exe -c "from scripts.design_dinner_layout import current_poses; print(len(current_poses(42)))"`, and `.\.venv\Scripts\python.exe -m unittest tests.test_dinner_layout tests.test_dinner -v`.
- Result: seven dinner objects load from the seeded current arrangement, exact x/y/z/yaw values can be edited/exported, and all seven layout/dinner contract tests pass.
- Follow-up: the editor now also shows both robot bases and their settled gripper positions, and exports robot base/gripper/joint metadata in the same JSON without requiring the server or browser.
- Correction: the editor default now loads the intended Target example arrangement rather than Task start, so fork/spoon begin on the tabletop. `--arrangement task` remains available for inspecting the drawer-start state; robot positions are the exact settled home/reset pose.
- The 1-seed pilot batch at `datasets/duet-2-pilot-seed-1000` from the earlier (pre-color-fix, pre-mesh-cleanup) pass is now stale relative to the current scene identity — it must not be published or used as training data as-is. Superseded by `datasets/duet-2-demo-seed-1000` for anything beyond pipeline-proof purposes; a real training batch still needs to be collected across the full 1000-1099 range after any further scene decisions (layout/shelf) land.
- Next: decide on the layout/shelf redesign (own scoped task) before bulk data collection, since any further geometry change invalidates whatever is collected first.

### 2026-09-15 - Fork path-around-plate recovery

- Added `simulation_lab/carry_routing.py`: a small planar detour selector that inflates an obstacle by the carried object's footprint and validates every route segment before returning the shortest side route.
- The physical dinner teacher now preserves its direct fork transfer whenever the existing carried-object collision check accepts it. If that check rejects direct transfer because the held fork intersects another object, the teacher plans a two-corner route around the physically placed plate, then runs the entire joint trajectory through the existing carried-object contact checker before commanding it. No pose writes, forces, equality constraints, or grasp abstraction were added.
- Verification: `tests.test_carry_routing` and `tests.test_dinner` passed (5 tests); `py_compile` passed for the new routing module and dinner teacher; the six-skill physical teacher run on seed 1000 succeeded with zero controller state writes, external forces, and equality constraints (`.run/fork-route-after-seed1000.json`). The direct route was already clear on that seed, so the new contingency was not exercised there.
- Remaining evidence gap: the reported original fork/plate collision has not reproduced in the current workspace's seed-1000 or seed-42 configurations (seed 42 presently fails earlier at plate/cabinet clearance). Preserve or provide the exact failing seed/layout manifest to execute and retain an end-to-end detour rollout before claiming the historical case is closed.

### 2026-09-15 - Root-caused the position fragility; generalized grasp/place orientation search

- **Root cause identified**: `DinnerTask._select_item`/`_plan` in `simulation_lab/dinner_autonomy.py` pick a wrist-roll orientation (`ik.x_target`) via **hardcoded world-frame constants** per skill (e.g. mug `[.55,-sqrt(1-.55**2),0]`, plate `[cos(-1.0),sin(-1.0),0]`), tuned only to work at the *original* Talos spawn/target position for that object. `ArmIK.solve` (`simulation_lab/autonomy.py`) is a real, general damped-least-squares Jacobian IK solver — the fragility is entirely in these memorized orientation constants, not the solver. A 5-DOF arm cannot satisfy an arbitrary fixed position *and* fixed orientation simultaneously everywhere, so moving an object's spawn or target by even a few mm can make the fixed constant geometrically infeasible, producing "outside this arm's useful reach" regardless of how much slack the raw position change looks like it should have. This explains every reachability failure hit this session (mug placement, plate pick/place under a moved spawn) and the two mirror/shift attempts recorded in the 2026-09-15 "Layout-mirror attempt" entry above.
- **Fix**: added `DinnerTask._search_grasp_direction` (pick: grasp + hover) and `DinnerTask._search_align_point` (place) in `dinner_autonomy.py`. Both try the direction from the acting arm's own base toward the actual target first, then 12 swept fallback angles, keeping the first direction the IK solver can actually reach — replacing the hardcoded constants with a computed search. Wired in for **mug and plate only** so far (the two skills exercised tonight); bottle/fork/spoon still use their original hardcoded/skill-specific paths untouched.
- **Verified non-regressing**: full `tests` suite (76 tests) still lands on exactly the same 2 pre-existing failures as every prior session checkpoint (`test_complete_contact_only_sequence`, `test_relay_expands_and_physically_finishes`, identical error values) — the search reproduces the original working solution for the original, untouched positions. Standalone mug test (`.265,.015` target) now succeeds with **2.3mm** placement error (better than the earlier one-off `x_target=None` patch's 5.85mm), and that ad-hoc patch has been replaced by the general search.
- **New, more precise finding on the current seed-42 fork failure**: it is a **pick-stage** failure ("left: insufficient clearance cabinet_roof/left_moving_jaw; right: outside reach"), not the mid-carry plate collision the new `carry_routing.py` detour (previous entry) targets. The fork's *approach into the still-mostly-closed drawer geometry* is blocked before it ever grasps, so the detour fix and this session's search generalization do not yet touch this specific failure. Next session should treat this as a third, distinct issue from (a) the mirror/reach fragility and (b) the mid-carry plate collision.
- **Custom full-layout test** (`.run/dinner-layout.json`, a captured scene snapshot including robot pose): loads cleanly via the existing `dinner_layout` machinery (`simulation_lab/dinner_layout.py`, wired into `simulation_lab/dinner.py`'s `add_dinner_scene` as `exact_layout`, overriding **spawn** position only, not placement targets). Confirmed **physically stable** (0mm penetration, all seven objects settle flat, zero tilt). Confirmed **not yet end-to-end pickable**: `bottle` succeeds from its new spawn; `plate` now reaches grasp+hover+lift+hold with the new search (previously failed immediately), but then fails again because **side selection during `_plan()` only validates grasp reachability, not whether the same arm can also complete the placement afterward** — plate's new near-center spawn lets the *right* arm win the grasp search, but the right arm then cannot reach plate's original *far-left* placement target while carrying it. mug/drawer/fork/spoon not yet tested under this custom layout. The live app (`simulation_lab/engine.py` `_reset`) does **not** currently pass `dinner_layout` — it is not wired as any kind of default yet.
- Kept uncommitted in the working tree: `simulation_lab/dinner_autonomy.py` (the two search helpers above), `simulation_lab/dinner.py` (spoon color `#4caf50` for visibility, mug target `[.265,.015,TABLE_Z]`; drawer and fork/spoon storage/target positions are back at the exact original Talos values after this session's repositioning attempts were found to break reachability with near-zero slack). See `docs/SESSION_HANDOFF.md` for the full next-session brief.

## 2026-09-15 — drawer-free cabinet service implemented

- User JSON is the FINAL requested arrangement, preserved in `config/dinner-layout.json`; independent `cabinet_source.py` creates accessible seeded starts. Glass and side plate stay fixed. Cutlery target height is physical table support, not historical elevated drawer height.
- Removed drawer geometry, slide joint, task, prerequisite and UI controls. New sequence: bottle → plate → mug → fork → spoon. Live simulator transforms rotate local grasp frames; arm/grasp choice validates pickup, placement and retreat reach. Carry trajectories constrain object center directly, avoiding wrist-offset iteration drift. No authoritative object position/velocity writes, applied forces or equality attachments.
- Root causes resolved: vessel grasps require placement/retreat reach validation; carry-center IK must account for changing grip offsets; full vessel-sized jaw sweeps hit plate/mug at fork release. Thin cutlery now opens 0.12 rad and chooses a feasible grasp nearest COM instead of an unstable far-handle grip. Original mass/contact proxies and carrying-tilt safety limits remain intact.
- `.\.venv\Scripts\python.exe scripts/evaluate_dinner_autonomy.py --seeds 1000 --output .run/cabinet-aperture-1000.json`: all five skills succeeded. `--seeds 42,1001,1002 --output .run/cabinet-rotation-checks.json`: 3/3 full sequences succeeded. All placements below 3 mm XY error; both arms parked; no hidden writes/forces/equalities. This covers sampled seeded variation, not arbitrary poses outside useful reach.
- `.\.venv\Scripts\python.exe -m unittest tests.test_dinner tests.test_dinner_layout tests.test_spawn_grasp tests.test_carry_routing tests.test_dinner_autonomy tests.test_dinner_monitor tests.test_command_task tests.test_autonomy -q`: 28/28 passed. Additional server/web contracts: 3/3 passed. Updated center-grasp assertion plus server checks: 4/4 passed. `node --check simulation_lab/web/app.js` and `git diff --check`: passed.
- `.\.venv\Scripts\python.exe scripts/collect_duet_demos.py --output datasets/duet-2-cabinet-seed-1000 --seeds 1000`: 5/5 real-contact demonstrations succeeded AND independently replayed saved float32 motor actions with scoring-only monitor (zero teacher updates). No unexpected contacts or unsupported finger gaps. Replay XY errors: bottle 0.44 mm, plate 2.32 mm, mug 2.48 mm, fork 0.25 mm, spoon 0.35 mm.
- `.\.venv\Scripts\python.exe scripts/export_duet_pilot_video.py --batch datasets/duet-2-cabinet-seed-1000/seed-1000 --output .run/duet-2-cabinet-seed-1000.mp4`: exported 465 frames in physical skill order. Historic drawer videos remain unchanged.
- Failed development reports retained under `.run/cabinet-*.json`, including handle tilt and carry-center/release-clearance failures. Old six-skill models/recordings remain historic, never relabelled or used as hidden cabinet fallback. Incompatible learned modes are disabled/rejected. Exact-state programmed simulator evidence only; no trained-policy/hardware/official ten-seed result claimed.
- Installed already-pinned FastAPI/Uvicorn into `.venv` for browser verification. Browser confirmed live camera, five-skill goal, disabled incompatible models, explicit drawer-command refusal, and set-table acceptance. Fixed strict command schema to accept the UI's existing recording flag; non-training recording requests are explicitly rejected rather than silently disabled.
- Full browser instruction `set the table`, seed 42, recording off: **SUCCEEDED**, completed 5 physical skills, objects released, both arms parked, 210.9 simulation seconds. Browser placement errors: bottle 0.6 mm, plate 2.4 mm, mug 0.3 mm, fork 0.2 mm, spoon 0.1 mm. Browser tab left open as deliverable. Video's final overhead frame visually inspected at `.run/cabinet-final-preview.png`.
- Existing recording metadata integration retained with overlapping engine changes. Language/server contracts passed; recording tests require discovery because of their existing sibling import: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_recording.py -q`: 3/3 passed. Existing mesh assets and editor are included as scene/test dependencies; unrelated hosting, speech and session changes remain unstaged.

## Final evaluation ledger

| Seed | Baseline outcome | Duet outcome | Failure/recovery | Recording |
| --- | --- | --- | --- | --- |
| 3000 | Not run | Not run | Not observed | None |
| 3001 | Not run | Not run | Not observed | None |
| 3002 | Not run | Not run | Not observed | None |
| 3003 | Not run | Not run | Not observed | None |
| 3004 | Not run | Not run | Not observed | None |
| 3005 | Not run | Not run | Not observed | None |
| 3006 | Not run | Not run | Not observed | None |
| 3007 | Not run | Not run | Not observed | None |
| 3008 | Not run | Not run | Not observed | None |
| 3009 | Not run | Not run | Not observed | None |

Success rate: not measured. No completed trial exists in this tracker.

## Update protocol

After each logical unit, record: date/time, stage, files changed, commit, exact verification command, observed result, artifact paths, blocker and next action. Commit verified changes in the working project. Preserve unsuccessful trials. Do not turn intended outcomes into reported results.

## Next action

M0: hash and safely extract the Talos ZIP into a non-conflicting Documents folder, preserve upstream notices, commit the untouched baseline, and reproduce one documented dinner trial before modification.

## Duet 2 setup evidence

- Working folder: `C:/Users/Prince/Documents/Duet 2`.
- Source ZIP SHA-256: `4FB1C82B91FCFBC161D035CC167602099E16A96F1ABFAB7394CB56E97A5326B2`.
- Imported 1,405 ZIP entries with destination-path validation.
- Untouched source commit: `6dc5878` (`import talos baseline`).
- Existing Documents/Duet preserved, including uncommitted control edits.
- Next: inspect setup instructions, establish an isolated runtime, and reproduce one documented dinner trial. Imported models, recordings, and results belong to the upstream baseline; no new Duet 2 outcomes are claimed.
