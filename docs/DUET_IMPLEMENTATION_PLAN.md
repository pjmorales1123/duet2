# Duet 2 — Talos-based implementation plan

Updated: 2026-09-15. Status: baseline imported; runtime verification and feature changes pending.

## Goal and source of truth

Start from `C:/Users/Prince/Downloads/talos-ai-infra-main.zip` and improve that working project into Duet. Retain useful Talos grasp/place control, physics, training infrastructure, and evidence practices with attribution. Add original assets and interface, generate new demonstrations across varied scenes, improve learned behavior, and evaluate on held-out seeds. This supersedes the earlier independent-rebuild and kit-packing plans.

This is a derivative project, not a claim of wholly original source code. Do not remove upstream notices or disguise inherited implementations. The user's latest instruction explicitly selects modification of Talos.

Execution: work inline through the gates below. Implement and verify one stage at a time; record evidence in `DUET_PROGRESS.md`. The user requested a separate Duet 2 project. The source import and project documentation are established; feature implementation remains planned.

## Constraints and confirmed context

- Competition: simulated dual SO-101 arms, MuJoCo dinner setup, language and camera observations, learned policy, randomized evaluation, OpenVINO, reproducible repository and video.
- User reports organizer acceptance of their 8th-generation Intel system. Treat it as the demonstration target; preserve confirmation in the submission records and report its exact model. Do not label it Core Ultra.
- Deadline is today according to the user; the exact cutoff/timezone and available Intel access remain unconfirmed. The technical PDF contains no cutoff or reuse eligibility policy.
- Keep grasp/placement physics real. No object teleportation or runtime weld attachment presented as physical grasping. Evaluation-only ground truth must not generate actions in a policy claimed to be camera-driven.
- Existing `C:/Users/Prince/Documents/Duet` contains files and uncommitted changes to `duet/config/constants.py`, `duet/control/grasp.py`, `duet/control/ik.py`, and `duet/control/skills.py`. Preserve them. This directory is not a verified Talos baseline.
- Its README claims SO-ARM100, a weld grasp abstraction, 10/10 results, and collection on seeds 0–19 with evaluation on 0–9. Those are unverified README claims; if the seed usage is accurate, it overlaps training and evaluation. Do not import those claims or results into Duet's new evidence.
- Context7, sequential-thinking, and `/init` are unavailable in the exposed tool set. At implementation kickoff, document this and create project instructions manually; use primary library documentation before changing external API integrations.

## Architecture and change boundaries

Keep Talos's working `simulation_lab/`, `models/`, `training/`, `scripts/`, and `tests/` layout initially. Do not restructure the whole project during the deadline window. Add small modules by responsibility and use existing interfaces after inspecting them.

| Area | Existing files to inspect before editing | Proposed changes |
| --- | --- | --- |
| Scene/assets | `simulation_lab/scene.py`, `simulation_lab/assets/dinner/` | Add `simulation_lab/duet_scene/` with appearance and physical randomization separated |
| Coordination | `learned_plan.py`, `learned_dinner.py`, `dinner_monitor.py` | Add `simulation_lab/duet_recovery/`; bounded recovery through the existing executor |
| Runtime | `engine.py`, `server.py` | Integrate scene/profile selection and recording without changing stable contracts unnecessarily |
| Interface | `simulation_lab/web/index.html`, `app.js`, `style.css` | Duet identity, original composition, task timeline, trial results; split new modules by responsibility |
| Dataset | `scripts/collect_dinner_learning.py`, `training/` | Add collection, validation, and manifest tooling for Duet scenes |
| Policy | selected profile and its actual training/export scripts | Train a versioned Duet candidate; preserve upstream baseline weights |
| Evaluation | existing evaluation and Intel scripts | Add consistent split-aware runner, failure artifacts, and hardware report |

Proposed new contracts: `SceneSpec(seed, split, physical, visual)`, `EpisodeManifest(scene_hash, profile_hash, outcome, failure_reason)`, and `TrialResult(seed, task, policy, success, metrics, artifacts)`. Define exact serialization and observation/action shapes from the inspected runtime before implementation. No assumption that the separate project's 24D observations are compatible.

## Stage 0 — establish the correct baseline

Files: new Talos-derived working folder under Documents; `AGENTS.md`, `README.md`, `.gitignore`, `.env.example`, ignored `credentials.md`, `docs/PROVENANCE.md`, `docs/BASELINE.md`.

- [ ] Record ZIP SHA-256 and inventory; validate extraction paths stay within the destination.
- [ ] Create a fresh, non-conflicting Talos-derived folder, proposed `C:/Users/Prince/Documents/Duet 2`. Preserve existing `Duet` and failed clone folders.
- [ ] Extract the ZIP with original source paths, license notices, models, and assets intact. Initialize Git; commit the untouched import before edits.
- [ ] Add project instructions and credential exclusions before any credentials are written. Preserve the existing setup README while documenting Duet changes.
- [ ] Inspect setup scripts before running them; create an isolated environment with the reference dependency versions.
- [ ] Run relevant baseline tests and one documented dinner trial. Record runtime, hardware, selected model, success/failure and logs.
- [ ] Save a baseline screenshot/video and profile hashes for later comparisons.

Gate: runnable baseline or a precise reproduction blocker, with an untouched import commit. Do not claim success solely because the repository includes successful recordings. Suggested commits: `import talos baseline`, then `document duet baseline and provenance`.

## Stage 1 — build original assets without invalidating control

Files: `simulation_lab/duet_scene/{__init__,appearance,randomization,spec}.py`, original dinner assets, `tests/test_duet_scene.py`, `docs/SCENES.md`.

- [ ] Create an original dinner setting: materials, placemats, backdrop, table presentation, labels and camera framing.
- [ ] Keep robot geometry, object collision shapes, reachability and camera calibration stable for the first visual comparison.
- [ ] Separate display-camera styling from policy-camera inputs. If policy views change, rerun perception tests and collect matching training data.
- [ ] Introduce bounded position, mass, friction, shape, lighting and background variation gradually. Reject object overlaps, invalid drawer starts and unreachable placements.
- [ ] Freeze numerical ranges in a versioned configuration after pilot trials; seed labels must correspond to actual recorded parameter changes.

Acceptance tests: same seed/config reproduces the scene; different seeds change recorded parameters; instances remain physically valid; baseline grasp and stable placement still execute. Visually inspect original assets. Commit: `add duet scene and bounded randomization`.

## Stage 2 — collect usable training data

Files: `scripts/collect_duet_demos.py`, `scripts/validate_duet_dataset.py`, `training/duet_v1/{split_manifest,observation_schema}.json`, `tests/test_duet_dataset.py`.

- [ ] Freeze separate seed ranges: training 1000–1099, validation 2000–2019, final evaluation 3000–3009. These are initial planned ranges, not a guarantee of sufficient data.
- [ ] Pilot ten training seeds. Review failures before increasing collection; stop wasted rollouts early.
- [ ] Use the working physical controller as a labeled expert to collect synchronized RGB, joint observations, actions, instruction, task stage, timestamps and scene parameters.
- [ ] Record reset state, camera calibration, policy/controller hashes and terminal outcome in every episode manifest.
- [ ] Validate observation/action dimensions, finite values, timestamp ordering, file completeness, split membership and image availability.
- [ ] Retain failed rollouts with labels. Default imitation training uses physically successful episodes; never silently treat failure trajectories as expert success.
- [ ] Check train/validation/test scene overlap beyond seed values; different RNG seeds alone do not guarantee useful physical variation.

Gate: validated dataset with actual counts and variation coverage. Never call generated still images motion-training trajectories. Do not tune on final test episodes; if they are inspected for tuning, retire them and register a new final set. Commit scripts/manifests, not uncontrolled large binary dumps: `add seeded demonstration collection`.

## Stage 3 — improve one learned behavior

Files: actual selected-policy training script, `models/duet_placement_v1/`, `scripts/train_duet_policy.py`, `scripts/evaluate_duet_policy.py`, `tests/test_duet_policy_contract.py`.

- [ ] Select one narrow improvement using baseline failure evidence: preferred target is placement robustness across supported start offsets.
- [ ] Reuse or fine-tune the compatible Talos policy with explicit provenance. Record any inherited weights and data.
- [ ] Match input preprocessing, cameras, timing and action limits to collection exactly. Save configuration, training/validation curves, dataset hash and checkpoint hash.
- [ ] Test actual closed-loop physical rollouts on validation scenes. Low training loss alone does not pass.
- [ ] Compare with the unchanged inherited model on the same validation protocol.
- [ ] Keep the inherited model available; promote the Duet candidate only if evidence supports its claimed advantage without unacceptable regression.

Gate: candidate has physical validation evidence and no hidden teacher fallback. If training does not improve outcomes, report it as an experiment, not an improvement. Commit: `add duet placement policy and validation`.

## Stage 4 — add bounded recovery where evidence supports it

Files: `simulation_lab/duet_recovery/{__init__,state,policy}.py`, executor integration, `tests/test_duet_recovery.py`.

- [ ] Add one failure trigger, such as a missed grasp detected by declared observations.
- [ ] Reobserve, check a supported recovery starting state, retreat if safe, and retry at most twice.
- [ ] Reserve the shared region while either arm occupies it. Cancellation must not mark occupied space as free.
- [ ] Stop explicitly when confidence, reachability, retry count or safety limits fail.
- [ ] Compare enabled/disabled recovery under the same validation perturbations. Record every attempt.

Gate: recovery handles the selected failure without teleporting objects, restarting secretly, or substituting the teacher. Do not invoke a learned skill from a state outside its supported starts. This stage is optional if it threatens evidence time. Commit: `add bounded duet grasp recovery`.

## Stage 5 — original Duet interface

Files: `simulation_lab/web/` and corresponding frontend tests where meaningful.

- [ ] Replace Talos-facing identity with Duet while retaining attribution in documentation/About.
- [ ] Implement an original scene-first layout with distinct arm colors plus labels, a two-arm task timeline, command panel, visible stop control, and evaluation results.
- [ ] Display active controller, actual seed, task status and failure messages accurately. Never show an intended final state as a completed run.
- [ ] Verify a complete command, cancellation, failure/disconnection state, keyboard access, and narrow-screen layout in a browser.

Gate: observed golden path and edge case work without breaking existing controls. Commit: `introduce duet interface and task timeline`.

## Stage 6 — freeze, evaluate, benchmark and record

Files: `scripts/evaluate_duet_suite.py`, `scripts/benchmark_duet_intel.py`, `docs/RESULTS.md`, `docs/HARDWARE.md`, `docs/SUBMISSION.md` and immutable trial artifacts.

- [ ] Freeze candidate, dataset, scene configuration and evaluation manifest before final runs.
- [ ] Execute the complete supported dinner task on seeds 3000–3009, logging every outcome. Demonstrate at least one verified complementary dual-arm action or relay and label it precisely.
- [ ] Run the inherited baseline and Duet under matched conditions where meaningful. Report per-seed success, placement error, collisions, retries, task duration and failures.
- [ ] Export supported models to OpenVINO; test output parity and physical task behavior. Apply INT8 only with separate representative calibration data and successful validation.
- [ ] Run on the user's accepted Intel system. Record CPU/iGPU model, actual OpenVINO device, precision, software versions, warmup, sample count, p50/p95 latency and throughput methodology.
- [ ] Record readable command, starting variation, physical actions and outcome for all ten trials; retain failures and do not replace them silently.
- [ ] Produce reproducible setup/run commands, training/evaluation instructions, architecture summary, attribution and honest limitations.

Gate: every reported measurement links to an artifact. FP32 execution is preferable to unsupported INT8 performance claims. A bench script alone is not hardware evidence. Commit: `record duet evaluation and submission evidence`.

## Deadline policy

Prioritize runnable baseline, physical behavior, meaningful scene variation, and evidence. Reserve at least the final third of available time for evaluation, Intel runs, recording and packaging. Run a small pilot before bulk collection or long training. Exact hours cannot be assigned until cutoff, hardware access and pilot duration are known.

If the candidate underperforms, retain baseline operation and disclose unsuccessful training. If fewer than ten successful seeds are obtained, report the actual count and identify the requirement gap. Cosmetic changes cannot substitute for learned control or benchmark evidence.

## Definition of done

Talos-derived baseline is traceable; original Duet changes are identified; new scene and data are versioned; learned behavior is evaluated independently; ten-seed outcomes and Intel benchmarks are recorded; browser behavior is verified; submission artifacts reproduce the stated results. No completion is inferred from existing README claims.
