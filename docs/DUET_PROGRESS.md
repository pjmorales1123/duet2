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
| M1 | Original Duet assets and seeded scene | Not started | M0 | Scene tests, parameter manifests, visual check, physical rollout |
| M2 | Training dataset and frozen splits | Not started | M1 | Validated episodes, coverage and provenance report |
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
