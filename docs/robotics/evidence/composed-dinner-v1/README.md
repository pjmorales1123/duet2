# Camera-planned relay plus dinner sequence

The actual language-command engine selects a two-arm bottle relay from RGB images, then chains plate, mug, drawer, fork and spoon skills. All four development scenes pass. The separately frozen ten-scene evaluation passes **8/10 full workflows**. Both failures are final spoon placement errors; all ten complete the other six learned steps, including both relay legs.

| Frozen seed | Full workflow | Completed learned steps | Failure |
| --- | --- | --- | --- |
| 2026092801 | Pass | 7/7 | — |
| 2026092802 | Fail | 6/7 | Spoon: 10.094 mm placement error |
| 2026092803 | Pass | 7/7 | — |
| 2026092804 | Pass | 7/7 | — |
| 2026092805 | Pass | 7/7 | — |
| 2026092806 | Pass | 7/7 | — |
| 2026092807 | Pass | 7/7 | — |
| 2026092808 | Pass | 7/7 | — |
| 2026092809 | Fail | 6/7 | Spoon: 14.254 mm placement error |
| 2026092810 | Pass | 7/7 | — |

The unchanged limit is 8 mm, with verified physical release, parked arms, collision checks and preservation of prior placements. All failed trials are retained. Passing scenes finish in about 283.3 simulated seconds; the two failures stop around 319.5 seconds. Measured wall times on this AMD/RTX PC are 55.61–64.80 seconds, with up to three isolated trials running concurrently. These are not Intel or interactive-browser timing measurements.

## What this verifies

The evaluator calls `LabEngine._reset` and `LabEngine._language_command` with `set the table` and `mode=learned_dinner` in an isolated engine. It supplies no skill list. The production parser and RGB scene observer select:

1. `reverse_bottle_right`: first arm places the bottle on shared table support.
2. `reverse_bottle_left`: second arm finishes bottle placement.
3. Plate, mug, drawer, fork and spoon, with a fresh initial visual observation for each neural skill.

All steps run without a reset. A separate simulator-state monitor scores contacts, releases, placement and parking; it does not generate learned targets. Every controller update is checked for physical-state writes and applied forces. Runtime snapshots confirm OpenVINO CPU inference, no teacher updates and no retrieval of demonstration actions.

This closes the composition test for the existing finite left-reach preset. The bottle starts at x = -0.0734068 ±0.006 m and y = -0.1215734 ±0.006 m, with yaw 0.309641 ±0.08 rad. Existing dinner randomization varies other object starts, masses, friction and lighting within its documented small ranges. It is not a test of arbitrary reachable positions, new shapes, unrestricted language or continuous image correction. The test executes the browser engine path directly; it is not a new human microphone rehearsal or a hosted-cloud trial.

## Evidence and reproduction

- [Audit and all outcomes](audit.json), [frozen selection](frozen-selection.json) and [byte-level package manifest](package-manifest.json).
- Four development and ten evaluation reports, all original synchronized state arrays, and portable scene XMLs. Only mesh-directory references change in the XMLs.
- Forty-six evaluated source files, the batch driver, and hashes of all 70 model/configuration artifacts. No weights or model metadata were changed.
- [All-ten-trial video](../../../../submission/final-v6/Talos-composed-dinner-ten-v1.mp4): 1280×720, 82.9 seconds, 4× labeled playback. All 829 frames decode; eight sampled frames and the final outcome display were visually checked. No failed trial is omitted.

In the full local browser, select **Learned dinner sequence**, **Task start**, **Closed**, and **Left reach practice**, then load a seed and enter `set the table`. These evaluation seeds are now exposed reproductions, not further holdouts.

```powershell
.venv-training/Scripts/python scripts/evaluate_composed_dinner.py --seed 2026092801 --split evaluation --freeze docs/robotics/experiments/composed-dinner-final-v1.json --output .run/composed-dinner-v1-reproduction-2801
```

The evaluator refuses a changed frozen input or an existing output folder. Every dataset/export checks disk, retaining 10 GiB plus expected writes. Raw and packaged state evidence total approximately 133 MiB, below the separate 256 MiB protocol budget. Original models and all previous evidence remain unchanged. Any revised controller needs new development/evaluation separation; these final failures must not be used for hidden tuning of the reported result.

September 14 deployment regression: the private hosted CPU demo also completes the seven-step workflow on exposed seed 42, in **283.31 simulated / 297.31 wall seconds**. Every step releases and parks with zero unexpected collisions, state writes or hidden forces. [Selected browser-report fields](cloud-cpu-seed42.json) preserve this separate deployment observation; it is not another frozen trial, anonymous-access verification or an Intel run. Its 2.69-second margin under the former 300-second hosting limit motivates a 420-second hosting allowance; the controller and physical success limits are unchanged.
