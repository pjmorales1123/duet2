# Duet 2 follow-up assignments

Read AGENTS.md, docs/DUET_IMPLEMENTATION_PLAN.md, and docs/DUET_PROGRESS.md before working. Do not overwrite existing artifacts. Record exact commands and results in the progress tracker.

## Assignment A - Original interface and presentation

Scope: simulation_lab/web/ only, plus focused documentation.

- Complete the Duet 2 visual identity with a new scene-first layout.
- Add a clear two-arm task timeline, active arm/status labels, seed display, stop control, and explicit trial result panel.
- Preserve existing API contracts and all controls.
- Test a complete task path, cancellation, and a narrow viewport in a browser.

Do not change physics, policy code, model files, or dataset split rules.

## Assignment B - Demonstration collection and validation

Scope: scripts/collect_duet_demos.py, new dataset-validation code, and training/duet_v1/.

- Collect a small pilot batch from training seeds 1000-1009 only.
- Validate file completeness, finite observations/actions, timestamps, scene manifests, and physical replay eligibility.
- Report counts of successes and retained failures. Do not train on failures unless a separate recovery label is defined.
- Never use validation or final-evaluation seeds for data collection.

Do not change scene geometry or the final-evaluation manifest.

## Assignment C - Candidate policy and evaluation

Scope: new Duet training/evaluation scripts and versioned model directories.

- Select one placement-robustness improvement using the pilot data.
- Train or fine-tune a clearly attributed candidate.
- Evaluate on validation seeds 2000-2019 with closed-loop physical execution.
- Compare to the inherited baseline under the same protocol.
- Export only a candidate that shows actual measured behavior; retain failed experiments and document them.

Do not inspect or tune on final evaluation seeds 3000-3009.
