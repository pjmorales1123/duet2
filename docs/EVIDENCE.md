# Measured results and limitations

These are finite physical simulation tests. They do not prove success from every reachable position. The public demo uses the promoted late-mug visual profile; most movements still use initial camera features and joint-feedback progress guards. Exact simulator state is restricted to the separate safety/scoring monitor in learned execution.

| Claim | Evidence | Interpretation |
| --- | --- | --- |
| Original learned dinner: 8/10 | [All-ten record](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/submission/final-v6/Talos-dinner-ten-v6.json) | Two failed final placements remain in the denominator. |
| Bottle relay: 5/5 per direction | [All-ten record](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/submission/final-v6/Talos-relays-ten-v3.json) | Release onto the table, then regrasp; no airborne exchange. |
| Composed relay + dinner: 8/10 | [Summary](robotics/evidence/composed-dinner-v1/original-final-summary.json), [complete traces](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/docs/robotics/evidence/composed-dinner-v1) | Separate frozen starting distribution. |
| Live mug comparison: 20/20 live, 20/20 original baseline, 0/20 frozen images | [Final audit](robotics/evidence/mug-visual-correction-v2/evaluation-audit.json), [protocol](robotics/evidence/mug-visual-correction-v2/protocol.json), [all 96 development/final outcomes and traces](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/docs/robotics/evidence/mug-visual-correction-v2) | Current images matter to this correction controller; the baseline also passes all final scenes. |
| Human “Set the table” rehearsal | [Recorded outcome](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/docs/robotics/evidence/human-voice-dinner-v1) | One exposed microphone-triggered six-skill run. |
| Hosted CPU six-skill run: 287.79 seconds | [Original hosted evidence](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/docs/robotics/evidence/human-voice-dinner-v1) | Measured wall time on the hosted worker, not a universal latency promise. |
| Legacy Intel six-skill inference/rendering | [Audit](robotics/evidence/intel-final-legacy-v3/audit.json), [complete device/trace evidence](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/docs/robotics/evidence/intel-final-legacy-v3) | Intel i7-10850H, CPU/iGPU inference and UHD rendering. Qualifying Core Ultra Series 2/3 execution remains unresolved; the latest live-mug profile needs separate Intel verification. |

Original bottle results, including the older failed wider tests, remain in the [baseline evidence](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14/submission/evidence). The [full development archive](ARCHIVE.md) preserves every stopped experiment, model and published trace. JSON summaries kept here retain their original hashes; their trace paths resolve in that archive.

## Unfinished work

- No complete jointly randomized seven-item dinner scene has passed. Separate plate, bottle, glass, spoon and regrasp successes cannot be combined into a claimed end-to-end success.
- Wider-bottle placement was developed separately; the other objects retain narrow starting distributions. The demonstrated live-mug workflow is not a general tabletop policy.
- Continuous visual correction is limited to late mug placement in the promoted profile. General moving-object grasp correction and visual recovery remain unfinished.
- Supported language follows a limited grammar. Arbitrary goals, pouring, direct hand-to-hand exchanges and general object handoffs are not implemented.
- Adapted CLIPort heads and a pretrained CLIP representation ran within the RTX 4070 memory budget with finite gradients. No shared Talos spatial policy was trained. That unpromoted research is in the archive and is not the demo controller.

The final 4:49.5 video combines the user's real local recordings and narration. Accelerated footage and a removed desktop interval are labeled. Both local recorded workflows finish successfully. The hosted excerpt shows a new trial in progress; it does not claim to show that trial's completion.

## Final release checks

The cleaned checkout passes 69 application and 20 model/geometry tests. The documented CPU environment also completes both exposed production-entry workflows. A new anonymous hosted CPU trial, with no authorization headers or cookies, completes all six skills and parks both arms in 289.54 seconds end to end. [Release checks](release/checks.json), [anonymous trial and per-skill physical metrics](release/anonymous-demo.json), [persisted media hashes](release/submission-media.json). These are release regressions on exposed seed 42, not new coverage claims.
