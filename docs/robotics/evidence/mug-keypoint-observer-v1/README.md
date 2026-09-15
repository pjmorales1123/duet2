# Mug image-point observer: bounded perception passes

September 14, 2026. [Protocol](protocol.json) SHA-256 `e73c7058c662e3e200fc7264dea2c90e0fc1241a281a195d549cbddbd2442d4e` was frozen before data generation. Source/data/fit/evaluation revisions are recorded in the original reports. This experiment introduces no new physical controller trials and does not replace the selected dinner models.

| Check | Coverage / absent false accepts | Position p95 / maximum | Result |
| --- | --- | --- | --- |
| Step 4,000 development | 448/448 / 0 of 64 | 0.385 / 0.709 mm | Pass; retained |
| Step 8,000 development | 448/448 / 0 of 64 | 0.265 / 0.504 mm | Pass; selected |
| Reserved fresh synthetic CPU | 448/448 / 0 of 64 | 0.270 / 0.422 mm | Pass |
| All exposed physical states | 144/144 / no absent states | 0.617 / 0.893 mm | Pass |

All rows, refusals, errors, source/input hashes and timings are retained in the adjacent JSON files. Gates remain at ≥95% present coverage, zero absent false accepts, ≤1.5 mm p95 XYZ/XY and ≤3 mm maximum XYZ error. Export parity across all 512 development states has maximum keypoint difference 0.000687 pixels, below 0.01. The OpenVINO FP32 measurements are from AMD CPU, not Intel hardware.

The fresh split is newly sampled posed perception data (RNG 2026099853); the physical regression reuses all 144 previously exposed mug placement states from the [visibility diagnostic](../mug-visual-visibility-v1/README.md). It scores the original RGB mosaic crops before accessing body state for ground-truth comparison. No fitting, exclusion or new physics occurs in this regression. The earlier failed direct-XYZ observer used a different development split; these results do not establish a controlled architectural comparison.

All three data audits pass. The training RGB is byte-identical to the preserved original input; all new development/fresh images independently re-render exactly. An additional packaged reproduction regenerates all 1,024 fresh RGB views and reproduces every selected prediction exactly, without reading raw `.run` artifacts. This repeats exposed observations and is labeled accordingly.

One first read-only training audit was stopped because repeated NPZ access redundantly decompressed whole shards. Its original source, incident record and empty console log remain under `audit-attempts/attempt-1` / `console`. Materializing each shard once fixed the audit's performance; neither data, model nor gate changed. The packager's own console was copied while it was still empty; its final closed output is retained separately in `completion/`. That empty snapshot is not a completed-run log.

The package preserves [both models and evaluated IR](../../../../models/mug_keypoint_observer_v1/README.md), [all exact compact input recipes](../../../../training/mug_keypoint_observer_v1/README.md), source snapshots, all results and the stopped audit attempt. Core manifests preserve original bytes; the completion supplement has its own manifest. All 41 original protected submission artifacts remain unchanged.

**Still open:** a separately declared motor model and live-versus-frozen-image physical placement test. Position accuracy in bounded images does not establish successful corrective movement, reliable grasping, arbitrary object placement, or broader dinner success. No hosted baseline changes follow from this perception result alone.
