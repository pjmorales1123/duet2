# Joint-limited motor map: offline gates pass

September 14, 2026. [V2](protocol.json) retains every sampled configuration from the [failed V1 input gate](../mug-correction-motor-v1/README.md), scaling movement according to required joint motion. Every training/development label passes with the tighter 0.02 rad joint bound; no points are dropped.

| Check | Accepted / attempted | Endpoint p95 / maximum |
| --- | --- | --- |
| Step 3,000 development | 1,024 / 1,024 | 0.073 / 0.158 mm |
| Step 6,000 development, selected | 1,024 / 1,024 | 0.056 / 0.136 mm |
| Fresh OpenVINO CPU | 1,024 / 1,024 | 0.061 / 0.131 mm |

Unchanged accuracy gates require ≥99% accepted, ≤0.1 mm p95 and ≤0.3 mm maximum. Both candidates and all rows are retained. The one fit completes in 18.160 seconds on RTX 4070. CPU export parity is below 1.4×10⁻⁷ rad across all development/fresh commands, against 10⁻⁶. All three independent geometry audits pass.

Errors are relative to each **effective scaled translation**. Not every requested 2 mm displacement executes in one step. The fresh minimum step fraction is 0.0407, median 1.0; every factor is recorded. Repeated physical convergence and task success remain untested.

The [packaged callable interface](../../../../models/mug_correction_motor_v2/README.md) exactly reproduces all 1,024 fresh joint outputs from [compact inputs](../../../../training/mug_correction_motor_v2/README.md), without raw `.run` data. Subsequent verification checks 101 core manifest files. Core manifests remain unchanged; completion reports/logs have a separate manifest.

Two launcher failures are retained: a relative protocol path was rejected before initial data creation, and the fresh collector hit a variable/helper naming collision before sampling. Path resolution was fixed before fitting; a separately frozen standalone collector fixes fresh dispatch while preserving fitted-source bytes. Neither incident changed weights, generated data, gates or seeds. Original logs remain local; packaged logs redact personal filesystem prefixes and record original hashes.

**No promotion or new physical trial:** selected dinner models remain unchanged. Next is a separately frozen live-versus-frozen-image physical mug-placement experiment with the stereo observer, motor feedback, explicit timing/limits and original physical monitors. Intel execution, voice rehearsal and public judge access remain separate open requirements.
