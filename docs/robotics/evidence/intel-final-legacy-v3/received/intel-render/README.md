# Talos Intel rendering verification

Windows per-app graphics preference for the actual base Python executable was changed from Windows decides to Power saving (Intel UHD Graphics), through Windows Settings. A fresh MuJoCo renderer probe and both fresh verification processes measure Intel OpenGL. No GPU was disabled. The venv launcher preference was not changed. To restore the prior behavior, set this same Python entry to Windows decides (or remove its newly added entry). This preference also applies to other processes using that shared base Python runtime.

Base Python: `<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`. Existing venv launcher: `<OLD_CHECKOUT>\.venv-training\Scripts\python.exe`. No dependencies or environment files were changed.

Kit SHA-256: `6a9b1228b5212efe815d07cbb5aca96e9966e354557afbd42583e986edf22394` (19,526,128 bytes). Source revision: `57d6ff6185da832d49ae378de393f4a3e7c8cc6a`. All 318 manifest-listed files still match.

Runtime: Python 3.12.14, Torch 2.8.0+cu128, MuJoCo 3.12.0, OpenVINO 2026.3.0-22451-8a17657b995-releases/2026/3.

CPU: Intel Core i7-10850H; **not Core Ultra Series 2/3**. OpenVINO GPU.0 identifies Intel UHD Graphics; GPU.1 identifies NVIDIA Quadro T2000. Export benchmarks include all devices, but physical inference is explicitly CPU or GPU.0.

| Run | OpenGL renderer | Six-skill sequence | Intel renderer flag | Intel inference flag | Strict hardware + physics | Simulation s | Physical-loop wall s | Whole invocation wall s | Exit |
|---|---|---|---|---|---|---:|---:|---:|---:|
| intel-render-laptop-cpu | Intel(R) UHD Graphics | succeeded (6/6) | True | True | False | 247.655 | 266.682 | 307.842 | 1 |
| intel-render-laptop-gpu0 | Intel(R) UHD Graphics | succeeded (6/6) | True | True | False | 247.655 | 193.214 | 226.345 | 1 |

Intel OpenGL version: `4.6.0 - Build 31.0.101.2137`. The unchanged strict flag remains false because this is a legacy i7. Exit code 1 does not imply physical failure; read verification.json. Legacy hardware eligibility still needs organizer confirmation.

| Skill | CPU outcome | Intel GPU.0 outcome | CPU / GPU.0 placement error mm (drawer: opening mm) | Both arms parked CPU / GPU.0 |
|---|---|---|---:|---|
| bottle | succeeded | succeeded | 1.115 / 1.115 | True / True |
| plate | succeeded | succeeded | 1.112 / 1.112 | True / True |
| mug | succeeded | succeeded | 1.107 / 1.107 | True / True |
| drawer | succeeded | succeeded | 117.764 / 117.764 | True / True |
| fork | succeeded | succeeded | 0.651 / 0.651 | True / True |
| spoon | succeeded | succeeded | 4.493 / 4.493 | True / True |

| Skill network | CPU parity max rad | GPU.0 parity max rad | CPU median / p95 ms | GPU.0 median / p95 ms | CPU / GPU.0 synchronous chunks/s |
|---|---:|---:|---:|---:|---:|
| bottle | 2.37e-05 | 4.8e-06 | 0.135 / 0.193 | 0.608 / 4.301 | 6914.8 / 1152.9 |
| plate | 3.34e-06 | 3.82e-06 | 0.152 / 0.252 | 0.603 / 4.528 | 6005.6 / 1121.5 |
| mug | 3.74e-06 | 2.77e-06 | 0.124 / 0.227 | 0.614 / 1.295 | 7107.6 / 1196.4 |
| drawer | 3.75e-06 | 3.03e-06 | 0.145 / 0.201 | 0.644 / 4.270 | 6503.0 / 1122.6 |
| fork | 3.66e-06 | 8.61e-06 | 0.129 / 0.218 | 0.622 / 1.125 | 6977.6 / 1178.2 |
| spoon | 3.71e-06 | 3.1e-06 | 0.145 / 0.180 | 0.651 / 4.441 | 6624.7 / 1072.3 |

Parity uses the original <1e-4 rad threshold, 32 synthetic batches, 20 warmups and 300 timed calls. Each call emits 20 endpoints. Network timings exclude rendering, feedback and physics; chunks/s is not robot control rate. Physical-loop wall time excludes initial setup/export and final compression; whole invocation includes them. This is a headless physical verification, not a browser rendering performance benchmark.

GPU.0 used the unchanged Python verifier directly because the packaged PowerShell wrapper only allows CPU/GPU and the installed runtime exposes GPU.0/GPU.1. This avoids the previously documented GPU alias lookup failure without changing verification code or thresholds.

Scope: one exposed seed 42, six selected dinner skills. This does not repeat the frozen 8/10 evaluation or relay evaluation, and does not demonstrate arbitrary placement generalization. No training, selected-model changes, source changes, repository operations, uploads, microphone or API-key access occurred. Earlier NVIDIA-rendered evidence and all failed attempts remain retained.

Original evidence paths:

- Renderer probe: `.\.run\session\intel-renderer-probe-01.json`
- Reports and physical trace: `.\.run\intel-render-laptop-cpu` (verification.json, hardware.json, physical.json, per-skill benchmarks, physical-recording/scene.xml, states.npz and report.json).
- Console: `.\.run\session\intel-render-laptop-cpu.console.log`
- Invocation receipt: `.\.run\session\intel-render-laptop-cpu.console.run.json`
- Reports and physical trace: `.\.run\intel-render-laptop-gpu0` (verification.json, hardware.json, physical.json, per-skill benchmarks, physical-recording/scene.xml, states.npz and report.json).
- Console: `.\.run\session\intel-render-laptop-gpu0.console.log`
- Invocation receipt: `.\.run\session\intel-render-laptop-gpu0.console.run.json`

New shareable ZIP: `<WORKSPACE_PARENT>\Talos_Intel_Render_Evidence_2026-09-14.zip`. Copies redact identifying paths; original reports remain unchanged. Required scene meshes and numeric physical traces are included. Environments, model weights, credentials and unrelated private files are excluded. See evidence-manifest.json for hashes and privacy checks.
