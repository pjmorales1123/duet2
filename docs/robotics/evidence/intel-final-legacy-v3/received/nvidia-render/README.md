# Talos selected baseline: Intel laptop verification

The complete six-skill sequence passed on both Intel CPU and Intel UHD inference on exposed seed 42. Both strict hardware-and-physics flags remain false: this is a legacy Intel i7, and MuJoCo used NVIDIA OpenGL. No eligibility checks, physical thresholds, selected models or source files were changed.

Kit: `talos-intel-verification.zip`, 19,526,128 bytes; SHA-256 `6a9b1228b5212efe815d07cbb5aca96e9966e354557afbd42583e986edf22394`. Source revision `57d6ff6185da832d49ae378de393f4a3e7c8cc6a`. All 318 listed files matched before and after execution.

Runtime reused without installation: Python 3.12.14, Torch 2.8.0+cu128, MuJoCo 3.12.0, OpenVINO 2026.3.0-22451-8a17657b995-releases/2026/3. Python executable: `<OLD_CHECKOUT>\.venv-training\Scripts\python.exe`.

CPU: Intel(R) Core(TM) i7-10850H CPU @ 2.70GHz; Core Ultra Series 2/3: **no**. OpenGL: `Quadro T2000 with Max-Q Design/PCIe/SSE2`, vendor `NVIDIA Corporation`, version `4.6.0 NVIDIA 581.95`.

OpenVINO discovery: `CPU` = Intel i7-10850H; `GPU.0` = Intel UHD Graphics (iGPU); `GPU.1` = Quadro T2000 (dGPU). Physical policy reports confirm CPU or GPU.0 respectively for every skill. GPU.1 benchmark entries are retained but are not Intel evidence.

| Run | Physical sequence | Simulation seconds | Physical-loop wall seconds | Entire invocation wall seconds | Exit | Intel inference | Strict hardware + physics |
|---|---|---:|---:|---:|---:|---|---|
| CPU | succeeded (6/6) | 247.655 | 43.545 | 83.153 | 1 | True | False |
| GPU wrapper | Not executed | — | — | 15.549 | 1 | Not established by this attempt | No final flag/report |
| Intel GPU.0 | succeeded (6/6) | 247.655 | 55.255 | 92.567 | 1 | True | False |

Physical-loop timing excludes initial scene/first-task setup, network export and final trace compression; entire invocation includes preflight/export/benchmarking and report writing. Simulation time is separate from elapsed real time. This is a headless check, not a browser performance measurement.

| Skill | CPU result | CPU sim s | GPU.0 result | GPU.0 sim s | CPU / GPU XY error mm, or drawer opening mm |
|---|---|---:|---|---:|---|
| bottle | succeeded | 38.355 | succeeded | 38.355 | 1.115 / 1.115 |
| plate | succeeded | 37.365 | succeeded | 37.365 | 1.112 / 1.112 |
| mug | succeeded | 41.155 | succeeded | 41.155 | 1.107 / 1.107 |
| drawer | succeeded | 29.665 | succeeded | 29.665 | 117.768 / 117.768 |
| fork | succeeded | 48.550 | succeeded | 48.550 | 0.666 / 0.666 |
| spoon | succeeded | 52.565 | succeeded | 52.565 | 4.406 / 3.650 |

Every skill reports both arms parked and stable physical release. Drawer opening uses handle contact; object skills report verified supported holds. No unexpected collisions, hidden forces, equality attachments or controller state writes were reported. See original per-skill metrics for lift, support duration, settling and disturbance checks.

| Network | CPU parity max rad | GPU.0 parity max rad | CPU median / p95 ms | GPU.0 median / p95 ms | CPU / GPU.0 synchronous chunks per second |
|---|---:|---:|---:|---:|---:|
| bottle | 2.37e-05 | 4.8e-06 | 0.123 / 0.208 | 0.594 / 1.928 | 7418.5 / 1324.9 |
| plate | 3.34e-06 | 3.82e-06 | 0.125 / 0.208 | 0.622 / 1.737 | 7303.0 / 1333.8 |
| mug | 3.74e-06 | 2.77e-06 | 0.123 / 0.220 | 0.617 / 2.216 | 7182.6 / 1289.5 |
| drawer | 3.75e-06 | 3.03e-06 | 0.121 / 0.233 | 0.648 / 1.389 | 7195.1 / 1361.4 |
| fork | 3.66e-06 | 8.61e-06 | 0.163 / 0.248 | 0.669 / 1.369 | 5754.3 / 1282.6 |
| spoon | 3.71e-06 | 3.1e-06 | 0.125 / 0.225 | 0.670 / 1.383 | 6879.0 / 1291.5 |

All six FP32 exports passed the unchanged <1e-4 rad joint-error parity threshold on CPU and Intel GPU.0. Each device uses 32 synthetic parity batches, 20 warmups and 300 synchronous timed calls. Each call emits 20 endpoints. These are network-only timings/throughput, excluding visual encoding, feedback, rendering and physics; they are not robot control frequency or full-system throughput.

The requested PowerShell `-Device GPU` attempt failed after the bottle export because the verifier matches device strings exactly: only GPU.0/GPU.1 were listed. Its message says device/parity failed, but its retained benchmark shows successful GPU.0 numerical parity. No physical run or verification.json was produced. After that process finished, the unchanged Python verifier ran with explicit `--device GPU.0` in another unused folder. CPU and GPU.0 final exit code 1 reflects the strict hardware flag, not physical failure.

This checks the packaged selected dinner baseline on one exposed seed, not a new frozen success-rate evaluation. The previously reported 8/10 dinner sequences, 5/5 relay results per direction, integrated 8/10 workflow, hosted demo and saved submission draft were not re-evaluated here. Relay models are present and manifest-validated, but this verifier executes the six dinner skills only. Unpromoted visual-correction experiments were not selected.

No repository operation, upload, dependency installation, microphone use, API-key access or Windows graphics-setting change was performed. The older checkout and its environments were reused only for the existing Python executable. Legacy-Intel eligibility still requires organizer confirmation. Intel GPU inference does not establish Intel graphics rendering.

To attempt Intel rendering later, the user can select the actual Python executable in Windows Settings → System → Display → Graphics, choose the option explicitly identifying Intel UHD (often Power saving), and save. The base executable is `<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`; the venv launcher is `<OLD_CHECKOUT>\.venv-training\Scripts\python.exe`. This can affect other apps sharing that Python runtime. Restart the verification process and inspect the measured renderer; preference selection is not proof it took effect. Do not disable devices. This cannot turn an i7 into Core Ultra. [Microsoft guidance](https://support.microsoft.com/en-us/windows/experience/power-battery/battery-saving-tips-for-windows).

Original evidence paths (preserved unchanged):

- Extracted kit: `.`
- Preflight and manifest validation: `.\.run\session\preflight.json`
- `.\.run\intel-final-laptop-cpu`: hardware report, per-network benchmark/configuration files, verification.json, physical.json and physical-recording/{report.json,scene.xml,states.npz}.
- Console: `.\.run\session\intel-final-laptop-cpu.console.log`
- Invocation/exit/timing: `.\.run\session\intel-final-laptop-cpu.console.run.json`
- `.\.run\intel-final-laptop-gpu`: hardware report, per-network benchmark/configuration files; no final physical/verification report.
- Console: `.\.run\session\intel-final-laptop-gpu.console.log`
- Invocation/exit/timing: `.\.run\session\intel-final-laptop-gpu.console.run.json`
- `.\.run\intel-final-laptop-gpu0`: hardware report, per-network benchmark/configuration files, verification.json, physical.json and physical-recording/{report.json,scene.xml,states.npz}.
- Console: `.\.run\session\intel-final-laptop-gpu0.console.log`
- Invocation/exit/timing: `.\.run\session\intel-final-laptop-gpu0.console.run.json`

Shareable archive: `<WORKSPACE_PARENT>\Talos_Intel_Evidence_2026-09-14.zip`. It contains sanitized report/log copies and all mesh files required by the retained scene XMLs, preserving relative scene paths. It omits model weights (available in the hashed kit), environments, credentials and unrelated data. Physical state traces and meshes are byte-identical copies. See evidence-manifest.json for original/copy hashes and the sanitization audit.
