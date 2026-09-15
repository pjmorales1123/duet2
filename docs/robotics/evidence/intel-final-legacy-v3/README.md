# Legacy Intel laptop verification

The received September 14 evidence verifies the selected six-skill baseline on an **Intel Core i7-10850H**. Both explicit OpenVINO devices complete exposed seed 42 with real simulated contact, release and parked arms.

| OpenGL renderer | Inference | Six-skill result | Simulated seconds | Physics-loop wall seconds |
| --- | --- | --- | --- | --- |
| NVIDIA Quadro | CPU | 6/6 | 247.655 | 43.545 |
| NVIDIA Quadro | GPU.0 | 6/6 | 247.655 | 55.255 |
| Intel UHD | CPU | 6/6 | 247.655 | 266.682 |
| Intel UHD | GPU.0 | 6/6 | 247.655 | 193.214 |

**Hardware eligibility remains unresolved.** The processor is not Core Ultra Series 2/3. The later two runs verify Intel UHD OpenGL rendering and Intel CPU/GPU.0 inference together. Earlier NVIDIA Quadro T2000 Max-Q rendering remains separately recorded. Every strict Core Ultra hardware flag remains false.

The laptop operator selected Power saving / Intel UHD for the actual shared base Python executable in Windows graphics preferences. No GPU was disabled and no environment, source or model files changed. The received summary records how to restore Windows decides. This preference affects other programs using that same base runtime.

Each network benchmark uses FP32, 20 warmups and 300 measured synchronous calls; one output contains 20 action endpoints. These small-network timings exclude vision, rendering and physics. Read each run separately: changing the renderer also changes full-workflow wall time.

The earlier generic `GPU` attempt is also retained. Its bottle parity checks passed on CPU, GPU.0 and GPU.1, but the requested alias matched none of the exact report names. The verifier stopped before full physics with a misleading parity message. The later explicit `GPU.0` run passed.

All 169 transport-manifest files match; the original 318-file kit manifest and six source checkpoint hashes match. All 4 complete 4,955-frame traces are unchanged. Portable scenes load, every trace array is checked, and restored completion geometry agrees with the original reports. Continuous contact/hold evidence comes from the retained original runtime monitor; the 20 Hz saved trace cannot independently reconstruct every 200 Hz contact.

Every supplied attempt, console, report and transport provenance is under `received/`; its `.run` directory is mapped to `runs` for publication. `portable/` contains unchanged trace copies, original XML and scenes whose only change is the mesh path. The original received ZIP remains local. Re-exported laptop IR files were not supplied; benchmark reports are preserved without claiming that they were rerun here.

This kit predates the new late-mug live-vision option. Its results apply to the original six-skill suite. A separate target-machine check is needed for the new option. GitHub/HF remain private until final release; final Submit belongs to the user.
