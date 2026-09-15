# Talos Intel verification kit

This kit contains the exact selected dinner/relay models, tracked source, robot
assets and MIT/Apache notices. It requires the existing clean Python 3.12 training
environment with MuJoCo, PyTorch and OpenVINO. It contains no environment, API key,
microphone recording or training dataset. No installation or upload occurs.

Extract into a NEW folder. From that folder in PowerShell, use your laptop's
existing Python executable:

```powershell
$talosPython = Read-Host "Full path to the existing .venv-training\Scripts\python.exe"
& .\verify-final-on-intel.ps1 -Python $talosPython -Output ".run/intel-final-laptop-cpu"
```

The wrapper checks Python/dependencies and free space, then benchmarks actual
OpenVINO devices and runs all six learned skills on exposed seed 42. Keep the
output directory unused. It retains a 10 GiB reserve plus estimated writes and
does not overwrite models or previous evidence.

On the historical i7-10850H laptop the strict Core Ultra check remains false even
if physics succeeds. Read both PhysicalSequencePassed and
StrictHardwareAndPhysicsPassed. No switch can change the measured CPU/graphics
identity or establish organizer eligibility.

Retain the entire .run/intel-final-laptop-cpu folder, especially verification.json,
hardware.json, physical.json, all benchmark.json files and physical-recording/.
Share the verification results with the Talos task for review. No Speechmatics key
or microphone is needed. Do not put credentials in this kit or send them in chat.

kit-manifest.json records every original source/model/asset hash and the source
Git revision. The separate kit audit records a PC portability check; AMD/NVIDIA
execution does not substitute for actual Intel execution.

The original project and this archive remain private until the final release.
Only the user presses the hackathon's final Submit button.
