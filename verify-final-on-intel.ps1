param(
    [string]$Python = ".venv-training\Scripts\python.exe",
    [ValidateSet("CPU", "GPU")][string]$Device = "CPU",
    [string]$Output = "",
    [switch]$Diagnostic
)

$ErrorActionPreference = "Stop"
$taskRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = ".run/intel-final-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
$pythonPath = if ([System.IO.Path]::IsPathRooted($Python)) { $Python } else { Join-Path $taskRoot $Python }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python runtime not found. Install the documented training/OpenVINO environment or pass -Python with the existing Python 3.12 executable."
}
$pythonPath = (Resolve-Path -LiteralPath $pythonPath).Path
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $taskRoot $Output }
if (Test-Path -LiteralPath $outputPath) { throw "The output path already exists. Choose a fresh path; earlier evidence is retained." }

Push-Location -LiteralPath $taskRoot
try {
    & $pythonPath -c "import sys; import mujoco, openvino, torch; from pathlib import Path; from simulation_lab.storage import require_space; assert sys.version_info[:2] == (3, 12), 'Use the documented Python 3.12 runtime'; require_space(Path(sys.argv[1]), 512*1024**2)" $Output
    if ($LASTEXITCODE -ne 0) { throw "Environment or disk preflight failed. No verification output was created." }
    $arguments = @("scripts/verify_intel_submission.py", "--suite", "models/dinner_suite/suite.json", "--output", $Output, "--device", $Device)
    if ($Diagnostic) { $arguments += "--diagnostic" }
    & $pythonPath @arguments
    $verificationExit = $LASTEXITCODE
    $reportPath = Join-Path $outputPath "verification.json"
    if (Test-Path -LiteralPath $reportPath) {
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
        [pscustomobject]@{
            CPU = $report.hardware.cpu
            Renderer = $report.hardware.opengl.renderer
            InferenceDevice = $Device
            InferenceDevicesAreIntel = $report.inference_devices_intel
            PhysicalSequencePassed = $report.physical_passed
            StrictHardwareAndPhysicsPassed = $report.strict_hardware_and_physics_passed
            EvidenceFolder = $Output
        } | Format-List
        Write-Output "Physical execution and hardware eligibility are separate results. This script does not upload or submit anything."
    }
    if ($verificationExit -ne 0) { throw "Verification returned $verificationExit. Inspect retained evidence for physical and hardware results." }
}
finally {
    Pop-Location
}
