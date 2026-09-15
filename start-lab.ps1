param([switch]$NoBrowser, [switch]$Learned, [int]$Port = 8765, [string]$DinnerSuite)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if ($DinnerSuite -and -not $Learned) { throw 'DinnerSuite requires -Learned.' }
if ($DinnerSuite) { $DinnerSuite = (Resolve-Path -LiteralPath $DinnerSuite).Path }
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (($Learned -or -not (Test-Path -LiteralPath $taskPython)) -and (Test-Path -LiteralPath (Join-Path $taskRoot '.venv-training\Scripts\python.exe'))) {
    $taskPython = Join-Path $taskRoot '.venv-training\Scripts\python.exe'
}
$taskRunDir = Join-Path $taskRoot '.run'
$taskUrl = "http://127.0.0.1:$Port"
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'The local Python environment is missing. Follow simulation_lab/README.md to install it.'
}
New-Item -ItemType Directory -Path $taskRunDir -Force | Out-Null
$taskAlreadyRunning = $false
try {
    $taskState = Invoke-RestMethod -Uri "$taskUrl/api/state" -TimeoutSec 2
    $taskAlreadyRunning = $taskState.ready -and $taskState.controller -in @('manual_joint_targets', 'ground_truth_ik', 'learned_bottle', 'learned_dinner')
} catch { }
if ($taskAlreadyRunning -and $Learned) {
    $taskExistingPath = Join-Path $taskRunDir "server-$Port.json"
    $taskExistingLearned = $false
    if (Test-Path -LiteralPath $taskExistingPath) {
        $taskExistingRecord = Get-Content -LiteralPath $taskExistingPath | ConvertFrom-Json
        $taskExistingLearned = $taskExistingRecord.learned -eq $true
        if ($DinnerSuite -and $taskExistingRecord.dinner_suite -ne $DinnerSuite) {
            throw 'This port is running a different dinner suite. Stop it first or use another port.'
        }
    }
    if (-not $taskExistingLearned) {
        throw "A simulator already uses port $Port. Stop it first or choose another port to start the learned runtime."
    }
}
if (-not $taskAlreadyRunning) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is already used by another application. Run this script with -Port followed by another port."
    }
    $taskArguments = @('-m', 'simulation_lab.server', '--port', $Port)
    if ($Learned) { $taskArguments += '--learned' }
    if ($DinnerSuite) { $taskArguments += @('--dinner-suite', ('"' + $DinnerSuite + '"')) }
    $taskLogStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
    $taskStdout = Join-Path $taskRunDir "server-$Port-$taskLogStamp.stdout.log"
    $taskStderr = Join-Path $taskRunDir "server-$Port-$taskLogStamp.stderr.log"
    if ((Test-Path -LiteralPath $taskStdout) -or (Test-Path -LiteralPath $taskStderr)) {
        throw 'The new server log paths already exist. Earlier logs are preserved.'
    }
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
    @{pid=$taskProcess.Id; port=$Port; python=$taskPython; learned=[bool]$Learned; dinner_suite=$DinnerSuite; stdout=$taskStdout; stderr=$taskStderr; started_at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRunDir "server-$Port.json")
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 80; $taskAttempt++) {
        Start-Sleep -Milliseconds 250
        if ($taskProcess.HasExited) { break }
        try {
            $taskState = Invoke-RestMethod -Uri "$taskUrl/api/state" -TimeoutSec 2
            if ($taskState.ready) { $taskReady = $true; break }
        } catch { }
    }
    if (-not $taskReady) {
        throw "The simulator did not become ready. See $taskStderr."
    }
}
Write-Output "Talos is running at $taskUrl (dinner challenge and BenchLab practice)."
Write-Output 'Stop it with .\stop-lab.ps1 (use the same -Port if changed).'
if (-not $NoBrowser) { Start-Process $taskUrl }
