param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$taskPidPath = Join-Path $PSScriptRoot ".run\server-$Port.json"
if (-not (Test-Path -LiteralPath $taskPidPath)) { Write-Output 'No recorded BenchLab process for that port.'; exit }
$taskRecord = Get-Content -LiteralPath $taskPidPath | ConvertFrom-Json
$taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskRecord.pid)" -ErrorAction SilentlyContinue
if ($taskProcess) {
    $taskAllowedPython = @('.venv\Scripts\python.exe', '.venv-training\Scripts\python.exe') | ForEach-Object { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $_)) }
    $taskExpectedPython = [System.IO.Path]::GetFullPath($taskRecord.python)
    if ($taskExpectedPython -notin $taskAllowedPython -or $taskProcess.ExecutablePath -ne $taskExpectedPython -or $taskProcess.CommandLine -notmatch '-m simulation_lab\.server' -or $taskProcess.CommandLine -notmatch "--port $Port(\s|$)") {
        throw 'The recorded process does not match this workspace simulator. Nothing was stopped.'
    }
    # Windows virtual environments can use a launcher that owns a CPython child.
    # Stop only children running this exact module/port, then its verified launcher.
    $taskChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($taskRecord.pid)"
    foreach ($taskChild in $taskChildren) {
        if ($taskChild.CommandLine -match '-m simulation_lab\.server' -and $taskChild.CommandLine -match "--port $Port(\s|$)") {
            Stop-Process -Id $taskChild.ProcessId -ErrorAction SilentlyContinue
        }
    }
    $taskRemaining = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskRecord.pid)" -ErrorAction SilentlyContinue
    if ($taskRemaining -and $taskRemaining.ExecutablePath -eq $taskExpectedPython -and $taskRemaining.CommandLine -match '-m simulation_lab\.server') {
        Stop-Process -Id $taskRecord.pid -ErrorAction SilentlyContinue
    }
}
Remove-Item -LiteralPath $taskPidPath
Write-Output 'BenchLab stopped.'
