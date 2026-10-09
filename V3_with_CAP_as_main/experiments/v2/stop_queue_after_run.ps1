# Stops the first post-audit queue at a run boundary: waits until the named run has written its
# summary, then pauses the watchdog and ends the queue process before the next run saves a case.
# Nothing is lost: the queue is resumable (delete data\results\v2\QUEUE_PAUSED and run
# "runner.py --queue"), and the runs it had not started simply do not exist.
#
#   powershell -ExecutionPolicy Bypass -File experiments\v2\stop_queue_after_run.ps1 -Experiment e4_repaired_Full -Run seed1
param([Parameter(Mandatory = $true)][string]$Experiment, [Parameter(Mandatory = $true)][string]$Run, [int]$TimeoutMinutes = 360)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$results = Join-Path $project "data\results\v2"
$status = Join-Path $project "data\logs\v2\status.txt"
$summary = Join-Path $results "$Experiment\$Run\summary.json"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)

function Note($text) {
    Add-Content -Path $status -Value ("stop-after-run {0}: {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $text) -Encoding utf8
}

function QueueProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'experiments[\\/]v2[\\/]runner\.py' -and $_.CommandLine -match '--queue' -and $_.CommandLine -notmatch '--limit'
    }
}

Note "waiting for $Experiment/$Run to finish"
while (-not (Test-Path $summary)) {
    if ((Get-Date) -gt $deadline) { Note "gave up after $TimeoutMinutes minutes; nothing was stopped"; exit 2 }
    if (-not (QueueProcesses)) { Note "the queue is not running; nothing to stop"; exit 3 }
    Start-Sleep -Seconds 3
}

# The watchdog must not start the queue again.
New-Item -ItemType File -Force (Join-Path $results "QUEUE_PAUSED") | Out-Null
$stopped = @()
foreach ($process in (QueueProcesses)) {
    try { Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop; $stopped += $process.ProcessId } catch {}
}
Start-Sleep -Seconds 3
$left = @(QueueProcesses | ForEach-Object { $_.ProcessId })
Note ("{0}/{1} finished; queue paused and stopped (processes {2}); still running: {3}" -f $Experiment, $Run, ($stopped -join ", "), ($(if ($left) { $left -join ", " } else { "none" })))
"stopped after $Experiment/$Run"
