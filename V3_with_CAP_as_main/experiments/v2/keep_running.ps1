# Keeps the experiment queue running until it has finished.
#
# Run by the Windows scheduled task "LexAgentV2Queue" every 5 minutes (see keep_running_install.ps1).
# It does nothing while a queue is running. If none is running and the queue has not finished, it
# starts one. The queue resumes from its saved cases, so a crash, a closed terminal, a stopped
# Claude session or a reboot costs at most the case that was in progress.
#
# Since 7 October 2026 the queue it keeps alive is the FINAL queue ("runner.py --final": the system
# after repairs 6 and 7, its baselines and the standard measures). The first post-audit queue
# ("runner.py --queue") was stopped on purpose at a run boundary once its informative runs were
# complete; it can still be continued by hand, and the two can never run together (they hold the
# same lock).
#
# The queue is started through WMI, so it belongs neither to this script nor to whatever started
# this script: it keeps running when they exit.
#
#   Stop everything:   powershell -ExecutionPolicy Bypass -File experiments\v2\keep_running_install.ps1 -Remove

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$results = Join-Path $project "data\results\v2"
$logs = Join-Path $project "data\logs\v2"
$status = Join-Path $logs "status.txt"
$taskName = "LexAgentV2Queue"
New-Item -ItemType Directory -Force $logs | Out-Null

function Note($text) {
    Add-Content -Path $status -Value ("watchdog {0}: {1}" -f (Get-Date -Format "MM-dd HH:mm"), $text) -Encoding utf8
}

# Finished: remove the scheduled task and stop.
if (Test-Path (Join-Path $results "FINAL_QUEUE_FINISHED")) {
    $existing = schtasks /Query /TN $taskName 2>$null
    if ($LASTEXITCODE -eq 0) {
        schtasks /Delete /TN $taskName /F | Out-Null
        Note "final queue finished; scheduled task removed"
    }
    exit 0
}

# Paused on purpose: do nothing (create the file data\results\v2\QUEUE_PAUSED to pause, delete it to continue).
if (Test-Path (Join-Path $results "QUEUE_PAUSED")) { exit 0 }

# A queue of either kind is running: do nothing.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'experiments[\\/]v2[\\/]runner\.py' -and $_.CommandLine -match '--(queue|final)' -and $_.CommandLine -notmatch '--limit' }
if ($running) { exit 0 }

# Not running and not finished: start it, detached.
$python = Join-Path $project ".venv\Scripts\python.exe"
$log = Join-Path $logs "final_queue.log"
$command = 'cmd.exe /c "set PYTHONIOENCODING=utf-8&& set PYTHONUNBUFFERED=1&& set HF_HUB_DISABLE_SYMLINKS_WARNING=1&& "' + $python + '" experiments/v2/runner.py --final >> "' + $log + '" 2>&1"'
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $command; CurrentDirectory = $project }
if ($created.ReturnValue -eq 0) {
    Note ("final queue was not running; started it (process {0})" -f $created.ProcessId)
} else {
    Note ("final queue was not running; starting it FAILED with code {0}" -f $created.ReturnValue)
    exit 1
}
