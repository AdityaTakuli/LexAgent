# Installs (or removes) the scheduled task that keeps the experiment queue running.
#
#   Install:  powershell -ExecutionPolicy Bypass -File experiments\v2\keep_running_install.ps1
#   Remove:   powershell -ExecutionPolicy Bypass -File experiments\v2\keep_running_install.ps1 -Remove
#             (-Remove also stops a running queue; finished cases stay saved and it can be resumed later)
#
# The task runs as the current user, every 5 minutes, only while that user is logged on, and needs no
# administrator rights. It removes itself when the queue has finished.
param([switch]$Remove)

$taskName = "LexAgentV2Queue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $here "keep_running_hidden.vbs"

if ($Remove) {
    schtasks /Delete /TN $taskName /F 2>$null | Out-Null
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'experiments[\\/]v2[\\/]runner\.py' -and $_.CommandLine -match '--(queue|final)' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    "Scheduled task removed and the queue stopped. Finished cases are saved; install again to resume."
    exit 0
}

$action = 'wscript.exe //B //Nologo "' + $launcher + '"'
schtasks /Create /F /SC MINUTE /MO 5 /TN $taskName /TR $action | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Could not create the scheduled task $taskName" }
# The default task settings would stop it on battery power and never run a missed start: relax both.
try {
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
    Set-ScheduledTask -TaskName $taskName -Settings $settings | Out-Null
} catch { "Note: could not adjust the task settings ($($_.Exception.Message)); the defaults are used." }
"Scheduled task '$taskName' installed: every 5 minutes it starts the queue if it is not running."
