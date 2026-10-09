# Rebuild the paper: regenerate the numbers from the saved result files, then compile the PDF.
#   powershell -ExecutionPolicy Bypass -File paper\build.ps1            (numbers + PDF)
#   powershell -ExecutionPolicy Bypass -File paper\build.ps1 -SkipResults   (PDF only)
param([switch]$SkipResults)
$ErrorActionPreference = "Stop"
$paper = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Split-Path -Parent $paper
$tectonic = Join-Path $paper ".tectonic\tectonic.exe"
if (-not (Test-Path $tectonic)) {
    throw "Tectonic is not in paper\.tectonic. Download tectonic-<version>-x86_64-pc-windows-msvc.zip from https://github.com/tectonic-typesetting/tectonic/releases and unzip tectonic.exe there."
}
if (-not $SkipResults) {
    & (Join-Path $project ".venv\Scripts\python.exe") (Join-Path $paper "make_results.py")
    if ($LASTEXITCODE -ne 0) { throw "make_results.py failed" }
}
$env:TECTONIC_CACHE_DIR = Join-Path $paper ".tectonic\cache"
Push-Location $paper
try {
    & $tectonic --keep-logs lexagent_ieee.tex
    if ($LASTEXITCODE -ne 0) { throw "tectonic failed; see paper\lexagent_ieee.log" }
} finally { Pop-Location }
"Built " + (Join-Path $paper "lexagent_ieee.pdf")
