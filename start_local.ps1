Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

foreach ($key in @("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")) {
    Remove-Item "Env:$key" -ErrorAction SilentlyContinue
}

$pythonBin = $env:AUDIOBOOKSTUDIO_PYTHON
if (-not $pythonBin -and $env:CONDA_PREFIX) {
    $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $condaPython) {
        $pythonBin = $condaPython
    }
}
if (-not $pythonBin -and (Test-Path "I:\conda_envs\omnivoice\python.exe")) {
    $pythonBin = "I:\conda_envs\omnivoice\python.exe"
}
if (-not $pythonBin -and (Test-Path ".venv\Scripts\python.exe")) {
    $pythonBin = ".venv\Scripts\python.exe"
}
if (-not $pythonBin) {
    $pythonBin = "python"
}

& $pythonBin "start_local.py"
