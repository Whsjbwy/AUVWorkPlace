param(
    [string]$Python = ".\.venv-stage1\Scripts\python.exe",
    [string]$Out = "",
    [switch]$WithLearning
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (!(Test-Path -LiteralPath $Python)) { throw "Python not found: $Python. Install requirements-stage1-learning.txt in .venv-stage1 first." }
$env:MPLBACKEND = "Agg"
# Complete acceptance always includes pretrained weights. Keep the legacy switch.
if ($Out) {
    & $Python scripts_stage1/run_acceptance.py --out $Out
} else {
    & $Python scripts_stage1/run_acceptance.py
}
if ($LASTEXITCODE -ne 0) { throw "Acceptance failed. Inspect acceptance_report.json and command logs in the printed output directory." }
