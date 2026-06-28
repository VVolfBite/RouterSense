# Windows local smoke test for RouteSense PoC1.
# Purpose: verify Python env -> real OLMoE load -> real routing trace export.
# This is not the full server experiment and does not run ablation.
param(
  [string]$Python = "python",
  [string]$ModelId = "model\OLMoE",
  [string]$TextInput = "The history of science is a story of",
  [string]$OutputDir = "outputs\poc1_windows_smoke",
  [switch]$UseCurrentPython,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$RuntimePython = $VenvPython
$Requirements = Join-Path $Root "legacy/shared/requirements-gpu.txt"
$ResolvedOutputDir = Join-Path $Root $OutputDir

Set-Location $Root

function Run-Step {
  param(
    [string]$Name,
    [scriptblock]$Command,
    [string]$FailureHint
  )

  Write-Host "[windows-smoke] START $Name"
  try {
    & $Command
  }
  catch {
    Write-Host "[windows-smoke] FAILED $Name`: $($_.Exception.Message)"
    if ($FailureHint) {
      Write-Host "[windows-smoke] Hint: $FailureHint"
    }
    throw
  }
  Write-Host "[windows-smoke] DONE  $Name"
}

function Invoke-Native {
  param(
    [string]$FilePath,
    [string[]]$Arguments
  )

  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "'$FilePath $($Arguments -join ' ')' exited with code $LASTEXITCODE"
  }
}

Write-Host "[windows-smoke] RouteSense PoC1 local trace smoke test"
Write-Host "[windows-smoke] This validates real model loading and real routing trace export only."
Write-Host "[windows-smoke] Root: $Root"
Write-Host "[windows-smoke] ModelId: $ModelId"
Write-Host "[windows-smoke] OutputDir: $ResolvedOutputDir"
Write-Host "[windows-smoke] Text: $TextInput"
Write-Host "[windows-smoke] UseCurrentPython: $UseCurrentPython"

Run-Step "check-python" {
  Invoke-Native $Python @("--version")
} "Python was not found. Pass -Python C:\Path\To\python.exe or install Python 3.10+."

if ($UseCurrentPython) {
  $RuntimePython = $Python
  Write-Host "[windows-smoke] Using current Python directly; venv creation/install is skipped."
}
elseif (-not (Test-Path $VenvPython)) {
  Run-Step "create-venv" {
    Invoke-Native $Python @("-m", "venv", (Join-Path $Root ".venv"))
  } "Virtual environment creation failed. Check that Python venv support is installed."
}

if ($UseCurrentPython) {
  Write-Host "[windows-smoke] SKIP install-dependencies"
}
elseif (-not $SkipInstall) {
  Run-Step "install-dependencies" {
    Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Invoke-Native $VenvPython @("-m", "pip", "install", "-r", $Requirements)
  } "Dependency installation failed. Check network access and CUDA/PyTorch package compatibility."
}
else {
  Write-Host "[windows-smoke] SKIP install-dependencies"
}

New-Item -ItemType Directory -Force $ResolvedOutputDir | Out-Null
$TracePath = Join-Path $ResolvedOutputDir "one_sample_trace.json"
$SummaryPath = Join-Path $ResolvedOutputDir "batch_routing_summary.json"
Remove-Item -LiteralPath $TracePath, $SummaryPath -Force -ErrorAction SilentlyContinue

Run-Step "preflight" {
  Invoke-Native $RuntimePython @((Join-Path $PSScriptRoot "preflight_gpu.py"))
} "Environment preflight failed. This is usually a Python, torch, CUDA, or dependency issue."

Run-Step "real-trace" {
  Invoke-Native $RuntimePython @(
    (Join-Path $PSScriptRoot "trace_one_token.py"),
    "--text", $TextInput,
    "--layer", "auto",
    "--model-id", $ModelId,
    "--output-dir", $OutputDir
  )
} "Real trace failed. Common causes: model path is wrong, weights are incomplete, CUDA is unavailable, or GPU memory/offload is insufficient."

if (-not (Test-Path $TracePath)) {
  throw "[windows-smoke] Missing expected output: $TracePath"
}
if (-not (Test-Path $SummaryPath)) {
  throw "[windows-smoke] Missing expected output: $SummaryPath"
}

Write-Host "[windows-smoke] SUCCESS real trace outputs generated:"
Write-Host "  $TracePath"
Write-Host "  $SummaryPath"
Write-Host "[windows-smoke] Outputs:"
Get-ChildItem $ResolvedOutputDir -File | Sort-Object Name | ForEach-Object {
  Write-Host ("  {0} ({1} bytes)" -f $_.Name, $_.Length)
}
