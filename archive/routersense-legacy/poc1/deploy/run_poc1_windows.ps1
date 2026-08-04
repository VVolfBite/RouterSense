# Deployment-facing Windows entrypoint.
# Keeps deployment commands under deploy/ while reusing the existing smoke-test script.
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
$Script = Join-Path $Root "scripts\run_poc1_windows.ps1"

$ArgsList = @(
  "-ExecutionPolicy", "Bypass",
  "-File", $Script,
  "-Python", $Python,
  "-ModelId", $ModelId,
  "-TextInput", $TextInput,
  "-OutputDir", $OutputDir
)

if ($UseCurrentPython) {
  $ArgsList += "-UseCurrentPython"
}
if ($SkipInstall) {
  $ArgsList += "-SkipInstall"
}

& powershell @ArgsList
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
