$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required.' }
}
& $taskPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $taskPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
Write-Host 'SkinDraw is ready. Run start.pyw or .\.venv\Scripts\python.exe -m skin_draw'

