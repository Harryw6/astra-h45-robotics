$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
& .\.venv\Scripts\python.exe -m pip install -r outputs\H45_live_repro_20261107\requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed.' }
& .\.venv\Scripts\python.exe -m pip install -e 'outputs\wam_astra[test]'
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
