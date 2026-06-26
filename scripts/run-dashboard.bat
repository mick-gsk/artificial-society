@echo off
REM Start the Artificial Society dashboard on the Windows PC (GPU host).
REM Reachable from the LAN at http://<this-pc-ip>:8000  (find the IP via: ipconfig)
REM
REM Assumes a venv at the repo root created per docs\serve-setup.md, with torch
REM installed from the cu128 index so the RTX 5070 Ti (Blackwell / sm_120) is used.

setlocal
cd /d "%~dp0\.."

REM Reproducible seeded runs need a fixed hash seed set BEFORE the process starts.
set PYTHONHASHSEED=0

if exist "venv\Scripts\python.exe" (
  set "PY=venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo Starting dashboard on http://0.0.0.0:8000  (Ctrl+C to stop)
"%PY%" -m artificial_society.serve --host 0.0.0.0 --port 8000

endlocal
