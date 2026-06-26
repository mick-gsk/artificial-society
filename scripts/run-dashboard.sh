#!/usr/bin/env bash
# Local dashboard run for development / testing (e.g. on the MacBook, CPU).
# On the GPU host use scripts/run-dashboard.bat instead.
set -euo pipefail
cd "$(dirname "$0")/.."

# Reproducible seeded runs need a fixed hash seed set before the process starts.
export PYTHONHASHSEED=0

# venv lives one level up at ../venv (see CLAUDE.md); fall back to PATH python.
if [ -x "../venv/bin/python" ]; then
  PY="../venv/bin/python"
elif [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
else
  PY="python3"
fi

echo "Starting dashboard on http://0.0.0.0:8000  (Ctrl+C to stop)"
exec "$PY" -m artificial_society.serve --host 0.0.0.0 --port 8000
