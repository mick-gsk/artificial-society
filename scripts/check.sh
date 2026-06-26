#!/usr/bin/env bash
# Local dev gate — mirrors .github/workflows/ci.yml so a green /check predicts a
# green CI run. Lints/format-checks only the Python files you changed (legacy code
# is grandfathered) and runs the FULL pytest suite (incl. the determinism contract:
# golden trajectory + headless digest).
#
# Usage: bash scripts/check.sh [base-ref]      (base defaults to origin/main)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
ROOT="$(pwd)"

PY="${PYTHON:-$ROOT/../venv/bin/python}"
RUFF="${RUFF:-$ROOT/../venv/bin/ruff}"
[ -x "$PY" ] || PY="python"
[ -x "$RUFF" ] || RUFF="ruff"
BASE="${1:-origin/main}"

# Changed Python files: committed vs base + unstaged + staged, de-duplicated.
changed="$( {
  git diff --name-only --diff-filter=ACMR "${BASE}...HEAD" -- '*.py' 2>/dev/null
  git diff --name-only -- '*.py'
  git diff --name-only --cached -- '*.py'
} | sort -u | grep -E '\.py$' || true )"

rc=0
if [ -n "$changed" ]; then
  echo "== ruff check (changed files) =="
  printf '  %s\n' $changed
  echo "$changed" | xargs "$RUFF" check || rc=1
  echo "== ruff format --check (changed files) =="
  echo "$changed" | xargs "$RUFF" format --check || rc=1
else
  echo "== ruff: no changed Python files =="
fi

echo "== pytest (full suite incl. determinism contract) =="
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "$PY" -m pytest -q || rc=1

if [ "$rc" -eq 0 ]; then
  echo "OK — check passed (predicts green CI)."
else
  echo "FAIL — fix the above before opening a PR. Never edit a determinism test to pass."
fi
exit "$rc"
