"""Generiert docs/physics/kalibrierung.md aus der Kalibrierungstabelle.

SSOT ist artificial_society/environment/physics/calibration.py — die Markdown-
Datei ist ein committetes Artefakt; test_reality_gate.py prüft die Synchronität.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from artificial_society.environment.physics.calibration import render_markdown  # noqa: E402


def main() -> None:
    out = REPO_ROOT / "docs" / "physics" / "kalibrierung.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
