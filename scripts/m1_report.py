#!/usr/bin/env python
"""Wertet den Meilenstein-1-Pilot aus: A/B (learn vs nolearn) über Seeds.

Liest <dir>/*.jsonl (von m1_pilot.py), aggregiert die Final-Records je Arm und gibt
einen Vergleich der Emergenz-Metriken aus. Kernfrage: schlägt learn nolearn?
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def _load_finals(directory: str) -> dict:
    arms: dict = {"learn": [], "nolearn": []}
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        final = None
        arm = None
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("record") == "meta":
                    arm = rec["arm"]
                elif rec.get("record") == "final":
                    final = rec
        if arm in arms and final is not None:
            arms[arm].append(final)
    return arms


def _agg(finals: list, key: str) -> tuple:
    vals = [f[key] for f in finals if key in f]
    if not vals:
        return (0.0, 0.0, 0)
    mean = st.mean(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return (mean, sd, len(vals))


METRICS = [
    ("tool_cut_ratio", "Anteil Schnitte MIT Werkzeug"),
    ("cuts_with_tool", "Schnitte mit Werkzeug (Σ)"),
    ("fragments_total", "Schlag-Fragmente (Σ)"),
    ("discoveries", "entdeckte Materialien"),
    ("pop", "Endpopulation"),
    ("mean_energy", "mittlere Energie"),
    ("births", "Geburten (Σ)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pilot_results")
    args = ap.parse_args()
    arms = _load_finals(args.dir)
    nL, nN = len(arms["learn"]), len(arms["nolearn"])
    print(f"\n=== Meilenstein-1-Pilot: A/B-Auswertung ===")
    print(f"Seeds: learn={nL}, nolearn={nN}  (Ordner: {args.dir})\n")
    print(f"{'Metrik':<34}{'learn':>16}{'nolearn':>16}{'Δ (L-N)':>14}")
    print("-" * 80)
    for key, label in METRICS:
        lm, ls, _ = _agg(arms["learn"], key)
        nm, ns, _ = _agg(arms["nolearn"], key)
        delta = lm - nm
        print(f"{label:<34}{lm:>10.3f}±{ls:<4.2f}{nm:>10.3f}±{ns:<4.2f}{delta:>14.3f}")
    print("-" * 80)
    # Verdikt auf dem Kern-Signal.
    lm, _, _ = _agg(arms["learn"], "tool_cut_ratio")
    nm, _, _ = _agg(arms["nolearn"], "tool_cut_ratio")
    if nL and nN:
        if lm > nm and lm > 0:
            print(f"\n→ learn > nolearn beim Werkzeug-Schnitt-Anteil "
                  f"({lm:.3f} vs {nm:.3f}). Lernen zahlt sich mechanistisch aus.")
        elif lm <= nm:
            print(f"\n→ KEIN Lernvorteil beim Werkzeug-Schnitt-Anteil "
                  f"({lm:.3f} vs {nm:.3f}). Engpass = Lern-Maschinerie/Kopplung "
                  f"(vgl. archivierter Kernbefund).")
    print()


if __name__ == "__main__":
    main()
