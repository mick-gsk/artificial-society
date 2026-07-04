#!/usr/bin/env python
"""Meilenstein-1-Pilot: Emergenz-A/B (Lernen ON vs OFF) auf der Physik-v2-Welt.

Frage (Spec 2026-07-02, §1): Schlägt eine Population *mit* Lernen eine ohne — über
mehrere Seeds — in der frischen v2-Welt, in der Werkzeuge nur mechanistisch (scharfe
Kante → mehr Schneid-Ertrag) und nicht per Rezept-Bonus etwas bringen?

Zwei Arme, identische Welt pro Seed:
  * learn : unverändert.
  * nolearn: eingefrorenes Zufalls-Netz — PPO-Training (Brain.maybe_train) UND soziale
    Gewichtsangleichung (Brain.imitate_from) sind No-ops. Genau die Lern-Maschinerie ist
    aus; Wahrnehmung/Handeln/Fortpflanzung/Physik laufen identisch. Die RNG-Guards in
    social_learning werden VOR imitate_from ausgewertet, der No-op verschiebt den
    Zufallsstrom also nicht — beide Arme starten auf demselben Seed-Weltzustand.

Dies ist ein Experiment-Harness (kein Paket-Monkeypatch beim Import): die No-ops werden
nur im nolearn-Prozess gesetzt, HOT-Files bleiben unberührt.

Ausgabe je Run: <out>/<arm>_seed<seed>.jsonl (eine Snapshot-Zeile je --snapshot-interval
Ticks, erste Zeile ist ein "meta"-Record). Auswertung: scripts/m1_report.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def _freeze_learning() -> None:
    """nolearn-Arm: Policy-Verbesserung global abschalten (nur in diesem Prozess)."""
    from artificial_society.agents.brain import Brain

    Brain.maybe_train = lambda self: None  # kein PPO/Gradienten-Update
    Brain.imitate_from = lambda self, *a, **k: None  # keine soziale Gewichtskopie


def _snapshot(sim, tick: int) -> dict:
    agents = sim.agents
    pop = len(agents)
    mean_energy = sum(a.energy for a in agents) / pop if pop else 0.0
    m = sim.world.objects.metrics_snapshot()
    cuts_tool = int(m.get("cuts_with_tool", 0))
    cuts_hand = int(m.get("cuts_bare_hand", 0))
    cuts_total = cuts_tool + cuts_hand
    verbs = m.get("verbs_fired", {}) or {}
    return {
        "tick": tick,
        "pop": pop,
        "mean_energy": round(mean_energy, 2),
        # Kern-Emergenz-Signal: Anteil der Schnitte, die MIT Werkzeug erfolgen.
        "cuts_with_tool": cuts_tool,
        "cuts_bare_hand": cuts_hand,
        "tool_cut_ratio": round(cuts_tool / cuts_total, 4) if cuts_total else 0.0,
        "fragments_total": int(m.get("fragments_total", 0)),
        "discoveries": len(sim.world.objects.discovery.entries),
        "verbs_fired": {k: int(v) for k, v in verbs.items()},
        "births": sum(getattr(a, "children", 0) for a in agents),
    }


def run_one(arm: str, seed: int, ticks: int, grid_w: int, grid_h: int, pop: int,
            snapshot_interval: int, out_dir: str) -> str:
    if arm == "nolearn":
        _freeze_learning()
    # Import NACH dem Freeze, damit die gepatchte Klasse verwendet wird.
    from artificial_society.simulation import Simulation

    sim = Simulation(
        headless=True,
        load_checkpoint=False,
        physics_v2=True,
        grid_w=grid_w,
        grid_h=grid_h,
        initial_population=pop,
        seed=seed,
    )
    path = os.path.join(out_dir, f"{arm}_seed{seed}.jsonl")
    t0 = time.time()
    with open(path, "w") as f:
        meta = {
            "record": "meta", "arm": arm, "seed": seed, "ticks": ticks,
            "grid": [grid_w, grid_h], "pop": pop, "snapshot_interval": snapshot_interval,
        }
        f.write(json.dumps(meta) + "\n")
        for tick in range(ticks):
            sim.step()
            if (tick + 1) % snapshot_interval == 0:
                rec = _snapshot(sim, tick + 1)
                rec["record"] = "snap"
                f.write(json.dumps(rec) + "\n")
                f.flush()
        final = _snapshot(sim, ticks)
        final["record"] = "final"
        final["walltime_s"] = round(time.time() - t0, 1)
        f.write(json.dumps(final) + "\n")
    print(f"[{arm} seed{seed}] done pop={final['pop']} "
          f"tool_cut_ratio={final['tool_cut_ratio']} "
          f"discoveries={final['discoveries']} in {final['walltime_s']}s -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["learn", "nolearn"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--grid-w", type=int, default=40)
    ap.add_argument("--grid-h", type=int, default=30)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--snapshot-interval", type=int, default=250)
    ap.add_argument("--out", default="pilot_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_one(args.arm, args.seed, args.ticks, args.grid_w, args.grid_h, args.pop,
            args.snapshot_interval, args.out)


if __name__ == "__main__":
    main()
