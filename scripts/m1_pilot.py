#!/usr/bin/env python
"""M1-Diagnose-Batterie: 9 gezielte Ablations-Arme im Physik-v2-Pfad.

Spec: docs/superpowers/specs/2026-07-04-m1-diagnose-batterie.md (§1/§2/§3).
Vorläufer: das ursprüngliche `learn`-vs-`nolearn`-A/B (Meilenstein-1-Pilot,
2026-07-02) ist als A1/A2 in dieser Batterie enthalten.

Jeder Arm ist eine Patch-Funktion in der `EXPERIMENTS`-Registry, die VOR dem
`Simulation`-Bau angewendet wird (Runtime-Patches im Runner-Prozess, HOT-Files
bleiben unberührt, kein Import-Time-Monkeypatch im Paket selbst):

  A1 learn              — unverändert (H4-Referenz, volle Maschinerie).
  A2 nolearn            — PPO (`maybe_train`), Todes-Training (`_train_v2`,
                           Review-Finding C1: `finalize_terminal` ruft
                           `_train_v2` DIREKT, ohne über `maybe_train` zu
                           gehen) und soziale Gewichtsangleichung
                           (`imitate_from`) sind No-ops.
  A3 nolearn-nosocial   — A2-Patches + `nosocial`-No-op (echte Random-Null,
                           V-1: A2 allein lässt den Kausal-Kultur-Kanal über
                           Obs-Dims 34-36 weiterlaufen).
  B1 gamma-short        — `brain.GAMMA_V2 = 0.80` (Kredit-Horizont, H1).
  C1 curio-off          — `Agent._assemble_curiosity_v2` auf `0.0 × Original`
                           gewrapped (H2, Original bleibt für Seiteneffekte
                           in jedem Tick lebendig, siehe Spec §1 C-1).
  C2 curio-high         — wie C1, aber `3.0 × Original` (H2, Überdosierung).
  D1 nosocial           — `social_learning_step` im `agents.agent`-Modul-
                           Namespace auf No-op (H3, Kultur-Kanal isolieren;
                           patcht bewusst NICHT `systems.social_learning`,
                           weil `agent.py` die Funktion per
                           `from ... import social_learning_step` gebunden
                           hat — das Quellmodul-Attribut zu patchen hätte
                           dort keine Wirkung mehr).
  E1 entropy-high       — `brain.ENTROPY_COEF_NEW = 0.05` (Verb-Exploration).
  F1 verb-bias-high     — `brain.VERB_INIT_BIAS = 0.6` vor Sim-Bau
                           (construction-time, `Brain.__init__`).

Review-Finding I1 (Checkpoint-Kollision paralleler Prozesse) gilt einheitlich
in JEDEM Arm: `simulation.CHECKPOINT_INTERVAL` wird nach dem Import auf 0
gesetzt (Modul-Global, call-time in `Simulation.step` gelesen) — kein Prozess
schreibt `checkpoint.pkl` ins CWD.

Ausgabe je Run: <out>/<exp>_seed<seed>.jsonl (eine Snapshot-Zeile je
--snapshot-interval Ticks, erste Zeile ein "meta"-Record mit "exp" +
"exp_id" + den gepatchten Werten). Auswertung: scripts/m1_report.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time

# --------------------------------------------------------------------------
# Patch-Bausteine (siehe Spec §1 für das exakte Muster je Knob).
# --------------------------------------------------------------------------


def _freeze_learning() -> None:
    """A2/nolearn-Basis (auch Teil von A3): Policy-Verbesserung UND
    Todes-Training global abschalten (nur in diesem Prozess).

    Review C1 (PFLICHT): `finalize_terminal` (agents/brain.py) ruft
    `self._train_v2(...)` DIREKT beim Agententod, ohne über `maybe_train` zu
    gehen. Ein Freeze, der nur `maybe_train` no-op't, lässt den nolearn-Arm
    bei jedem Tod weitertrainieren (PPO-`optimizer.step()`). Daher wird
    zusätzlich `_train_v2` selbst no-op't (deckt beide Aufrufpfade ab).
    """
    from artificial_society.agents.brain import Brain

    Brain.maybe_train = lambda self: None  # kein PPO-Update über den Rollout-Zyklus
    Brain.imitate_from = lambda self, *a, **k: None  # keine soziale Gewichtskopie
    Brain._train_v2 = lambda self, *a, **k: None  # Review C1: Todes-Trainingspfad


def _nosocial() -> None:
    """D1 (und Teil von A3): Kultur-Kanal (Kausal-Sequenz-Transfer +
    Gewichts-Imitation über social_learning_step) abschalten.

    Muss auf dem Import-Namen in `artificial_society.agents.agent` patchen:
    `agent.py` bindet die Funktion via `from ...social_learning import
    social_learning_step` an einen eigenen Modul-Namen. Das Quellmodul-
    Attribut `systems.social_learning.social_learning_step` zu patchen hätte
    dort keine Wirkung mehr, weil `agent.py` bereits eine eigene Referenz auf
    das alte Funktionsobjekt hält.
    """
    import artificial_society.agents.agent as agent_mod

    agent_mod.social_learning_step = lambda agent, agents, tick: 0.0


def _curiosity_scale(factor: float):
    """C1 (factor=0.0) / C2 (factor=3.0): `_assemble_curiosity_v2` umhüllen,
    NICHT ersetzen — das Original muss für seine Seiteneffekte (Brain-EMA-
    Puffer, Novelty-Zähler, `_causal_pending`) in jedem Tick unbedingt
    laufen; nur der zurückgegebene Skalar wird skaliert (Spec §1, C-1)."""

    def _apply() -> None:
        from artificial_society.agents.agent import Agent

        orig = Agent._assemble_curiosity_v2
        Agent._assemble_curiosity_v2 = lambda self, *a, **kw: factor * orig(self, *a, **kw)

    return _apply


def _set_brain_const(name: str, value):
    """B1 (GAMMA_V2) / E1 (ENTROPY_COEF_NEW) / F1 (VERB_INIT_BIAS): Modul-
    Global in agents/brain.py setzen. Für GAMMA_V2/ENTROPY_COEF_NEW reicht
    "vor dem ersten sim.step()" (call-time in `_train_v2` gelesen); für
    VERB_INIT_BIAS ist es zwingend "vor dem Simulation()-Bau", weil
    `Brain.__init__` den Wert construction-time liest."""

    def _apply() -> None:
        import artificial_society.agents.brain as brain_mod

        setattr(brain_mod, name, value)

    return _apply


def _compose(*fns):
    def _apply() -> None:
        for fn in fns:
            fn()

    return _apply


def _noop() -> None:
    pass


# --------------------------------------------------------------------------
# Experiment-Registry (Spec §2, Tabelle).
# --------------------------------------------------------------------------

EXPERIMENTS = {
    "learn": {
        "id": "A1",
        "seeds_default": 5,
        "tests": "H4-Referenz (volle Maschinerie)",
        "apply": _noop,
        "patched": {},
    },
    "nolearn": {
        "id": "A2",
        "seeds_default": 5,
        "tests": "H4 (Lernen als Ganzes) + Untergrenze (mit Restkultur)",
        "apply": _freeze_learning,
        "patched": {"maybe_train": "noop", "imitate_from": "noop", "_train_v2": "noop"},
    },
    "nolearn-nosocial": {
        "id": "A3",
        "seeds_default": 3,
        "tests": "echte Random-Null (H4/H3-Referenz ohne Restkultur, V-1)",
        "apply": _compose(_freeze_learning, _nosocial),
        "patched": {
            "maybe_train": "noop",
            "imitate_from": "noop",
            "_train_v2": "noop",
            "social_learning_step": "noop",
        },
    },
    "gamma-short": {
        "id": "B1",
        "seeds_default": 3,
        "tests": "H1 (Kredit-Horizont, Planer-Ersatz)",
        "apply": _set_brain_const("GAMMA_V2", 0.80),
        "patched": {"GAMMA_V2": 0.80},
    },
    "curio-off": {
        "id": "C1",
        "seeds_default": 3,
        "tests": "H2 (trägt Intrinsik nichts?)",
        "apply": _curiosity_scale(0.0),
        "patched": {"curiosity_scale": 0.0},
    },
    "curio-high": {
        "id": "C2",
        "seeds_default": 3,
        "tests": "H2 (Intrinsik unterdosiert?)",
        "apply": _curiosity_scale(3.0),
        "patched": {"curiosity_scale": 3.0},
    },
    "nosocial": {
        "id": "D1",
        "seeds_default": 3,
        "tests": "H3 (Kultur-Kanal isolieren)",
        "apply": _nosocial,
        "patched": {"social_learning_step": "noop"},
    },
    "entropy-high": {
        "id": "E1",
        "seeds_default": 3,
        "tests": "Zusatz-Lever: Verb-Exploration",
        "apply": _set_brain_const("ENTROPY_COEF_NEW", 0.05),
        "patched": {"ENTROPY_COEF_NEW": 0.05},
    },
    "verb-bias-high": {
        "id": "F1",
        "seeds_default": 3,
        "tests": "Zusatz-Lever: Manipulations-Prior",
        "apply": _set_brain_const("VERB_INIT_BIAS", 0.6),
        "patched": {"VERB_INIT_BIAS": 0.6},
    },
}

# Kanonische Report-Reihenfolge (Spec §2-Tabelle).
EXPERIMENT_ORDER = [
    "learn",
    "nolearn",
    "nolearn-nosocial",
    "gamma-short",
    "curio-off",
    "curio-high",
    "nosocial",
    "entropy-high",
    "verb-bias-high",
]


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


def run_one(
    exp: str,
    seed: int,
    ticks: int,
    grid_w: int,
    grid_h: int,
    pop: int,
    snapshot_interval: int,
    out_dir: str,
) -> str:
    conf = EXPERIMENTS[exp]
    # Patches VOR dem Simulation-Bau anwenden (Spec §1). Modul-Konstanten wie
    # VERB_INIT_BIAS sind construction-time (Brain.__init__); Methoden-Wraps
    # und No-ops sind Klassen-/Modul-Attribute und wirken unabhängig von der
    # Reihenfolge relativ zum `Simulation`-Import, aber wir halten uns an das
    # in der Spec vorgeschriebene Muster.
    conf["apply"]()

    from artificial_society import simulation as simulation_mod
    from artificial_society.simulation import Simulation

    # Review I1 (PFLICHT, alle Arme einheitlich): kein `checkpoint.pkl` im
    # gemeinsamen CWD paralleler Arm×Seed-Prozesse. Modul-Global, call-time
    # in `Simulation.step` gelesen (`if CHECKPOINT_INTERVAL and ...`).
    simulation_mod.CHECKPOINT_INTERVAL = 0

    sim = Simulation(
        headless=True,
        load_checkpoint=False,
        physics_v2=True,
        grid_w=grid_w,
        grid_h=grid_h,
        initial_population=pop,
        seed=seed,
    )
    path = os.path.join(out_dir, f"{exp}_seed{seed}.jsonl")
    t0 = time.time()
    with open(path, "w") as f:
        meta = {
            "record": "meta",
            "exp": exp,
            "exp_id": conf["id"],
            "seed": seed,
            "ticks": ticks,
            "grid": [grid_w, grid_h],
            "pop": pop,
            "snapshot_interval": snapshot_interval,
            "patched": dict(conf["patched"], CHECKPOINT_INTERVAL=0),
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
    print(
        f"[{exp} seed{seed}] done pop={final['pop']} "
        f"tool_cut_ratio={final['tool_cut_ratio']} "
        f"discoveries={final['discoveries']} in {final['walltime_s']}s -> {path}"
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=sorted(EXPERIMENTS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--grid-w", type=int, default=40)
    ap.add_argument("--grid-h", type=int, default=30)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--snapshot-interval", type=int, default=250)
    ap.add_argument("--out", default="pilot_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_one(
        args.exp,
        args.seed,
        args.ticks,
        args.grid_w,
        args.grid_h,
        args.pop,
        args.snapshot_interval,
        args.out,
    )


if __name__ == "__main__":
    main()
