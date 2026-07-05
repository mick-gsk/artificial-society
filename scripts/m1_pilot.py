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

K2 (Respawn-Mühle, docs/superpowers/specs/2026-07-04-m1-batterie-ergebnis-und-
empfehlungen.md): `simulation.MIN_POPULATION` (Default 8, call-time gelesen
in `Simulation.step`, simulation.py:639) und `simulation.RESPAWN_COUNT`
(Default 6, call-time gelesen in `emergency_respawn`, simulation.py:305)
werden nach dem Import per `--min-pop`/`--respawn-count` überschrieben —
gleiches Muster wie `CHECKPOINT_INTERVAL`. Beide Werte landen im meta-Record.

Demografie-Instrumentierung (Runner-seitig, kein Paket-Eingriff):
  - `respawns`: `sim.emergency_respawn` wird NACH dem Sim-Bau durch eine
    zählende Wrapper-Closure ersetzt (Bound-Method-Wrap auf der Instanz,
    nicht auf der Klasse — betrifft nur diesen einen `sim`). Kumulativer
    Zähler (Anzahl AUFRUFE, nicht Anzahl respawnter Agenten) in jedem
    snap/final-Record als `respawns`.
  - `deaths` (Review I-1): `sim.remove_dead` wird NACH dem Sim-Bau durch eine
    zählende Wrapper-Closure ersetzt (gleiches Bound-Method-Wrap-Muster wie
    `emergency_respawn`). `remove_dead()` wird laut `simulation.py` GENAU
    EINMAL pro Tick in `Simulation.step()` aufgerufen (Zeile ~627) und ist der
    ursachen-agnostische Todes-Aggregationspunkt (Spec B3) — jeder Agent, der
    stirbt, durchläuft ihn. Der Wrapper zählt `len(sim.agents)` vor und nach
    dem (genau einmal ausgeführten) Original-Aufruf und kumuliert die Differenz
    über den ganzen Lauf. Das ist ein EXAKTER Zähler, keine Untergrenze.
  - `ids_seen_total`: Menge aller `Agent.id`, die in irgendeinem Snapshot
    (inkl. final) als lebend beobachtet wurden, kumulativ über den ganzen
    Lauf. Das ist eine unvollständige, aber KORREKTE Untergrenze für
    "jemals gelebte Agenten" — Geburten UND Tode, die vollständig zwischen
    zwei Snapshots (Default alle 250 Ticks) liegen, werden nicht gezählt,
    weil dazu ein Per-Tick-Hook nötig wäre (Paket-Eingriff, hier bewusst
    vermieden). Ehrliche Unterschätzung, keine Überschätzung. Seit Review I-1
    ist `deaths` der exakte Zähler; `ids_seen_total` bleibt als billige
    Komplementärmetrik (Geburten+Tode kombiniert) im Record erhalten. Die
    Identität `ids_seen_total == pop_final + deaths` gilt NUR, wenn zusätzlich
    jede Geburt zwischen zwei Snapshots in einem späteren Snapshot noch lebend
    beobachtet wurde; stirbt ein Agent vollständig zwischen zwei Snapshots
    (geboren UND gestorben, ohne je in einem Snapshot lebend gezählt zu
    werden), zählt `deaths` ihn, `ids_seen_total` aber nicht — dann gilt nur
    noch `deaths >= ids_seen_total - pop_final`.
  - `mean_age`: Mittelwert von `tick - a.birth_tick` über die aktuell
    lebenden Agenten je Snapshot (nur Momentaufnahme, kein kumulativer Wert).
  - `turnover_final` (nur im final-Record): `ids_seen_total / max(pop, 1)`
    — grobe Kennzahl, wie oft sich die Population über den Lauf "ausgetauscht"
    hat (untere Schranke, siehe `ids_seen_total`-Einschränkung oben; für den
    exakten Todes-Zähler siehe `deaths`).

Ausgabe je Run: <out>/<exp>_seed<seed>.jsonl (eine Snapshot-Zeile je
--snapshot-interval Ticks, erste Zeile ein "meta"-Record mit "exp" +
"exp_id" + den gepatchten Werten). Auswertung: scripts/m1_report.py.

Etappe 1 (docs/superpowers/specs/2026-07-04-k3-auswertung-und-naechste-etappe.md,
§Empfehlung [1]): K3 zeigte, dass sich die Population NICHT selbst trägt
(synchroner Gründer-Overshoot 30→44 → Massentod → Respawn-Boden). Drei
Stellschrauben aus der dortigen Empfehlung sind hier als CLI-Knobs sweepbar,
damit ein Kalibrier-Sweep eine selbsttragende Konfiguration findet (Gate:
respawns→0 nach Einschwingen, births≈deaths, Pop stabil ~20-40):

  --age-structured-founders — gestaffelte Startalter statt der synchronen
    Alter-0-Gründerkohorte (Details: `_age_structure_founders`).
  --regrow-scale <float>    — skaliert die Nahrungs-Zufluss-Konstanten in
    `environment/resources.py` multiplikativ (Details: `_scale_regrowth`).
  --min-food-per-capita <float> — patcht `agents.agent.
    REPRODUCTION_MIN_FOOD_PER_CAPITA` (Details: `_set_min_food_per_capita`).

Alle drei landen im meta-Record (`age_structured_founders`, `regrow_scale`,
`min_food_per_capita`), analog zu `min_pop`/`respawn_count`. Zusätzlich neu:
`births_cum` (siehe `_snapshot`-Docstring) — ein exakter kumulativer
Geburtenzähler, der das bestehende `births`-Feld (Σ `children` NUR der
aktuell LEBENDEN Agenten) ergänzt, weil dieses durch Tode untererfasst wird:
ein Agent, der ein Kind bekommt und dann stirbt, verschwindet aus der Summe,
obwohl die Geburt stattfand.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
# Etappe 1 (selbsttragende Demografie) — Harness-weite Knobs, unabhängig von
# der EXPERIMENTS-Registry oben (die patcht Lern-/Sozial-/Brain-Verhalten pro
# Arm; diese drei patchen Demografie/Ökologie und gelten für JEDEN Arm
# gleich). Siehe Modul-Docstring "Etappe 1" für den Kontext (K3-Empfehlung).
# --------------------------------------------------------------------------


def _age_structure_founders(sim) -> None:
    """K3-Empfehlung [1].1 (höchster Hebel gegen den Overshoot-Crash): gibt
    den Gründer-Agenten gestaffelte Startalter statt der synchronen Alter-0-
    Kohorte, die `Simulation.spawn_initial_population` baut.

    Ohne dieses Flag werden ALLE Gründer gleichzeitig reproduktionsfähig
    (`MIN_REPRODUCTION_AGE=60`, `agents/agent.py`) und altern synchron
    Richtung `ELDER_AGE=3500` — das erzeugt den in der K3-Analyse belegten
    synchronen Geburts-Boom (Peak-Pop ~44 bei Tick ~500), gefolgt vom
    synchronen Malthusianischen Crash. Gestaffelte Alter desynchronisieren,
    WANN jeder Gründer reproduktionsfähig wird bzw. an Alter stirbt.

    Verteilung: gleichverteilt über [0, 2000) Ticks. Begründung der
    Obergrenze (`agents/life_stage.py`: `CHILD_MAX=120`, `ADULT_MAX=3500`;
    `agents/agent.py`: `MIN_REPRODUCTION_AGE=60`, `ELDER_AGE=3500`): 2000
    bleibt komfortabel unter `ELDER_AGE`/`ADULT_MAX` (3500) — kein Gründer
    wird bei t=0 bereits post-reproduktiv geboren — und deckt sowohl das
    CHILD- als auch einen großen Teil des ADULT-Spektrums ab. 2000 ist zudem
    ~5x die in K3 gemessene Steady-State-Lebensdauer (~360–400 Ticks) —
    großzügig genug, um Reproduktions-/Sterbefenster der Kohorte über
    mehrere "Generationslängen" zu verteilen, statt nur den ersten
    Boom-Bust-Zyklus einmalig zu verschieben.

    Setzt SOWOHL `agent.age` (das tatsächliche, separat inkrementierte Feld,
    das `life_stage`/`can_reproduce` gated — `agent.py:924` `self.age += 1`,
    `agent.py:500-504` `can_reproduce`) ALS AUCH `agent.birth_tick = -age`
    (Aufrufzeitpunkt: `sim.tick == 0`, direkt nach dem `Simulation`-Bau, also
    ergibt `birth_tick = 0 - age = -age`), damit `tick - birth_tick`-
    Konsumenten (`m1_pilot._snapshot`s `mean_age`, `simulation.py:530`
    `avg_age`) von Anfang an konsistente Werte sehen. `birth_tick` ALLEIN zu
    setzen hätte KEINE Wirkung auf Fertilität/Lebensphase — beide gaten über
    das separate `self.age`-Feld, nicht über eine `tick - birth_tick`-
    Berechnung zur Laufzeit.

    Determinismus: zieht `len(sim.agents)` Werte aus dem GLOBALEN `random`-
    Modul, das `Simulation.__init__` bereits über `rng.seed_all(seed)`
    geseedet hat, BEVOR diese Funktion aufgerufen wird — gleicher Seed
    erzeugt daher dieselbe Alters-Verteilung. Dieser Zusatz-Draw verschiebt
    den nachfolgenden `random`-Strom gegenüber einem Lauf OHNE dieses Flag
    (ein zusätzlicher `random.randint`-Aufruf pro Agent, bevor der erste
    `sim.step()` irgendeinen `random`-Aufruf tätigt) — das ist ERWARTET:
    age-structured und nicht-age-structured sind unterschiedliche
    Bedingungen, kein Determinismus-Bug (gleicher Seed + gleiches Flag ⇒
    weiterhin bitidentisch).
    """
    for a in sim.agents:
        age = random.randint(0, 1999)
        a.age = age
        a.birth_tick = sim.tick - age


def _scale_regrowth(scale: float) -> None:
    """K3-Empfehlung [1].2 (Tragfähigkeit anheben): skaliert die Nahrungs-
    ZUFLUSS-Konstanten in `environment/resources.py` multiplikativ.

    Gewählte Konstanten: `FOOD_SCARCITY_FACTOR` (Default 0.50) und
    `MEAT_SCARCITY_FACTOR` (Default 0.55) — laut Datei-Kommentar
    (`resources.py:29-30`) "Pflanzenwachstum auf 50% reduziert" /
    "Fleischnachwuchs auf 55% reduziert", die zentralen linearen Faktoren auf
    `plant_gain`/`meat_gain` (`resources.py:519-521` bzw. `537-539`) BEVOR
    die Headroom-/Ceiling-Clamps greifen. Andere Konstanten in der Datei
    (`SCARCITY_CEILING_FACTOR`, `WIND_PLANT_LOSS`, …) sind Decken bzw.
    Verlust-Terme, kein Zufluss, und bleiben unangetastet.

    Call-time-Beleg (Projekt-Gotcha: def-time-Defaults sind nicht
    patchbar): `regrow_grid` (`resources.py:475`, der Live-Pfad — `world.py`
    importiert und ruft NUR `regrow_grid`, nie das ältere `regrow_cell`)
    liest beide Namen als freie Modul-Globals in ihrem Funktionskörper
    (kein Default-Argument, keine construction-time-Kopie). `world.py`
    importiert per `from ... import regrow_grid` nur die FUNKTION, nicht die
    Konstanten — `regrow_grid` schlägt die Namen also weiterhin im
    Globals-Dict von `resources.py` nach. Patchen von
    `resources_mod.FOOD_SCARCITY_FACTOR`/`MEAT_SCARCITY_FACTOR` NACH dem
    Import wirkt daher auf jeden folgenden `regrow_grid`-Aufruf
    (`world.py:334`, genau einmal pro `Simulation.step()`-Tick).
    """
    import artificial_society.environment.resources as resources_mod

    resources_mod.FOOD_SCARCITY_FACTOR = resources_mod.FOOD_SCARCITY_FACTOR * scale
    resources_mod.MEAT_SCARCITY_FACTOR = resources_mod.MEAT_SCARCITY_FACTOR * scale


def _set_min_food_per_capita(value: float) -> None:
    """K3-Empfehlung [1].3 (Dichte-Gate justieren): patcht
    `agents/agent.py`s `REPRODUCTION_MIN_FOOD_PER_CAPITA` (Default 6.0) —
    der dichteabhängige Fruchtbarkeits-Gate, der laut K3-Analyse auf dem
    40×30-Feld ggf. zu streng ist und die Erholung nach dem Crash blockiert.

    Call-time-Beleg: `_try_reproduce` (`agent.py:817-822`) liest den Namen
    als freien Modul-Global im Funktionskörper (`if
    self._local_food_per_capita(world, agents) <
    REPRODUCTION_MIN_FOOD_PER_CAPITA`) — kein Default-Argument, keine
    construction-time-Kopie. Patchen von
    `agent_mod.REPRODUCTION_MIN_FOOD_PER_CAPITA` NACH dem Import wirkt daher
    auf jeden folgenden `_try_reproduce`-Aufruf (einmal pro lebendem,
    weiblichem Agenten mit `can_reproduce()`, pro Tick — `agent.py:1483`).
    """
    import artificial_society.agents.agent as agent_mod

    agent_mod.REPRODUCTION_MIN_FOOD_PER_CAPITA = value


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


def _snapshot(
    sim, tick: int, ids_seen: set, respawn_count: int, deaths: int, births_cum: int
) -> dict:
    """Baut den Snapshot-/Final-Record.

    `ids_seen` wird IN-PLACE um die aktuell lebenden Agent-IDs erweitert
    (Aufrufer hält das Set über den ganzen Lauf) -- siehe Docstring oben zur
    Untererfassung von Geburten+Toden, die vollständig zwischen zwei
    Snapshots liegen. `respawn_count` ist der kumulative Zähler aus dem
    `emergency_respawn`-Wrapper (Anzahl Aufrufe, siehe `run_one`). `deaths`
    ist der exakte kumulative Zähler aus dem `remove_dead`-Wrapper (Review
    I-1, siehe `run_one` und Modul-Docstring). `births_cum` ist der exakte
    kumulative Zähler aus dem `spawn_child_from_parent`-Wrapper (Etappe 1,
    siehe `run_one` und Modul-Docstring) -- im Gegensatz zu `births` unten
    (Σ `children` NUR der aktuell lebenden Agenten) zählt er JEDE Geburt
    genau einmal, unabhängig davon, ob Elternteil oder Kind später sterben.
    """
    agents = sim.agents
    pop = len(agents)
    mean_energy = sum(a.energy for a in agents) / pop if pop else 0.0
    ids_seen.update(a.id for a in agents)
    mean_age = (
        sum(tick - getattr(a, "birth_tick", tick) for a in agents) / pop if pop else 0.0
    )
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
        # Etappe 1: exakter kumulativer Geburtenzähler (siehe _snapshot-Docstring).
        "births_cum": births_cum,
        # K2 Respawn-Mühlen-Instrumentierung (siehe Modul-Docstring).
        "respawns": respawn_count,
        "ids_seen_total": len(ids_seen),
        # Review I-1: exakter Todes-Zähler aus dem `remove_dead`-Wrapper.
        "deaths": deaths,
        "mean_age": round(mean_age, 2),
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
    min_pop: int = 8,
    respawn_count_cfg: int = 6,
    age_structured_founders: bool = False,
    regrow_scale: float = 1.0,
    min_food_per_capita: float = 6.0,
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

    # K2: Respawn-Boden konfigurierbar machen. Beide sind Modul-Globals,
    # call-time gelesen (simulation.py:639 `len(self.agents) < MIN_POPULATION`,
    # simulation.py:305 `range(RESPAWN_COUNT)` in `emergency_respawn`) --
    # gleiches Patch-Muster wie CHECKPOINT_INTERVAL oben.
    simulation_mod.MIN_POPULATION = min_pop
    simulation_mod.RESPAWN_COUNT = respawn_count_cfg

    # Etappe 1, Stellschraube 2/3 (siehe `_scale_regrowth`/
    # `_set_min_food_per_capita`-Docstrings für den call-time-Beleg). Bei den
    # Defaults (1.0 / 6.0) sind das reine No-ops -- 1.0x der Original-
    # Konstanten bzw. der bereits im Paket gesetzte Default -- also unverändertes
    # Verhalten gegenüber Läufen ohne diese Flags.
    _scale_regrowth(regrow_scale)
    _set_min_food_per_capita(min_food_per_capita)

    sim = Simulation(
        headless=True,
        load_checkpoint=False,
        physics_v2=True,
        grid_w=grid_w,
        grid_h=grid_h,
        initial_population=pop,
        seed=seed,
    )

    # Etappe 1, Stellschraube 1 (siehe `_age_structure_founders`-Docstring):
    # MUSS nach dem `Simulation`-Bau laufen -- operiert auf den bereits
    # konstruierten Gründer-Agenten-Instanzen, kein Modul-/Klassen-Patch.
    if age_structured_founders:
        _age_structure_founders(sim)

    # K2-Instrumentierung: `emergency_respawn` auf der INSTANZ (nicht der
    # Klasse) durch eine zählende Wrapper-Closure ersetzen, damit jeder Aufruf
    # (= ein Respawn-Batch von RESPAWN_COUNT Agenten) mitgezählt wird, ohne
    # Paket-Code anzufassen.
    _respawn_state = {"count": 0}
    _orig_respawn = sim.emergency_respawn

    def _counting_respawn():
        _respawn_state["count"] += 1
        return _orig_respawn()

    sim.emergency_respawn = _counting_respawn

    # Review I-1: `remove_dead` auf der INSTANZ (nicht der Klasse) durch eine
    # zählende Wrapper-Closure ersetzen. `remove_dead()` wird laut
    # `Simulation.step()` GENAU EINMAL pro Tick aufgerufen und ist der
    # ursachen-agnostische Todes-Aggregationspunkt (Spec B3) -- die Differenz
    # aus `len(sim.agents)` vor/nach dem (genau einmal ausgeführten)
    # Original-Aufruf ist die exakte Anzahl in diesem Tick entfernter Toter.
    _death_state = {"count": 0}
    _orig_remove_dead = sim.remove_dead

    def _counting_remove_dead():
        before = len(sim.agents)
        result = _orig_remove_dead()
        _death_state["count"] += before - len(sim.agents)
        return result

    sim.remove_dead = _counting_remove_dead

    # Etappe 1: `spawn_child_from_parent` auf der INSTANZ (nicht der Klasse)
    # durch eine zählende Wrapper-Closure ersetzen (gleiches Bound-Method-
    # Wrap-Muster wie oben). Laut `Simulation.step()` (simulation.py:617) ist
    # dies der EINZIGE Aufrufort für Geburten aus abgeschlossener Trächtigkeit
    # (ein Aufruf je `child_genes is not None`-Rückgabe von `agent.update`) --
    # disjunkt von `emergency_respawn` (Zufallshirn-Respawns, kein Aufruf von
    # `spawn_child_from_parent`). Der Zähler ist daher ein exakter, nicht
    # durch spätere Tode untererfasster Geburtenzähler (siehe `_snapshot`-
    # Docstring zu `births_cum` vs. `births`).
    _births_state = {"count": 0}
    _orig_spawn_child = sim.spawn_child_from_parent

    def _counting_spawn_child(parent, genes):
        _births_state["count"] += 1
        return _orig_spawn_child(parent, genes)

    sim.spawn_child_from_parent = _counting_spawn_child

    ids_seen: set = set()
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
            "min_pop": min_pop,
            "respawn_count": respawn_count_cfg,
            # Etappe 1 (K3-Empfehlung [1]): die drei Demografie-Knobs, siehe
            # Modul-Docstring "Etappe 1" + `_age_structure_founders`/
            # `_scale_regrowth`/`_set_min_food_per_capita`-Docstrings.
            "age_structured_founders": age_structured_founders,
            "regrow_scale": regrow_scale,
            "min_food_per_capita": min_food_per_capita,
            "patched": dict(conf["patched"], CHECKPOINT_INTERVAL=0),
        }
        f.write(json.dumps(meta) + "\n")
        for tick in range(ticks):
            sim.step()
            if (tick + 1) % snapshot_interval == 0:
                rec = _snapshot(
                    sim,
                    tick + 1,
                    ids_seen,
                    _respawn_state["count"],
                    _death_state["count"],
                    _births_state["count"],
                )
                rec["record"] = "snap"
                f.write(json.dumps(rec) + "\n")
                f.flush()
        final = _snapshot(
            sim,
            ticks,
            ids_seen,
            _respawn_state["count"],
            _death_state["count"],
            _births_state["count"],
        )
        final["record"] = "final"
        final["walltime_s"] = round(time.time() - t0, 1)
        final["turnover_final"] = round(final["ids_seen_total"] / max(final["pop"], 1), 2)
        f.write(json.dumps(final) + "\n")
    print(
        f"[{exp} seed{seed}] done pop={final['pop']} "
        f"tool_cut_ratio={final['tool_cut_ratio']} "
        f"discoveries={final['discoveries']} respawns={final['respawns']} "
        f"deaths={final['deaths']} births_cum={final['births_cum']} "
        f"mean_age={final['mean_age']} in {final['walltime_s']}s -> {path}"
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
    ap.add_argument(
        "--min-pop",
        type=int,
        default=8,
        help="Patcht simulation.MIN_POPULATION (Respawn-Schwelle, Default 8).",
    )
    ap.add_argument(
        "--respawn-count",
        type=int,
        default=6,
        help="Patcht simulation.RESPAWN_COUNT (Agenten je Respawn-Batch, Default 6).",
    )
    ap.add_argument(
        "--age-structured-founders",
        action="store_true",
        default=False,
        help=(
            "Etappe 1 (K3-Empfehlung [1].1): staffelt die Startalter der "
            "Gründer-Agenten gleichverteilt über [0, 2000) Ticks statt der "
            "synchronen Alter-0-Kohorte, um den Gründer-Overshoot-Crash zu "
            "entsynchronisieren (Details: siehe _age_structure_founders)."
        ),
    )
    ap.add_argument(
        "--regrow-scale",
        type=float,
        default=1.0,
        help=(
            "Etappe 1 (K3-Empfehlung [1].2): multiplikative Skalierung von "
            "environment.resources.FOOD_SCARCITY_FACTOR/MEAT_SCARCITY_FACTOR "
            "(Default 1.0 = unverändert; Details: siehe _scale_regrowth)."
        ),
    )
    ap.add_argument(
        "--min-food-per-capita",
        type=float,
        default=6.0,
        help=(
            "Etappe 1 (K3-Empfehlung [1].3): patcht agents.agent."
            "REPRODUCTION_MIN_FOOD_PER_CAPITA (Default 6.0 = Paket-Default, "
            "unverändert; Details: siehe _set_min_food_per_capita)."
        ),
    )
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
        args.min_pop,
        args.respawn_count,
        args.age_structured_founders,
        args.regrow_scale,
        args.min_food_per_capita,
    )


if __name__ == "__main__":
    main()
