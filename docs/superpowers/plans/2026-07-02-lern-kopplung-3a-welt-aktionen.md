# Lern-Kopplung Plan 3a — Welt & verkörperte Aktionen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Physik-v2-Weltschicht bauen: sparse `ObjectLayer` (`world.objects`), biome-gebundenes Spawning, Kadaver-Umleitung mit exakter Massen-Erhaltung, Body/Hands-Embodiment (Default-strength 0.5) und die verkörperten Aktionen `do_grasp/do_release/do_strike/do_cut/do_eat` — vollständig testbar ohne jede Gehirn-Änderung (geskriptete Test-Treiber rufen `do_*` direkt).

**Architecture:** `physics_v2` ist eine additive Schicht hinter einem `Simulation`-Kwarg (Default `False` = byte-gleiches v1-Verhalten). Die Objekt-Schicht ist eine sparse Registry (`dict[pos, list[PhysObject]]`) als Schwester-Attribut `world.objects` neben `world.F/S` — **niemals** in den Zell-Arrays (GPU-Garantie). Der Pro-Tick-Anteil (Verwesung, Spawning) läuft als registriertes System `"physics_v2_objects"` über `systems/registry.py`; Umschaltpunkte im v1-Code (Todes-Routing, Loot, Forage-Fleischpools) sind einzelne `if physics_v2:`-Verzweigungen. Erhaltung wird über einen Massen-Ledger (`spawned/from_carcass/eaten/decayed`) bewiesen (Fuzzer, 50 Seeds × 200 Aktionen).

**Tech Stack:** Python 3.9, numpy, pytest; `ruff` (line-length 100); Spec: `docs/superpowers/specs/2026-07-02-lern-kopplung-design.md` (Abschnitte A, B1–B6, D1, D2, D4-3a).

## Global Constraints

Jeder Task erbt implizit alle folgenden Regeln:

1. **Flag aus = byte-gleich:** `physics_v2=False` (Default) ändert KEIN Verhalten. Der Golden-Test
   (`tests/test_regression_golden.py`) bleibt grün — **niemals das Golden regenerieren**, niemals
   einen Determinismus-Test anpassen. Bei Flag aus darf kein zusätzlicher RNG-Draw entstehen.
2. **Realitäts-Gate:** JEDE neue Konstante braucht einen `cal()`-Eintrag (Anker + Quelle) in der
   Kalibrierungstabelle UND eine regenerierte `docs/physics/kalibrierung.md`
   (`../venv/bin/python scripts/gen_kalibrierung.py`, Artefakt mit-committen). Der Gate-Test
   `tests/environment/physics/test_reality_gate.py` erzwingt beides.
3. **Exakte Erhaltung:** Masse und Energie werden exakt erhalten (Ledger-Invariante:
   `bodenmasse + händemasse_lebender + ledger[eaten] + ledger[decayed] == ledger[spawned] + ledger[from_carcass]`,
   `math.isclose` mit `rel_tol=1e-9, abs_tol=1e-9`). Tode münzen Energie in v1-Pools ODER
   v2-Objekte — nie beides, nie zusätzlich als Loot.
4. **Keine Änderungen an `artificial_society/agents/brain.py` und keinerlei Änderungen an
   Belohnungslogik** (Reward-Terme, `reward +=`-Zeilen) — das ist Plan 3b. In 3a erlaubt sind nur
   Energie-/Mechanik-Umschaltungen (Loot, Hamilton-Energie-Umverteilung, Mechanik-Blöcke aus B6).
5. **Vor jedem Commit:** `../venv/bin/ruff check --fix . && ../venv/bin/ruff format .`
   (die PostToolUse-Auto-Hooks feuern in Subagenten NICHT). Danach ggf. geänderte Dateien erneut stagen.
6. **Vor jedem Test-Lauf, der eine `Simulation` konstruiert:** `rm -f checkpoint.pkl`
   (ein stales root-`checkpoint.pkl` wird sonst auto-geladen). In Tests außerdem immer
   `load_checkpoint=False` übergeben, außer der Test testet das Checkpointing selbst.
7. **Konstanten verbatim aus der Spec** übernehmen (Werte stehen in jedem Task explizit);
   Feinabstimmung nur im Pilot, nie ad hoc.
8. **Arbeitsverzeichnis:** ausschließlich `/Users/moritzbecker/projekt/as-lern-kopplung-3a`
   (Worktree, Branch `feat/lern-kopplung-3a`). venv liegt eine Ebene höher:
   Python = `../venv/bin/python`, ruff = `../venv/bin/ruff`.
   Testaufruf immer: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest …`.
9. **Stil:** `from __future__ import annotations`, Typannotationen, deutsche Docstrings wie in
   `artificial_society/environment/physics/` üblich. Zeilenlänge ≤ 100.

## Datei-Landkarte (was entsteht / was sich ändert)

| Datei | Verantwortung |
|---|---|
| `artificial_society/environment/phys_objects.py` (NEU) | `ObjectLayer` (sparse Registry, Ledger, per-Welt `DiscoveryV2`), Spawning (`seed_initial`, `tick_spawn`), Verwesung (`tick_decay`), `CALIBRATED_SPAWN_PARAMS` |
| `artificial_society/environment/physics/actions.py` (NEU) | Kalibrierte Aktions-/Kopplungs-Konstanten (`CALIBRATED_ACTION_PARAMS`), `ActionResult`, `do_grasp/do_release/do_strike/do_cut/do_eat`, `enforce_carry_budget`, `blade_factor` |
| `artificial_society/environment/physics/calibration.py` | neue Kinds `"spawn"`, `"action"`; Doku-Renderer importiert die neuen Module |
| `artificial_society/environment/physics/body.py` | `BODY_MASS_DEFAULT_KG = 70.0` (+ cal-Eintrag `body_mass`) |
| `artificial_society/environment/physics/materials_v2.py` | Kadaver-Rekalibrierung: nutrition 0.30→**0.14**, toxicity 0.10→**0.02** |
| `artificial_society/environment/physics/discovery.py` + `physics/__init__.py` | Entfernung des Modul-Singletons `DISCOVERY_V2` |
| `artificial_society/world.py` | `world.objects = ObjectLayer(...)` in `__init__` + `ensure_array_storage()`-Migration |
| `artificial_society/systems/physics_v2.py` (NEU) | registriertes System `"physics_v2_objects"` (Verwesung + Spawning pro Tick) |
| `simulation.py` | `physics_v2`-Kwarg, Embodiment beim Spawn, Kadaver-Umleitung in `remove_dead()`, Pre-Filter-Aufhebung im v2-Modus, Hamilton aus im v2-Modus, Checkpoint-Payload + Guard |
| `artificial_society/agents/agent.py` | Felder `physics_v2/body/hands`, `attach_body()`, Überlast-Drop + carry/rest pro Tick, Loot=0, B6-Mechanik-Schalter (Forage-Fleischpools, Invention, Kochen, Goal-Stack) |
| `artificial_society/systems/world_objects.py` | → `archive/world_objects.py` (totes Alt-Modul) |
| `tests/…` | siehe Tasks; neue Dateien unter `tests/environment/`, `tests/environment/physics/`, `tests/` |

**Nicht in diesem Plan (bewusst, = Plan 3b):** Abschnitt C komplett (perception_v2, Attention-Encoder,
Aktionsköpfe, PPO, Causal-Model, Neugier, `strength`-Gen, `CHECKPOINT_FORMAT_VERSION`, Planner-Deaktivierung),
D3, sowie die **Reward-Spalte** aus B6 (Forage-Event-Bonus, Attack-Reward-Rückgabe 0.5/`agent.py:1179`,
Territorium-, Koop-, Sprach-, Trade-Reward-Terme) — mit den zwei Ausnahmen, die die Spec explizit 3a
zuordnet: Attack-**Loot** = 0 (Energie-Mechanik, B3) und Hamilton-Energie-Umverteilung aus
(`simulation.py`-Zeile der Architekturtabelle).

---

### Task 1: DiscoveryV2-Modul-Singleton entfernen

Das ungenutzte Modul-Singleton `DISCOVERY_V2` (`discovery.py:52`) lädt zu geteiltem Zustand zwischen
Welten ein. Es hat keine Konsumenten (verifiziert: nur Definition + Re-Export in
`physics/__init__.py`). DiscoveryV2 wird künftig pro Welt instanziert (`ObjectLayer.discovery`, Task 4).

**Files:**
- Modify: `artificial_society/environment/physics/discovery.py:52` (Zeile löschen)
- Modify: `artificial_society/environment/physics/__init__.py:18,44` (Import/Export anpassen)
- Test: `tests/environment/physics/test_discovery_v2.py` (Test ergänzen)

**Interfaces:**
- Consumes: `DiscoveryV2` (Klasse, bleibt unverändert)
- Produces: `artificial_society.environment.physics` exportiert NUR noch die Klasse `DiscoveryV2`,
  kein `DISCOVERY_V2`-Objekt mehr. Task 4 instanziert `DiscoveryV2()` pro `ObjectLayer`.

- [ ] **Step 1: Failing Test schreiben** — in `tests/environment/physics/test_discovery_v2.py` anhängen:

```python
def test_kein_modul_singleton_mehr():
    """Spec B1: DiscoveryV2 lebt pro Welt (ObjectLayer.discovery), nicht als
    Modul-Singleton — geteilter Zustand zwischen Simulationen ist verboten."""
    import artificial_society.environment.physics as physics_pkg
    from artificial_society.environment.physics import discovery as discovery_mod

    assert not hasattr(discovery_mod, "DISCOVERY_V2")
    assert not hasattr(physics_pkg, "DISCOVERY_V2")
    assert "DISCOVERY_V2" not in physics_pkg.__all__
```

- [ ] **Step 2: Test laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_discovery_v2.py::test_kein_modul_singleton_mehr -v`
Expected: FAIL mit `AssertionError` (erste Assertion: `hasattr` ist True).

- [ ] **Step 3: Implementieren**

In `artificial_society/environment/physics/discovery.py` die letzte Zeile löschen:

```python
DISCOVERY_V2 = DiscoveryV2()
```

In `artificial_society/environment/physics/__init__.py` Zeile 18 ändern von
`from .discovery import DISCOVERY_V2, DiscoveryV2` zu:

```python
from .discovery import DiscoveryV2
```

und in `__all__` die Zeile `"DISCOVERY_V2",` löschen (Zeile 44).

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: alle grün (auch die bestehenden Discovery-Tests — sie nutzen nur die Klasse).

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/physics/discovery.py artificial_society/environment/physics/__init__.py tests/environment/physics/test_discovery_v2.py
git commit -m "refactor(physics): DISCOVERY_V2-Modul-Singleton entfernt (DiscoveryV2 wird pro Welt instanziert)"
```

---

### Task 2: Kalibrierung — neue Kinds `"spawn"`/`"action"` + Körpermasse-Default

Das Gate kennt heute nur `("dim", "material", "process", "body")`. Spawn-Parameter (B2) und
Aktions-/Kopplungs-Konstanten (B4/B5) brauchen eigene Kinds, sonst weist `cal()` sie ab bzw. die
Orphan-Prüfung (kind `material` prüft strikt gegen `MATERIALS_V2`) würde sie als verwaist melden.
Außerdem: `BODY_MASS_DEFAULT_KG = 70.0` als kalibrierter Default für das Embodiment (Task 12) und
die Kadaver-Masse (Task 14) — der Spec-Anker „70-kg-Kadaver ≈ 1250 Sim-Energie“ (B5) setzt genau
diesen Wert voraus.

**Files:**
- Modify: `artificial_society/environment/physics/calibration.py:14` (`VALID_KINDS`) und `:127-132` (`_KIND_TITLES`)
- Modify: `artificial_society/environment/physics/body.py:27-28` (Konstante + `CALIBRATED_BODY_PARAMS`) und Datei-Ende (cal-Eintrag)
- Modify: `docs/physics/kalibrierung.md` (regeneriert)
- Test: `tests/environment/physics/test_reality_gate.py` (Test ergänzen)

**Interfaces:**
- Consumes: `cal(kind, name, anchor, source)`, `CALIBRATION`, `VALID_KINDS` aus `calibration.py`
- Produces: `VALID_KINDS == ("dim", "material", "process", "body", "spawn", "action")`;
  `body.BODY_MASS_DEFAULT_KG: float = 70.0`; `CALIBRATED_BODY_PARAMS` enthält `"body_mass"`.
  Tasks 3/6 registrieren Einträge unter `"action"`/`"spawn"`; Tasks 12/14 nutzen `BODY_MASS_DEFAULT_KG`.

- [ ] **Step 1: Failing Tests schreiben** — in `tests/environment/physics/test_reality_gate.py` anhängen:

```python
def test_neue_kinds_spawn_und_action_sind_gueltig():
    """B2/B4: Spawn- und Aktions-Konstanten bekommen eigene Kalibrierungs-Kinds."""
    from artificial_society.environment.physics.calibration import VALID_KINDS, cal

    assert "spawn" in VALID_KINDS
    assert "action" in VALID_KINDS
    # cal() akzeptiert die neuen Kinds (Aufräumen: Test-Eintrag wieder entfernen,
    # sonst schlägt die Orphan-Prüfung späterer Tasks im selben Prozess an).
    try:
        cal("spawn", "_test_eintrag", "Test-Anker", "Test-Quelle")
        assert ("spawn", "_test_eintrag") in CALIBRATION
    finally:
        CALIBRATION.pop(("spawn", "_test_eintrag"), None)


def test_body_mass_default_ist_kalibriert():
    from artificial_society.environment.physics.body import (
        BODY_MASS_DEFAULT_KG,
        CALIBRATED_BODY_PARAMS,
    )

    assert BODY_MASS_DEFAULT_KG == 70.0
    assert "body_mass" in CALIBRATED_BODY_PARAMS
    _assert_calibrated("body", "body_mass")
```

- [ ] **Step 2: Test laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_reality_gate.py -v -k "neue_kinds or body_mass"`
Expected: FAIL — `assert "spawn" in VALID_KINDS` bzw. `ImportError: cannot import name 'BODY_MASS_DEFAULT_KG'`.

- [ ] **Step 3: Implementieren**

`calibration.py` Zeile 14 ersetzen:

```python
VALID_KINDS = ("dim", "material", "process", "body", "spawn", "action")
```

`calibration.py`, `_KIND_TITLES` (Zeile 127) ersetzen durch:

```python
_KIND_TITLES = {
    "dim": "Eigenschafts-Dimensionen",
    "material": "Startmaterialien",
    "process": "Prozesse",
    "body": "Körper-Parameter",
    "spawn": "Spawn-Parameter (Vorkommen)",
    "action": "Aktions- & Kopplungs-Parameter",
}
```

`body.py`: nach Zeile 25 (`CARRY_FATIGUE_PER_TICK_AT_CAPACITY = ...`) einfügen:

```python
BODY_MASS_DEFAULT_KG = 70.0  # Default-Körpermasse fürs Embodiment (Plan 3a); Gen-Kopplung = Plan 3b
```

`body.py` Zeile 28 ersetzen:

```python
CALIBRATED_BODY_PARAMS = ("carry_capacity", "strike_energy", "fatigue", "hands", "body_mass")
```

`body.py`, am Datei-Ende (nach dem `cal("body", "hands", ...)`-Block) anhängen:

```python
cal(
    "body",
    "body_mass",
    "Default-Körpermasse 70 kg (erwachsener Mensch, Referenzperson); bestimmt Tragkapazität "
    "(~30 % davon) und die Kadaver-Masse beim Tod (Spec B3/B5: 70-kg-Kadaver ≈ 1250 Sim-Energie "
    "≈ 5× MAX_ENERGY)",
    "Anthropometrie: ICRP-Referenzperson ~70 kg",
)
```

Doku regenerieren: `../venv/bin/python scripts/gen_kalibrierung.py`
(erzeugt die zwei neuen — noch leeren — Abschnitts-Überschriften und den `body_mass`-Eintrag).

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_reality_gate.py -q`
Expected: alle grün (inkl. Doc-Sync-Test).

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/physics/calibration.py artificial_society/environment/physics/body.py docs/physics/kalibrierung.md tests/environment/physics/test_reality_gate.py
git commit -m "feat(physics): Kalibrierungs-Kinds spawn/action + BODY_MASS_DEFAULT_KG=70 (Gate-pflichtig)"
```

---

### Task 3: `actions.py` — kalibrierte Konstanten + Gate-Anbindung `"action"`

Alle B4/B5-Konstanten entstehen VOR den Funktionen, damit Verwesung (Task 7) und die `do_*`-Tasks
(8–11) sie konsumieren können. Jede Konstante bekommt einen `cal("action", ...)`-Eintrag; der
Gate-Test wird um die `"action"`-Quelle inkl. Orphan-Prüfung erweitert.

**Arbeits-Metabolik-Formel (Spec B5, Rev. 4):**
`energy -= joules / MUSCLE_EFFICIENCY / 4184 * SIM_ENERGY_PER_KCAL`, d. h.
`WORK_SIM_ENERGY_PER_JOULE = SIM_ENERGY_PER_KCAL / (MUSCLE_EFFICIENCY * 4184)`
(200 Schläge × 45 J → ≈ 0.275 ≈ Spec-Anker „200 Schläge ≈ 0.3 Sim-Energie“).
Ein Test hält den Anker fest.

**Files:**
- Create: `artificial_society/environment/physics/actions.py`
- Modify: `artificial_society/environment/physics/calibration.py:146` (Renderer-Import)
- Modify: `docs/physics/kalibrierung.md` (regeneriert)
- Test: `tests/environment/physics/test_reality_gate.py` (erweitern), `tests/environment/physics/test_actions.py` (neu)

**Interfaces:**
- Consumes: `cal` aus `.calibration`; `TICKS_PER_DAY = 240` aus `artificial_society.environment.daynight`;
  `MEAT_ENERGY = 45.0` aus `artificial_society.agents.agent` (nur im Test);
  `MATERIALS_V2`, `IDX2` (nur im Test)
- Produces (Konstanten, von Tasks 7–11 konsumiert — Namen exakt so):
  `SIM_ENERGY_PER_KCAL = 0.032`, `KCAL_PER_KG_PER_NUTRITION = 4000.0`, `MUSCLE_EFFICIENCY = 0.25`,
  `JOULE_PER_KCAL = 4184.0`, `WORK_SIM_ENERGY_PER_JOULE` (abgeleitet), `V_MAX_STRIKE = 14.0`,
  `BLADE_MASS_REF = 0.15`, `BLADE_HANDLE_MAX = 1.0`, `CUT_WORK_J_BASE = 15.0`,
  `CUT_WORK_J_PER_EFFORT = 35.0`, `BITE_MASS_KG = 0.3`, `MIN_NUTRITION_EDIBLE = 0.02`,
  `TOX_DAMAGE_PER_KG = 20.0`, `DECAY_HALF_LIFE_DAYS = 10.0`, `DECAY_RATE` (≈ 2.888e-4),
  `TOX_SPOILAGE_PER_TICK = 5e-4`, `TOX_SPOILAGE_CAP = 0.6`, `DECAY_MOISTURE_MIN = 0.5`,
  `CALIBRATED_ACTION_PARAMS: tuple[str, ...]`

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/environment/physics/test_actions.py`:

```python
"""Verkörperte Aktions-Mechanik (Plan 3a, Spec B4/B5) — Konstanten, Kopplung, do_*-Funktionen."""

from __future__ import annotations

import math

from artificial_society.agents.agent import MEAT_ENERGY
from artificial_society.environment.physics.actions import (
    DECAY_RATE,
    KCAL_PER_KG_PER_NUTRITION,
    SIM_ENERGY_PER_KCAL,
    TOX_SPOILAGE_CAP,
    TOX_SPOILAGE_PER_TICK,
    WORK_SIM_ENERGY_PER_JOULE,
)
from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.props import IDX2


def test_energie_kopplung_produkt_anker():
    """B5: nutrition-Konvention × SIM_ENERGY_PER_KCAL ≙ MEAT_ENERGY ± 1 (Produkt-Test —
    zwei Konstanten, eine Bilanz, kein stilles Driften)."""
    nutrition_raw_meat = float(MATERIALS_V2["raw_meat"][IDX2["nutrition"]])
    sim_energy_pro_kg = nutrition_raw_meat * KCAL_PER_KG_PER_NUTRITION * SIM_ENERGY_PER_KCAL
    assert abs(sim_energy_pro_kg - MEAT_ENERGY) <= 1.0  # 0.35*4000*0.032 = 44.8 ≈ 45


def test_arbeits_metabolik_anker_200_schlaege():
    """B5-Anker: 200 kräftige Schläge (45 J) kosten ≈ 0.3 Sim-Energie (bewusst klein —
    der reale Begrenzer ist die Ermüdung)."""
    kosten = 200 * 45.0 * WORK_SIM_ENERGY_PER_JOULE
    assert 0.2 < kosten < 0.4


def test_verwesungs_konstanten_anker():
    """B3.4: Halbwertszeit 10 Tage à 240 Ticks; Toxin-Kappe 0.6 nach ~5 Tagen erreicht."""
    assert math.isclose(DECAY_RATE, math.log(2.0) / (10 * 240), rel_tol=1e-12)
    assert TOX_SPOILAGE_PER_TICK == 5e-4
    assert TOX_SPOILAGE_CAP == 0.6
    # frisches Fleisch (0.02) erreicht die Kappe nach ≈ 1160 Ticks ≈ 5 Tagen
    ticks_bis_kappe = (TOX_SPOILAGE_CAP - 0.02) / TOX_SPOILAGE_PER_TICK
    assert 4 * 240 <= ticks_bis_kappe <= 6 * 240
```

In `tests/environment/physics/test_reality_gate.py`: Import ergänzen und zwei Tests anhängen.
Oben bei den Imports:

```python
from artificial_society.environment.physics.actions import CALIBRATED_ACTION_PARAMS
```

Am Datei-Ende:

```python
def test_every_action_param_is_calibrated():
    for name in CALIBRATED_ACTION_PARAMS:
        _assert_calibrated("action", name)


def test_no_orphan_action_entries():
    for kind, name in CALIBRATION:
        if kind == "action":
            assert name in CALIBRATED_ACTION_PARAMS, f"verwaister action-Eintrag: {name}"
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py tests/environment/physics/test_reality_gate.py -q`
Expected: FAIL/ERROR mit `ModuleNotFoundError: No module named 'artificial_society.environment.physics.actions'`.

- [ ] **Step 3: Implementieren** — neue Datei `artificial_society/environment/physics/actions.py`:

```python
"""Verkörperte Aktions-Mechanik der Physik v2 (Plan 3a, Spec B4/B5).

Dieses Modul bindet Body/Hands/Prozesse/ObjectLayer — es kennt weder Gehirn
noch Belohnung. Alle Konstanten sind real geankert (cal-Einträge unten,
Gate-pflichtig über CALIBRATED_ACTION_PARAMS). Energie-Konvention: nutrition
ist kcal/100 g ÷ 400 (raw_meat 0.35 ≙ 140 kcal/100 g), gekoppelt an die
v1-Energieskala über SIM_ENERGY_PER_KCAL (1-kg-Fleischmahlzeit ≙ MEAT_ENERGY 45).
"""

from __future__ import annotations

import math

from artificial_society.environment.daynight import TICKS_PER_DAY

from .calibration import cal

# --- Energie-Kopplung kcal ↔ Sim-Energie (B5) --------------------------------
SIM_ENERGY_PER_KCAL = 0.032
KCAL_PER_KG_PER_NUTRITION = 4000.0  # nutrition 1.0 ≙ 400 kcal/100 g = 4000 kcal/kg
MUSCLE_EFFICIENCY = 0.25
JOULE_PER_KCAL = 4184.0
# Spec B5 (Rev. 4): energy -= joules / MUSCLE_EFFICIENCY / 4184 * SIM_ENERGY_PER_KCAL
# — 200 Schläge à 45 J ≈ 0.28 Sim-Energie (Anker-Test in test_actions.py hält das fest).
WORK_SIM_ENERGY_PER_JOULE = SIM_ENERGY_PER_KCAL / (MUSCLE_EFFICIENCY * JOULE_PER_KCAL)

# --- Schlag (B4) --------------------------------------------------------------
V_MAX_STRIKE = 14.0  # m/s — ½·m·v² kappt die gelieferte Energie leichter Schläger

# --- Schneiden (B4) -----------------------------------------------------------
BLADE_MASS_REF = 0.15  # kg — brauchbare Schlacht-Abschläge ≥ ~150 g
BLADE_HANDLE_MAX = 1.0  # kg — einhändig führbares Schneidwerkzeug ≤ ~1 kg
CUT_WORK_J_BASE = 15.0
CUT_WORK_J_PER_EFFORT = 35.0  # CUT_WORK_J = 15 + 35·effort

# --- Essen (B4/B5) --------------------------------------------------------------
BITE_MASS_KG = 0.3
MIN_NUTRITION_EDIBLE = 0.02  # darunter wirkungslos (Steinbeißen = No-op)
TOX_DAMAGE_PER_KG = 20.0

# --- Verwesung (B3.4; Vollzug in phys_objects.tick_decay) ----------------------
DECAY_HALF_LIFE_DAYS = 10.0
DECAY_RATE = math.log(2.0) / (DECAY_HALF_LIFE_DAYS * TICKS_PER_DAY)  # ≈ 2.888e-4
TOX_SPOILAGE_PER_TICK = 5e-4
TOX_SPOILAGE_CAP = 0.6
DECAY_MOISTURE_MIN = 0.5  # nur feuchte, nahrhafte Stoffe verwesen (Weichgewebe)

# Vom Realitäts-Gate geprüfte Aktions-/Kopplungs-Parameter (kind "action").
CALIBRATED_ACTION_PARAMS = (
    "sim_energy_per_kcal",
    "muscle_efficiency",
    "v_max_strike",
    "blade_mass_factor",
    "cut_work",
    "bite",
    "toxin_damage",
    "decay",
    "spoilage",
)

cal(
    "action",
    "sim_energy_per_kcal",
    "Kopplung kcal↔Sim-Energie: 0.032 Sim-Energie/kcal — eine 1-kg-Fleischmahlzeit "
    "(0.35·4000 = 1400 kcal) ergibt ≈ 45 ≙ v1 MEAT_ENERGY; Gate prüft das PRODUKT "
    "nutrition-Konvention × Kopplung (± 1)",
    "USDA-Nährwerttabellen + v1-Energieökonomie (MEAT_ENERGY 45, MAX_ENERGY 240)",
)
cal(
    "action",
    "muscle_efficiency",
    "Brutto-Wirkungsgrad Skelettmuskel 0.25 (20–25 %); mechanische Arbeit kostet "
    "joules/0.25/4184 kcal ≙ ×0.032 Sim-Energie (200 Schläge à 45 J ≈ 0.3 Sim-Energie — "
    "bewusst klein, der reale Begrenzer ist die Ermüdung)",
    "Arbeitsphysiologie: Wirkungsgrad Muskelarbeit",
)
cal(
    "action",
    "v_max_strike",
    "Maximale Schlag-Endgeschwindigkeit 14 m/s; gelieferte Energie ≤ ½·m·v² — ein "
    "0.05-kg-Kiesel liefert damit max ~4.9 J (kein Flint-Knacken über Kiesel-Exploit), "
    "ein 0.5–2-kg-Schlagstein die vollen 49–50 J",
    "Biomechanik Hammerschlag/Knapping: Endgeschwindigkeit 10–15 m/s",
)
cal(
    "action",
    "blade_mass_factor",
    "Klingen-Massen-Faktor beidseitig begrenzt: clamp(sqrt(m/0.15), 0.2, 1) · "
    "clamp((2−m)/1, 0.2, 1) — brauchbare Abschläge ≥ ~150 g (BLADE_MASS_REF), einhändig "
    "führbar ≤ ~1 kg (BLADE_HANDLE_MAX); 20-g-Splitter und 4-kg-Brocken schneiden nur "
    "mit Faktor ≤ 0.4 bzw. 0.2 (kein 8-kg-Skalpell)",
    "Ethnographie/experimentelle Archäologie der Lithik: Handhabbarkeit von Schneidwerkzeug",
)
cal(
    "action",
    "cut_work",
    "Schneidearbeit CUT_WORK_J = 15 + 35·effort Joule pro Schneidevorgang, über den "
    "Ermüdungspfad (FATIGUE_PER_JOULE) und die Arbeits-Metabolik verbucht; der "
    "Schnitt-Ertrag skaliert mit Effort: yield × (0.5 + 0.5·effort)",
    "Arbeitsphysiologie: repetitives Schneiden/Zerwirken — Zerlegegeschwindigkeit "
    "skaliert mit aufgebrachter Kraft",
)
cal(
    "action",
    "bite",
    "Ein Biss pro Tick: bite = min(0.3 kg, Restmasse); Objekte mit nutrition ≤ 0.02 sind "
    "wirkungslos (Steinbeißen = No-op)",
    "Ernährungsphysiologie: Bolus-/Bissgrößen, Größenordnung Mahlzeit ~0.3–1 kg",
)
cal(
    "action",
    "toxin_damage",
    "Toxin-Schaden 20 Health/kg·toxicity: 0.3 kg stark toxischen Materials (0.8) ≈ 5 Health "
    "≙ spürbar, wiederholt tödlich; verdorbenes Fleisch (0.6) ≈ 3.6 Health/Biss",
    "Lebensmittelhygiene/Toxikologie: Dosis-Wirkung roher, verdorbener Tierprodukte",
)
cal(
    "action",
    "decay",
    "Verwesung: mass·(1−λ) und nutrition·(1−λ) pro Tick mit λ = ln(2)/(10 Tage · 240 Ticks) "
    "≈ 2.89e-4 (Weichgewebe-Halbwertszeit temperiert ~7–14 Tage); wirkt nur auf feuchte, "
    "nahrhafte Stoffe (moisture ≥ 0.5, nutrition > 0 — Stein/Holz verwesen nicht)",
    "Forensische Taphonomie: Weichgewebe-Dekomposition",
)
cal(
    "action",
    "spoilage",
    "Verderb: toxicity += 5e-4 pro Tick, gekappt bei 0.6 — rohes Fleisch wird bei "
    "Umgebungstemperatur in ~3–5 Tagen gefährlich (Kappe nach ~5 Tagen ≙ 1200 Ticks)",
    "Lebensmittelhygiene: Verderb roher Tierprodukte ungekühlt",
)
```

In `calibration.py`, `render_markdown()` (Zeile 146), den Schwester-Import erweitern von
`from . import body, materials_v2, processes  # noqa: F401` zu:

```python
    from . import actions, body, materials_v2, processes  # noqa: F401
```

Doku regenerieren: `../venv/bin/python scripts/gen_kalibrierung.py`

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/physics/actions.py artificial_society/environment/physics/calibration.py docs/physics/kalibrierung.md tests/environment/physics/test_actions.py tests/environment/physics/test_reality_gate.py
git commit -m "feat(physics): actions.py Kalibrierungs-Konstanten (B4/B5) + Gate-Quelle 'action'"
```

---

### Task 4: `ObjectLayer` — sparse Objekt-Registry mit Ledger und per-Welt-Discovery

**Files:**
- Create: `artificial_society/environment/phys_objects.py`
- Test: `tests/environment/test_phys_objects.py` (neu)

**Interfaces:**
- Consumes: `PhysObject`, `make_object` aus `physics.objects`; `DiscoveryV2` aus `physics.discovery`;
  `N_PROPS_V2` aus `physics.props`
- Produces (von Tasks 5–19 konsumiert — Signaturen exakt so):
  - `ObjectLayer(width: int, height: int, rng=None)` — `rng` duck-typed (braucht `.random()`,
    `.uniform()`, `.randrange()`); `None` ⇒ stdlib-`random`-Modul (global via `seed_all` geseedet)
  - `layer.add(obj: PhysObject, pos: tuple[int, int], source: str | None = None) -> None`
    (validiert 13 Props ∈ [0,1], `mass > 0`, pos in bounds, Objekt nicht schon im Layer;
    `source ∈ {None, "spawned", "from_carcass"}` schreibt den Ledger; `None` = ledger-neutral)
  - `layer.remove(obj: PhysObject) -> None` (Identität, nicht Wert; `KeyError` wenn nicht im Layer)
  - `layer.position_of(obj: PhysObject) -> tuple[int, int] | None`
  - `layer.objects_at(pos) -> list[PhysObject]` (Kopie)
  - `layer.objects_near(pos, radius: int) -> list[tuple[PhysObject, tuple[int, int]]]` (Chebyshev)
  - `layer.total_mass() -> float` (NUR Boden)
  - `layer.all_objects() -> Iterator[tuple[PhysObject, tuple[int, int]]]`
  - `layer.conservation_terms(held_mass_kg: float = 0.0) -> tuple[float, float]` (lhs, rhs der Invariante)
  - `layer.ledger: dict[str, float]` mit Keys `"spawned"/"from_carcass"/"eaten"/"decayed"`
  - `layer.discovery: DiscoveryV2` (PRO Welt)
  - picklebar (`__getstate__`/`__setstate__` bauen die id-Rückwärts-Map neu auf; ein
    Modul-Default-rng — das `random`-MODUL ist nicht picklebar — wird beim Pickling auf
    `None` gesetzt und beim Laden wieder ans Modul gebunden; eine explizite
    `random.Random`-Instanz bleibt mitsamt Zustand erhalten). Das ist der
    Checkpoint-Pfad: `_save_checkpoint` pickelt `world` inkl. `world.objects`
    (`simulation.py:343`)

- [ ] **Step 1: Failing Tests schreiben** — neue Datei `tests/environment/test_phys_objects.py`:

```python
"""ObjectLayer (Plan 3a, Spec B1): sparse Registry, Ledger, per-Welt-Discovery."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from artificial_society.environment.phys_objects import ObjectLayer
from artificial_society.environment.physics.objects import PhysObject, make_object
from artificial_society.environment.physics.props import pv


def _layer() -> ObjectLayer:
    return ObjectLayer(10, 8)


def test_add_und_objects_at():
    layer = _layer()
    a = make_object("granite", 2.0)
    layer.add(a, (3, 4))
    assert layer.objects_at((3, 4)) == [a]
    assert layer.objects_at((0, 0)) == []
    assert layer.position_of(a) == (3, 4)
    assert layer.total_mass() == 2.0


def test_add_validiert_props_masse_und_bounds():
    layer = _layer()
    with pytest.raises(ValueError):  # Props außerhalb [0,1]
        layer.add(PhysObject(props=pv(hardness=1.5), mass=1.0), (0, 0))
    with pytest.raises(ValueError):  # außerhalb des Grids
        layer.add(make_object("granite", 1.0), (10, 0))
    obj = make_object("granite", 1.0)
    layer.add(obj, (1, 1))
    with pytest.raises(ValueError):  # Doppel-Add desselben Objekts
        layer.add(obj, (2, 2))
    with pytest.raises(ValueError):  # unbekannte Ledger-Quelle
        layer.add(make_object("flint", 1.0), (0, 0), source="geschenkt")


def test_remove_ist_identitaetsbasiert():
    layer = _layer()
    a = make_object("granite", 1.0)
    b = make_object("granite", 1.0)  # wertgleich, aber anderes Objekt
    layer.add(a, (2, 2))
    layer.add(b, (2, 2))
    layer.remove(a)
    assert layer.objects_at((2, 2)) == [b]
    with pytest.raises(KeyError):
        layer.remove(a)


def test_objects_near_chebyshev():
    layer = _layer()
    nah = make_object("flint", 0.5)
    diagonal = make_object("flint", 0.5)
    fern = make_object("flint", 0.5)
    layer.add(nah, (5, 5))
    layer.add(diagonal, (6, 6))
    layer.add(fern, (8, 5))
    gefunden = layer.objects_near((5, 5), radius=1)
    assert {id(o) for o, _ in gefunden} == {id(nah), id(diagonal)}
    assert (diagonal, (6, 6)) in gefunden


def test_ledger_und_erhaltungsterme():
    layer = _layer()
    layer.add(make_object("granite", 3.0), (1, 1), source="spawned")
    layer.add(make_object("carcass", 70.0), (2, 2), source="from_carcass")
    layer.add(make_object("flint", 1.0), (3, 3))  # ledger-neutral (z. B. abgelegt)
    assert layer.ledger["spawned"] == 3.0
    assert layer.ledger["from_carcass"] == 70.0
    assert layer.ledger["eaten"] == 0.0 and layer.ledger["decayed"] == 0.0
    lhs, rhs = layer.conservation_terms(held_mass_kg=0.0)
    # Boden 74 == spawned 3 + from_carcass 70 + neutral 1 → neutral zählt links,
    # also balanciert die Invariante nur, wenn neutrale Adds Bewegungen sind
    # (Hand→Boden). Hier: bewusst unbalanciert um genau 1.0.
    assert math.isclose(lhs - rhs, 1.0, rel_tol=1e-9)


def test_discovery_ist_instanz_pro_layer():
    a, b = _layer(), _layer()
    assert a.discovery is not b.discovery
    a.discovery.register(np.zeros(13, dtype=np.float32))
    assert a.discovery.known_ids() and not b.discovery.known_ids()


def test_pickle_roundtrip_baut_rueckwaertsmap_neu():
    """Default-rng ist das random-MODUL (nicht picklebar) — der Roundtrip MUSS es
    trotzdem überleben: exakt der Checkpoint-Pfad (_save_checkpoint pickelt world
    inkl. world.objects, simulation.py:343)."""
    import random as random_mod

    layer = _layer()  # Default-rng = random-Modul
    layer.add(make_object("granite", 2.5), (4, 3), source="spawned")
    layer.add(make_object("carcass", 70.0), (4, 3), source="from_carcass")
    layer2 = pickle.loads(pickle.dumps(layer))
    objekte = layer2.objects_at((4, 3))
    assert len(objekte) == 2
    assert all(layer2.position_of(o) == (4, 3) for o in objekte)
    assert layer2.ledger == layer.ledger
    assert math.isclose(layer2.total_mass(), layer.total_mass(), rel_tol=1e-12)
    assert layer2.rng is random_mod  # Modul-Default wieder angebunden
    layer2.rng.random()  # und benutzbar


def test_pickle_erhaelt_explizite_rng_instanz():
    import random

    layer = ObjectLayer(10, 8, rng=random.Random(5))
    erwartet = random.Random(5).random()
    layer2 = pickle.loads(pickle.dumps(layer))
    assert isinstance(layer2.rng, random.Random)
    assert layer2.rng.random() == erwartet  # RNG-Zustand überlebt das Pickling
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_phys_objects.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'artificial_society.environment.phys_objects'`.

- [ ] **Step 3: Implementieren** — neue Datei `artificial_society/environment/phys_objects.py`:

```python
"""ObjectLayer — sparse Registry physischer Objekte (Plan 3a, Spec B1–B3).

Die Objekt-Schicht lebt als Schwester-Attribut ``world.objects`` neben den
Zell-Arrays ``world.F/S`` — sie schreibt NIE in Zell-Arrays und taucht NIE in
``world.obj[y][x]`` auf (GPU-Garantie, Spec Prinzip 6). Masse ist die
Erhaltungsgröße: jeder Zu-/Abfluss läuft über den Ledger
(spawned / from_carcass / eaten / decayed); Bewegungen Boden↔Hand sind
ledger-neutral. DiscoveryV2 wird PRO Welt instanziert (kein Modul-Singleton).
"""

from __future__ import annotations

import random as _random
import types
from typing import Iterator

import numpy as np

from artificial_society.environment.physics.discovery import DiscoveryV2
from artificial_society.environment.physics.objects import PhysObject
from artificial_society.environment.physics.props import N_PROPS_V2

_LEDGER_SOURCES = ("spawned", "from_carcass")


class ObjectLayer:
    def __init__(self, width: int, height: int, rng=None):
        self.width = int(width)
        self.height = int(height)
        # Duck-typed RNG (braucht .random/.uniform/.randrange). Default: das
        # global via artificial_society.rng.seed_all geseedete random-Modul.
        self.rng = rng if rng is not None else _random
        self._by_pos: dict = {}  # (x, y) -> list[PhysObject]
        self._pos_by_id: dict = {}  # id(obj) -> (x, y); prozess-lokal, s. __setstate__
        self.discovery = DiscoveryV2()
        self.ledger: dict = {"spawned": 0.0, "from_carcass": 0.0, "eaten": 0.0, "decayed": 0.0}

    # -- Kern-API ------------------------------------------------------------
    def add(self, obj: PhysObject, pos, source: str | None = None) -> None:
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"pos {pos!r} liegt außerhalb des {self.width}x{self.height}-Grids")
        props = np.asarray(obj.props)
        if props.shape != (N_PROPS_V2,) or bool(np.any(props < 0.0)) or bool(np.any(props > 1.0)):
            raise ValueError("props müssen 13-dimensional in [0,1] sein")
        if obj.mass <= 0.0:
            raise ValueError("mass muss > 0 sein")
        if id(obj) in self._pos_by_id:
            raise ValueError("Objekt liegt bereits im Layer (Doppel-Add)")
        if source is not None:
            if source not in _LEDGER_SOURCES:
                raise ValueError(f"unbekannte Ledger-Quelle: {source!r}")
            self.ledger[source] += obj.mass
        self._by_pos.setdefault((x, y), []).append(obj)
        self._pos_by_id[id(obj)] = (x, y)

    def remove(self, obj: PhysObject) -> None:
        """Entfernt per Identität (zwei wertgleiche Steine sind zwei Objekte)."""
        pos = self._pos_by_id.pop(id(obj), None)
        if pos is None:
            raise KeyError("Objekt liegt nicht im Layer")
        bucket = self._by_pos[pos]
        for i, kandidat in enumerate(bucket):
            if kandidat is obj:
                del bucket[i]
                break
        if not bucket:
            del self._by_pos[pos]

    def position_of(self, obj: PhysObject):
        return self._pos_by_id.get(id(obj))

    def objects_at(self, pos) -> list:
        return list(self._by_pos.get((int(pos[0]), int(pos[1])), ()))

    def objects_near(self, pos, radius: int) -> list:
        """Alle Boden-Objekte im Chebyshev-Radius, als (obj, pos)-Paare.

        Iteriert die sparse Positions-Map (Insertion-Order ⇒ deterministisch).
        """
        x, y = int(pos[0]), int(pos[1])
        out = []
        for (px, py), bucket in self._by_pos.items():
            if abs(px - x) <= radius and abs(py - y) <= radius:
                out.extend((obj, (px, py)) for obj in bucket)
        return out

    def all_objects(self) -> Iterator:
        for pos, bucket in self._by_pos.items():
            for obj in list(bucket):
                yield obj, pos

    def total_mass(self) -> float:
        """Gesamtmasse am Boden (Hände zählt die Erhaltungs-Invariante separat)."""
        return sum(obj.mass for bucket in self._by_pos.values() for obj in bucket)

    def conservation_terms(self, held_mass_kg: float = 0.0):
        """(lhs, rhs) der Massen-Invariante (Spec D1):
        boden + hände + eaten + decayed == spawned + from_carcass."""
        lhs = self.total_mass() + held_mass_kg + self.ledger["eaten"] + self.ledger["decayed"]
        rhs = self.ledger["spawned"] + self.ledger["from_carcass"]
        return lhs, rhs

    # -- Pickling: id()-Rückwärts-Map ist prozess-lokal; Modul-rng nicht picklebar --
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_pos_by_id")
        if isinstance(state.get("rng"), types.ModuleType):
            state["rng"] = None  # Modul-Default ist prozess-global, nicht picklebar
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.__dict__.get("rng") is None:
            self.rng = _random  # Modul-Default wieder anbinden (global via seed_all geseedet)
        self._pos_by_id = {
            id(obj): pos for pos, bucket in self._by_pos.items() for obj in bucket
        }
```

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_phys_objects.py -q`
Expected: 8 passed (inkl. beider Pickle-Tests — Modul-Default-rng UND Random-Instanz).

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/phys_objects.py tests/environment/test_phys_objects.py
git commit -m "feat(environment): ObjectLayer — sparse Objekt-Registry mit Massen-Ledger und per-Welt-DiscoveryV2"
```

---

### Task 5: World-Integration — `world.objects` + Checkpoint-Migration

Die Objekt-Schicht hängt IMMER an der Welt (auch bei Flag aus — Konstruktion zieht keinerlei RNG,
bleibt leer und ändert kein Verhalten ⇒ Golden bleibt grün). So brauchen weder Renderer noch
Systeme `hasattr`-Tänze.

**Files:**
- Modify: `artificial_society/world.py:1-13` (Import), `:28-29` (`__init__`, nach `self.cells = CellGrid(self)`), `:99-106` (`ensure_array_storage`)
- Test: `tests/environment/test_phys_objects.py` (erweitern)

**Interfaces:**
- Consumes: `ObjectLayer` (Task 4)
- Produces: `world.objects: ObjectLayer` — an jedem `World`-Objekt vorhanden, auch nach
  Checkpoint-Migration alter Welten (`ensure_array_storage`). Tasks 12–15 und das
  Registry-System (Task 13) greifen darüber zu.

- [ ] **Step 1: Failing Tests schreiben** — in `tests/environment/test_phys_objects.py` anhängen:

```python
def test_world_traegt_objectlayer_als_schwester_attribut():
    """Spec B1: world.objects lebt NEBEN world.F/S, nie in den Zell-Arrays."""
    from artificial_society.world import World

    world = World(12, 9)
    assert isinstance(world.objects, ObjectLayer)
    assert world.objects.width == 12 and world.objects.height == 9
    assert world.objects.total_mass() == 0.0  # Konstruktion spawnt nichts


def test_ensure_array_storage_migriert_alte_welten():
    from artificial_society.world import World

    world = World(6, 5)
    del world.objects  # simuliert eine Welt aus einem alten Checkpoint
    world.ensure_array_storage()
    assert isinstance(world.objects, ObjectLayer)


def test_discovery_pro_welt_verschieden():
    """Spec D2: zwei Welten teilen keinen Discovery-Zustand."""
    from artificial_society.world import World

    a, b = World(6, 5), World(6, 5)
    assert a.objects.discovery is not b.objects.discovery
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_phys_objects.py -q -k "world_traegt or migriert or pro_welt"`
Expected: FAIL mit `AttributeError: 'World' object has no attribute 'objects'`.

- [ ] **Step 3: Implementieren** — in `artificial_society/world.py`:

Import ergänzen (nach Zeile 7, bei den anderen `environment`-Imports):

```python
from artificial_society.environment.phys_objects import ObjectLayer
```

In `World.__init__`, direkt nach `self.cells = CellGrid(self)` (Zeile 28) einfügen:

```python
        # Physik-v2-Objektschicht (Spec B1): sparse Schwester-Struktur neben
        # F/S — NIE in den Zell-Arrays (GPU-Garantie). Konstruktion zieht
        # keine RNG und bleibt bei physics_v2=False dauerhaft leer.
        self.objects = ObjectLayer(width, height, rng=random)
```

In `ensure_array_storage()` (Zeile 99–106) am Ende ergänzen:

```python
        if not hasattr(self, "objects"):
            self.objects = ObjectLayer(self.width, self.height, rng=random)
```

- [ ] **Step 4: Tests + Determinismus-Kontrakt verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_phys_objects.py tests/test_headless.py tests/test_regression_golden.py tests/test_phase3_checkpoints.py -q`
Expected: alle grün — insbesondere der Golden-Test (die neue Schicht zieht keine RNG) und
der Checkpoint-Roundtrip (`test_phase3_checkpoints.py`): ab diesem Task hängt die
ObjectLayer an JEDER Welt und wird von `_save_checkpoint` mitgepickelt — das beweist den
Modul-rng-Pickle-Fix aus Task 4 früh, statt erst in Task 16 aufzufallen.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/world.py tests/environment/test_phys_objects.py
git commit -m "feat(world): world.objects (ObjectLayer) als Schwester-Attribut + Checkpoint-Migration"
```

---

### Task 6: Spawning — Biome-gebunden, kalibrierte Massen, kind `"spawn"`

**Files:**
- Modify: `artificial_society/environment/phys_objects.py` (Konstanten + `seed_initial`/`tick_spawn` anhängen)
- Modify: `artificial_society/environment/physics/calibration.py:146` (Renderer-Import um `phys_objects` erweitern)
- Modify: `docs/physics/kalibrierung.md` (regeneriert)
- Test: `tests/environment/test_phys_objects.py` (erweitern), `tests/environment/physics/test_reality_gate.py` (erweitern)

**Interfaces:**
- Consumes: `ObjectLayer` (Task 4), `make_object`, `cal`
- Produces (von Task 13 konsumiert):
  - `SPAWN_INITIAL_DENSITY = 0.03`, `SPAWN_RATE_PER_CELL_TICK = 1e-5`
  - `SPAWN_TABLE: tuple[tuple[str, str, float, float, float], ...]` — `(material, biome_tag, rate_mult, mass_min_kg, mass_max_kg)`; `biome_tag` ∈ Biome-Namen ∪ `{"shore"}` (Nicht-Wasser-Zelle mit Wasser-Nachbar, Chebyshev 1)
  - `seed_initial(layer: ObjectLayer, biomes: list[list[str]]) -> None` (Start-Seeding, Ledger `"spawned"`)
  - `tick_spawn(layer: ObjectLayer, biomes: list[list[str]]) -> None` (Poisson-Regeneration pro Tick, Ledger `"spawned"`)
  - `CALIBRATED_SPAWN_PARAMS: tuple[str, ...]`

- [ ] **Step 1: Failing Tests schreiben**

In `tests/environment/test_phys_objects.py` anhängen:

```python
def _biome_grid(biome: str, w: int = 10, h: int = 10) -> list:
    return [[biome for _ in range(w)] for _ in range(h)]


def test_seed_initial_spawnt_nur_passende_biome_und_massen():
    import random

    from artificial_society.environment.phys_objects import (
        SPAWN_TABLE,
        seed_initial,
    )

    grenzen = {m: (lo, hi) for m, _, _, lo, hi in SPAWN_TABLE}
    layer = ObjectLayer(40, 40, rng=random.Random(7))
    seed_initial(layer, _biome_grid("mountain", 40, 40))
    objekte = [obj for obj, _ in layer.all_objects()]
    assert objekte, "3 % von 1600 Bergzellen müssen deterministisch > 0 Objekte liefern"
    assert {o.kind for o in objekte} <= {"granite", "flint"}  # Berg: Geröll + Knollen
    for o in objekte:
        lo, hi = grenzen[o.kind]
        assert lo <= o.mass <= hi
    assert math.isclose(layer.ledger["spawned"], layer.total_mass(), rel_tol=1e-9)


def test_seed_initial_wueste_bleibt_leer():
    import random

    from artificial_society.environment.phys_objects import seed_initial

    layer = ObjectLayer(20, 20, rng=random.Random(7))
    seed_initial(layer, _biome_grid("desert", 20, 20))
    assert layer.total_mass() == 0.0


def test_seed_initial_ufer_lehm():
    import random

    from artificial_society.environment.phys_objects import seed_initial

    # linke Spalte Wasser, Rest Grasland → Ufer = Spalte x=1 (200 Ufer-Zellen,
    # Erwartung ≈ 6 Lehm-Objekte — groß genug, dass ein Seed praktisch nie 0 liefert)
    biomes = [["water"] + ["grassland"] * 39 for _ in range(200)]
    layer = ObjectLayer(40, 200, rng=random.Random(3))
    seed_initial(layer, biomes)
    lehm = [(o, p) for o, p in layer.all_objects() if o.kind == "clay_moist"]
    assert lehm, "Ufer-Zellen müssen Lehm tragen können (falls 0: anderen Seed wählen — deterministisch)"
    assert all(p[0] == 1 for _, p in lehm)  # nur die Ufer-Spalte


def test_tick_spawn_regeneriert_langsam_und_deterministisch():
    import random

    from artificial_society.environment.phys_objects import tick_spawn

    def _lauf(seed: int) -> list:
        layer = ObjectLayer(10, 10, rng=random.Random(seed))
        biomes = _biome_grid("mountain", 10, 10)
        for _ in range(20000):
            tick_spawn(layer, biomes)
        return sorted((o.kind, round(o.mass, 9), p) for o, p in layer.all_objects())

    a, b = _lauf(11), _lauf(11)
    assert a == b, "gleicher Seed ⇒ identische Spawns"
    # Erwartung ≈ 100 Zellen · 1e-5 · (1.0 + 0.5) · 20000 = 30 Objekte — nicht 0, nicht flutend
    assert 5 <= len(a) <= 100
```

In `tests/environment/physics/test_reality_gate.py` Import ergänzen:

```python
from artificial_society.environment.phys_objects import CALIBRATED_SPAWN_PARAMS
```

und am Datei-Ende anhängen:

```python
def test_every_spawn_param_is_calibrated():
    for name in CALIBRATED_SPAWN_PARAMS:
        _assert_calibrated("spawn", name)


def test_no_orphan_spawn_entries():
    for kind, name in CALIBRATION:
        if kind == "spawn":
            assert name in CALIBRATED_SPAWN_PARAMS, f"verwaister spawn-Eintrag: {name}"
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_phys_objects.py tests/environment/physics/test_reality_gate.py -q`
Expected: FAIL/ERROR mit `ImportError: cannot import name 'seed_initial'` bzw. `'CALIBRATED_SPAWN_PARAMS'`.

- [ ] **Step 3: Implementieren**

In `artificial_society/environment/phys_objects.py`: Import-Block oben ergänzen um

```python
import math

from artificial_society.environment.physics.calibration import cal
from artificial_society.environment.physics.objects import PhysObject, make_object
```

(`PhysObject`-Import besteht schon — nur `make_object` und `math`/`cal` kommen hinzu.)
Am Datei-Ende anhängen:

```python
# ---------------------------------------------------------------------------
# Spawning (Spec B2): Biome-gebunden, kalibrierte Massen, langsame Regeneration
# ---------------------------------------------------------------------------
SPAWN_INITIAL_DENSITY = 0.03  # 3 % der passenden Biom-Zellen tragen initial ein Objekt
SPAWN_RATE_PER_CELL_TICK = 1e-5  # ≈ 0.0024 Objekte je Biom-Zelle und Tag (240 Ticks)

# (material, biome_tag, rate_mult, mass_min_kg, mass_max_kg).
# "shore" = Nicht-Wasser-Zelle mit mind. einem Wasser-Nachbarn (Chebyshev 1).
SPAWN_TABLE = (
    ("granite", "mountain", 1.0, 0.5, 8.0),
    ("flint", "mountain", 0.5, 0.3, 4.0),  # Knollen
    ("flint", "grassland", 0.05, 0.3, 4.0),  # selten: Kiesel
    ("dry_wood", "forest", 1.0, 0.5, 6.0),
    ("plant_fiber", "grassland", 1.0, 0.05, 0.4),
    ("plant_fiber", "swamp", 1.0, 0.05, 0.4),
    ("clay_moist", "swamp", 1.0, 0.5, 5.0),
    ("clay_moist", "shore", 1.0, 0.5, 5.0),
)

# Vom Realitäts-Gate geprüfte Spawn-Parameter (kind "spawn").
CALIBRATED_SPAWN_PARAMS = (
    "initial_density",
    "regen_rate",
    "granite",
    "flint",
    "dry_wood",
    "plant_fiber",
    "clay_moist",
)


def _eligible_cells(biomes, biome_tag: str) -> list:
    """Zellen (x, y), auf denen ein biome_tag spawnen darf; Scan-Reihenfolge row-major."""
    h, w = len(biomes), len(biomes[0])
    if biome_tag != "shore":
        return [(x, y) for y in range(h) for x in range(w) if biomes[y][x] == biome_tag]
    out = []
    for y in range(h):
        for x in range(w):
            if biomes[y][x] == "water":
                continue
            nachbar_wasser = any(
                0 <= y + dy < h and 0 <= x + dx < w and biomes[y + dy][x + dx] == "water"
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if dx or dy
            )
            if nachbar_wasser:
                out.append((x, y))
    return out


def _cells_for(layer: ObjectLayer, biomes, biome_tag: str) -> list:
    """Pro Layer gecachte Eignungs-Listen (Biome sind nach Weltgenerierung statisch)."""
    cache = getattr(layer, "_spawn_cells", None)
    if cache is None:
        cache = {}
        layer._spawn_cells = cache
    if biome_tag not in cache:
        cache[biome_tag] = _eligible_cells(biomes, biome_tag)
    return cache[biome_tag]


def _poisson(lam: float, rng) -> int:
    """Knuth-Poisson — für die winzigen Pro-Tick-Raten (λ ≪ 1) fast immer 1 RNG-Draw."""
    if lam <= 0.0:
        return 0
    schwelle = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= schwelle:
            return k
        k += 1


def seed_initial(layer: ObjectLayer, biomes) -> None:
    """Start-Seeding beim Welt-Aufbau (nur physics_v2): 3 % der passenden Zellen."""
    for material, biome_tag, rate_mult, m_lo, m_hi in SPAWN_TABLE:
        for pos in _cells_for(layer, biomes, biome_tag):
            if layer.rng.random() < SPAWN_INITIAL_DENSITY * rate_mult:
                layer.add(
                    make_object(material, layer.rng.uniform(m_lo, m_hi)), pos, source="spawned"
                )


def tick_spawn(layer: ObjectLayer, biomes) -> None:
    """Langsame Regeneration: Poisson über alle geeigneten Zellen einer Quelle."""
    for material, biome_tag, rate_mult, m_lo, m_hi in SPAWN_TABLE:
        cells = _cells_for(layer, biomes, biome_tag)
        if not cells:
            continue
        lam = SPAWN_RATE_PER_CELL_TICK * rate_mult * len(cells)
        for _ in range(_poisson(lam, layer.rng)):
            pos = cells[layer.rng.randrange(len(cells))]
            layer.add(make_object(material, layer.rng.uniform(m_lo, m_hi)), pos, source="spawned")


cal(
    "spawn",
    "initial_density",
    "Start-Seeding: 3 % der geeigneten Biom-Zellen tragen initial ein Objekt "
    "(SPAWN_INITIAL_DENSITY = 0.03, je Quelle skaliert mit rate_mult)",
    "Größenordnung Oberflächen-Vorkommen von Lesesteinen/Totholz; Pilot-feinjustierbar (Spec B2)",
)
cal(
    "spawn",
    "regen_rate",
    "Regeneration 1e-5 Objekte je Zelle und Tick ≈ 0.0024/Zelle/Tag (240 Ticks/Tag); auf "
    "200×200 mit ~15 % Gebirge ≈ 14 neue Steine/Tag — versiegt nicht, flutet nicht",
    "Auslegungsrechnung Spec B2 (Pilot-feinjustierbar, nie zur Laufzeit pro Agent)",
)
cal(
    "spawn",
    "granite",
    "Granit-Gerölle 0.5–8 kg im Gebirge (Lesesteine/Hangschutt)",
    "Geologie: Hangschutt/Lesesteine im Mittelgebirge",
)
cal(
    "spawn",
    "flint",
    "Feuerstein 0.3–4 kg: Knollen im Gebirge (rate_mult 0.5), selten als Kiesel im "
    "Grasland (rate_mult 0.05)",
    "Geologie: Feuerstein-Knollen in Kreide/Schotterfluren",
)
cal(
    "spawn",
    "dry_wood",
    "Totholz-Äste 0.5–6 kg im Wald",
    "Forstökologie: Totholzaufkommen in Wäldern",
)
cal(
    "spawn",
    "plant_fiber",
    "Gras-/Bastbündel 0.05–0.4 kg in Grasland und Sumpf",
    "Ethnobotanik: Sammelmengen Faserpflanzen",
)
cal(
    "spawn",
    "clay_moist",
    "Ufer-Lehm 0.5–5 kg in Sumpf und an Ufern (Nicht-Wasser-Zelle mit Wasser-Nachbar)",
    "Sedimentologie: Ton-/Lehmablagerungen an Gewässerrändern",
)
```

In `calibration.py`, `render_markdown()` (Zeile 146): den lazy Import erweitern zu:

```python
    from artificial_society.environment import phys_objects  # noqa: F401

    from . import actions, body, materials_v2, processes  # noqa: F401
```

Doku regenerieren: `../venv/bin/python scripts/gen_kalibrierung.py`

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_phys_objects.py tests/environment/physics/test_reality_gate.py -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/phys_objects.py artificial_society/environment/physics/calibration.py docs/physics/kalibrierung.md tests/environment/test_phys_objects.py tests/environment/physics/test_reality_gate.py
git commit -m "feat(environment): Biome-gebundenes Objekt-Spawning (kind 'spawn', Gate-pflichtig, Poisson-Regeneration)"
```

---

### Task 7: Verwesung + Kadaver-Rekalibrierung

**Files:**
- Modify: `artificial_society/environment/phys_objects.py` (Import + `tick_decay` anhängen)
- Modify: `artificial_society/environment/physics/materials_v2.py:105-118` (carcass-Vektor) und `:173-179` (cal-Text)
- Modify: `docs/physics/kalibrierung.md` (regeneriert)
- Test: `tests/environment/test_phys_objects.py` (erweitern)

**Interfaces:**
- Consumes: `DECAY_RATE`, `TOX_SPOILAGE_PER_TICK`, `TOX_SPOILAGE_CAP`, `DECAY_MOISTURE_MIN` aus
  `physics.actions` (Task 3); `IDX2` aus `physics.props`
- Produces: `tick_decay(layer: ObjectLayer) -> None` (von Task 13 pro Tick gerufen);
  `MATERIALS_V2["carcass"]` mit `nutrition = 0.14`, `toxicity = 0.02`

- [ ] **Step 1: Failing Tests schreiben** — in `tests/environment/test_phys_objects.py` anhängen:

```python
def test_verwesung_masse_nutrition_toxicity_und_ledger():
    from artificial_society.environment.phys_objects import tick_decay
    from artificial_society.environment.physics.actions import (
        DECAY_RATE,
        TOX_SPOILAGE_CAP,
        TOX_SPOILAGE_PER_TICK,
    )
    from artificial_society.environment.physics.props import IDX2

    layer = _layer()
    kadaver = make_object("carcass", 70.0)
    layer.add(kadaver, (2, 2), source="from_carcass")
    n0 = float(kadaver.props[IDX2["nutrition"]])
    t0 = float(kadaver.props[IDX2["toxicity"]])

    tick_decay(layer)
    assert math.isclose(kadaver.mass, 70.0 * (1.0 - DECAY_RATE), rel_tol=1e-12)
    assert math.isclose(float(kadaver.props[IDX2["nutrition"]]), n0 * (1.0 - DECAY_RATE), rel_tol=1e-6)
    assert math.isclose(float(kadaver.props[IDX2["toxicity"]]), t0 + TOX_SPOILAGE_PER_TICK, rel_tol=1e-6)
    assert math.isclose(layer.ledger["decayed"], 70.0 * DECAY_RATE, rel_tol=1e-9)
    lhs, rhs = layer.conservation_terms()
    assert math.isclose(lhs, rhs, rel_tol=1e-9)

    for _ in range(2000):  # Toxin-Kappe (0.6) wird erreicht und nie überschritten
        tick_decay(layer)
    assert float(kadaver.props[IDX2["toxicity"]]) == pytest.approx(TOX_SPOILAGE_CAP)


def test_verwesung_verschont_trockene_stoffe():
    from artificial_society.environment.phys_objects import tick_decay

    layer = _layer()
    stein = make_object("granite", 3.0)
    holz = make_object("dry_wood", 2.0)
    layer.add(stein, (1, 1), source="spawned")
    layer.add(holz, (1, 2), source="spawned")
    for _ in range(100):
        tick_decay(layer)
    assert stein.mass == 3.0 and holz.mass == 2.0
    assert layer.ledger["decayed"] == 0.0


def test_kadaver_rekalibrierung():
    """Spec B5: dressed yield ~40 % → nutrition 0.35·0.40 = 0.14; frisch fast unbedenklich."""
    from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
    from artificial_society.environment.physics.props import IDX2

    assert float(MATERIALS_V2["carcass"][IDX2["nutrition"]]) == pytest.approx(0.14)
    assert float(MATERIALS_V2["carcass"][IDX2["toxicity"]]) == pytest.approx(0.02)
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_phys_objects.py -q -k "verwesung or rekalibrierung"`
Expected: FAIL — `ImportError: cannot import name 'tick_decay'` bzw. `0.30 != approx(0.14)`.

- [ ] **Step 3: Implementieren**

`artificial_society/environment/phys_objects.py` — Import ergänzen:

```python
from artificial_society.environment.physics.actions import (
    DECAY_MOISTURE_MIN,
    DECAY_RATE,
    TOX_SPOILAGE_CAP,
    TOX_SPOILAGE_PER_TICK,
)
from artificial_society.environment.physics.props import IDX2
```

und am Datei-Ende anhängen:

```python
# ---------------------------------------------------------------------------
# Verwesung (Spec B3.4): wirkt aus Eigenschaften (feucht + nahrhaft), nie aus Labels
# ---------------------------------------------------------------------------
def tick_decay(layer: ObjectLayer) -> None:
    """Ein Verwesungs-Tick über alle Boden-Objekte: Masse und nutrition sinken
    exponentiell, toxicity steigt bis zur Kappe. Verweste Masse fließt
    bilanziert in ledger['decayed'] (kein Culling, kein Leck)."""
    for obj, _pos in layer.all_objects():
        moisture = float(obj.props[IDX2["moisture"]])
        nutrition = float(obj.props[IDX2["nutrition"]])
        if moisture < DECAY_MOISTURE_MIN or nutrition <= 0.0:
            continue
        verlust = obj.mass * DECAY_RATE
        obj.mass -= verlust
        layer.ledger["decayed"] += verlust
        obj.props[IDX2["nutrition"]] = nutrition * (1.0 - DECAY_RATE)
        tox = float(obj.props[IDX2["toxicity"]])
        if tox < TOX_SPOILAGE_CAP:
            obj.props[IDX2["toxicity"]] = min(TOX_SPOILAGE_CAP, tox + TOX_SPOILAGE_PER_TICK)
```

`materials_v2.py`, im `"carcass"`-Vektor (Zeilen 105–118) genau zwei Werte ändern:
`nutrition=0.30` → `nutrition=0.14` und `toxicity=0.10` → `toxicity=0.02`.

`materials_v2.py`, `cal("material", "carcass", ...)`-Anker (Zeile 173–179) ersetzen durch:

```python
cal(
    "material",
    "carcass",
    "Tierkadaver: Verbund aus Haut/Sehnen/Fleisch/Knochen — sehr zäh; Nährwert praktisch nur "
    "durch Zerteilen (Schneiden) erschließbar. nutrition 0.14 = essbarer Anteil ~40 % "
    "(dressed yield) × Rohfleisch 0.35 — die intensive nutrition mittelt über "
    "Knochen/Haut/Innereien; toxicity 0.02 = frisches Fleisch nahezu unbedenklich, Gefahr "
    "entsteht erst über Verwesung",
    "Zoologie/Jagdpraxis: Zerwirken, Schlachtausbeute (dressed yield ~40 %)",
)
```

Doku regenerieren: `../venv/bin/python scripts/gen_kalibrierung.py`

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/ -q`
Expected: alle grün (bestehende Cut/Strike/Tool-Pressure-Tests nutzen nutrition/toxicity des Kadavers nicht mechanisch — sie bleiben grün).

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/phys_objects.py artificial_society/environment/physics/materials_v2.py docs/physics/kalibrierung.md tests/environment/test_phys_objects.py
git commit -m "feat(environment): Verwesung (Masse/nutrition ↓, toxicity ↑ bis 0.6) + Kadaver-Rekalibrierung 0.14/0.02"
```

---

### Task 8: `ActionResult` + `do_grasp` / `do_release` / `enforce_carry_budget`

**Files:**
- Modify: `artificial_society/environment/physics/actions.py` (nach den cal-Blöcken anhängen)
- Test: `tests/environment/physics/test_actions.py` (erweitern)

**Interfaces:**
- Consumes: `Body`, `Hands` aus `.body`; `PhysObject` aus `.objects`; `ObjectLayer` (duck-typed:
  `add/remove/position_of/objects_at/ledger/discovery`)
- Produces (von Tasks 9–11, 12, 18, 19 konsumiert — exakt so):
  - `@dataclass ActionResult: ok: bool; verb: str; reason: str = ""; energy_delta_sim: float = 0.0; health_delta: float = 0.0; fragments: list = field(default_factory=list); extracted: PhysObject | None = None; bite_kg: float = 0.0; work_j: float = 0.0`
  - `do_grasp(body: Body, hands: Hands, layer, pos: tuple[int, int], target: PhysObject) -> ActionResult`
  - `do_release(body: Body, hands: Hands, layer, pos: tuple[int, int], held: PhysObject) -> ActionResult`
  - `enforce_carry_budget(body: Body, hands: Hands, layer, pos: tuple[int, int]) -> list[PhysObject]`

- [ ] **Step 1: Failing Tests schreiben** — in `tests/environment/physics/test_actions.py` anhängen
  (Import-Block oben entsprechend erweitern):

```python
from artificial_society.environment.phys_objects import ObjectLayer
from artificial_society.environment.physics.actions import (
    do_grasp,
    do_release,
    enforce_carry_budget,
)
from artificial_society.environment.physics.body import Body, Hands
from artificial_society.environment.physics.objects import make_object


def _setup(strength: float = 0.5):
    return Body(body_mass=70.0, strength=strength), Hands(), ObjectLayer(10, 10)


def test_grasp_nimmt_boden_objekt_im_radius_1():
    body, hands, layer = _setup()
    stein = make_object("granite", 2.0)
    layer.add(stein, (5, 6), source="spawned")  # Chebyshev-1 zu (5, 5)
    res = do_grasp(body, hands, layer, (5, 5), stein)
    assert res.ok and res.verb == "grasp"
    assert hands.held == [stein]
    assert layer.position_of(stein) is None  # nicht mehr am Boden


def test_grasp_noop_ausser_reichweite_volle_haende_budget():
    body, hands, layer = _setup()
    fern = make_object("granite", 1.0)
    layer.add(fern, (9, 9), source="spawned")
    assert not do_grasp(body, hands, layer, (5, 5), fern).ok  # außer Reichweite

    schwer = make_object("granite", 25.0)  # > carry_capacity 16.8 kg (70 kg, strength 0.5)
    layer.add(schwer, (5, 5), source="spawned")
    res = do_grasp(body, hands, layer, (5, 5), schwer)
    assert not res.ok and res.reason == "hands_full_or_too_heavy"
    assert layer.position_of(schwer) == (5, 5)  # No-op: bleibt liegen

    for i in range(2):  # Hände füllen (MAX_HELD = 2)
        o = make_object("flint", 0.5)
        layer.add(o, (5, 5), source="spawned")
        assert do_grasp(body, hands, layer, (5, 5), o).ok
    dritter = make_object("flint", 0.5)
    layer.add(dritter, (5, 5), source="spawned")
    assert not do_grasp(body, hands, layer, (5, 5), dritter).ok


def test_release_legt_an_eigener_position_ab():
    body, hands, layer = _setup()
    stein = make_object("granite", 2.0)
    layer.add(stein, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), stein)
    res = do_release(body, hands, layer, (3, 2), stein)
    assert res.ok
    assert hands.held == []
    assert layer.position_of(stein) == (3, 2)
    nicht_gehalten = make_object("flint", 0.5)
    assert not do_release(body, hands, layer, (3, 2), nicht_gehalten).ok


def test_ueberlast_drop_wirft_schwerstes_bis_budget_passt():
    """B4: sinkt die Kapazität (Ermüdung), fällt zu Tick-Beginn das schwerste Objekt."""
    body, hands, layer = _setup()
    schwer = make_object("granite", 6.0)
    leicht = make_object("granite", 5.0)
    layer.add(schwer, (5, 5), source="spawned")
    layer.add(leicht, (5, 5), source="spawned")
    assert do_grasp(body, hands, layer, (5, 5), schwer).ok  # Kapazität ausgeruht: 16.8 kg
    assert do_grasp(body, hands, layer, (5, 5), leicht).ok
    body.fatigue = 1.0  # Kapazität fällt auf 8.4 kg — 11 kg sind Überlast
    dropped = enforce_carry_budget(body, hands, layer, (5, 5))
    assert dropped == [schwer]
    assert hands.held == [leicht]
    assert layer.position_of(schwer) == (5, 5)
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9)
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q`
Expected: FAIL — `ImportError: cannot import name 'do_grasp'`.

- [ ] **Step 3: Implementieren** — in `actions.py` Imports ergänzen:

```python
from dataclasses import dataclass, field

from .body import Body, Hands
from .objects import PhysObject
```

und nach den cal-Blöcken anhängen:

```python
# ---------------------------------------------------------------------------
# Aktions-Mechanik (B4): eine Manipulation pro Agent und Tick; Ergebnis-Objekt
# für Logging/Neugier/Tests — kein Zugriff auf Gehirn oder Belohnung.
# ---------------------------------------------------------------------------
@dataclass
class ActionResult:
    ok: bool
    verb: str
    reason: str = ""
    energy_delta_sim: float = 0.0  # Sim-Energie-Delta des Agenten (Essen +, Arbeit −)
    health_delta: float = 0.0  # Toxin-Schaden (≤ 0)
    fragments: list = field(default_factory=list)
    extracted: PhysObject | None = None
    bite_kg: float = 0.0
    work_j: float = 0.0


def do_grasp(body: Body, hands: Hands, layer, pos, target: PhysObject) -> ActionResult:
    """Boden-Objekt im Chebyshev-Radius 1 greifen. Scheitert als No-op bei
    vollen Händen (MAX_HELD) oder Massen-Budget-Überschreitung."""
    pos = (int(pos[0]), int(pos[1]))
    tpos = layer.position_of(target)
    if tpos is None:
        return ActionResult(ok=False, verb="grasp", reason="not_on_ground")
    if max(abs(tpos[0] - pos[0]), abs(tpos[1] - pos[1])) > 1:
        return ActionResult(ok=False, verb="grasp", reason="out_of_reach")
    if not hands.can_grasp(target, body):
        return ActionResult(ok=False, verb="grasp", reason="hands_full_or_too_heavy")
    layer.remove(target)
    hands.grasp(target, body)
    return ActionResult(ok=True, verb="grasp")


def do_release(body: Body, hands: Hands, layer, pos, held: PhysObject) -> ActionResult:
    """Gehaltenes Objekt an der eigenen Position ablegen (ledger-neutral)."""
    pos = (int(pos[0]), int(pos[1]))
    if held not in hands.held:
        return ActionResult(ok=False, verb="release", reason="not_held")
    hands.release(held)
    layer.add(held, pos)
    return ActionResult(ok=True, verb="release")


def enforce_carry_budget(body: Body, hands: Hands, layer, pos) -> list:
    """Überlast-Drop (B4): übersteigt die gehaltene Masse die AKTUELLE
    Tragkapazität (z. B. weil Ermüdung sie senkt), fällt das jeweils schwerste
    Objekt, bis das Budget wieder eingehalten ist. Zu Tick-Beginn aufrufen."""
    pos = (int(pos[0]), int(pos[1]))
    dropped: list = []
    while hands.held and hands.carried_mass_kg() > body.carry_capacity_kg():
        schwerstes = max(hands.held, key=lambda o: o.mass)
        hands.release(schwerstes)
        layer.add(schwerstes, pos)
        dropped.append(schwerstes)
    return dropped
```

Hinweis: `held not in hands.held` bzw. `max(...)` arbeiten identitätsbasiert korrekt, weil
`PhysObject` mit `eq=False` deklariert ist (Objekt-Identität statt Wert-Gleichheit).

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
git commit -m "feat(physics): do_grasp/do_release + Ueberlast-Drop (enforce_carry_budget)"
```

---

### Task 9: `do_strike` — ½mv²-Kappung, exert-Bindung, Fragment-Ersatz

**Files:**
- Modify: `artificial_society/environment/physics/actions.py` (anhängen; Import `strike` ergänzen)
- Test: `tests/environment/physics/test_actions.py` (erweitern)

**Interfaces:**
- Consumes: `strike(target, striker, impact_energy_j, rng) -> StrikeResult` aus `.processes`;
  `V_MAX_STRIKE`, `WORK_SIM_ENERGY_PER_JOULE` (Task 3); `ActionResult` (Task 8);
  `body.strike_energy_j(effort)`, `body.exert_strike(energy_j)` aus `.body`
- Produces: `do_strike(body: Body, hands: Hands, layer, pos, striker_held: PhysObject, target: PhysObject, effort: float, rng: random.Random) -> ActionResult`
  — Ziel: Boden-Objekt an eigener Position ODER das andere gehaltene Objekt; bei Bruch werden
  alle Fragmente am Boden abgelegt (ledger-neutraler Ersatz) und in `layer.discovery` registriert.

- [ ] **Step 1: Failing Tests schreiben** — in `tests/environment/physics/test_actions.py`
  Import ergänzen (`do_strike` zur `actions`-Importliste, außerdem):

```python
import random

import pytest

from artificial_society.environment.physics.actions import V_MAX_STRIKE, do_strike
from artificial_society.environment.physics.body import FATIGUE_PER_JOULE
```

und anhängen:

```python
def test_strike_kiesel_kappung_und_exert_bindung():
    """D2: 0.05-kg-Kiesel liefert ≤ ½mv² = 4.9 J, egal welcher Effort; Ermüdung
    steigt IMMER, auch wenn nichts bricht."""
    body, hands, layer = _setup(strength=1.0)
    kiesel = make_object("granite", 0.05)
    ziel = make_object("flint", 0.8)  # Schwelle 60·(1.05−0.9)·0.8 = 7.2 J
    layer.add(kiesel, (5, 5), source="spawned")
    layer.add(ziel, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), kiesel)

    res = do_strike(body, hands, layer, (5, 5), kiesel, ziel, effort=1.0, rng=random.Random(1))
    cap = 0.5 * 0.05 * V_MAX_STRIKE**2  # 4.9 J
    assert res.ok and res.reason == "no_fracture"
    assert res.work_j <= cap + 1e-9
    assert not res.fragments
    assert body.fatigue == pytest.approx(res.work_j * FATIGUE_PER_JOULE)
    assert res.energy_delta_sim < 0.0  # Arbeit kostet metabolisch


def test_strike_knapping_positiv_regression():
    """D2: 0.5-kg-Granit-Schlagstein bricht 0.8-kg-Flint bei Effort 0.8 — eine spätere
    Verschärfung von V_MAX_STRIKE darf Knapping nicht still killen."""
    body, hands, layer = _setup(strength=0.5)
    hammer = make_object("granite", 0.5)
    flint = make_object("flint", 0.8)
    layer.add(hammer, (5, 5), source="spawned")
    layer.add(flint, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), hammer)

    res = do_strike(body, hands, layer, (5, 5), hammer, flint, effort=0.8, rng=random.Random(42))
    assert res.ok and res.fragments, "Knapping muss möglich bleiben"
    # Mikro-Invariante (D1): Fragmentmassen summieren exakt zur Zielmasse
    assert math.isclose(sum(f.mass for f in res.fragments), 0.8, rel_tol=1e-9)
    # Ziel ist ersetzt: Fragmente liegen am Boden, das Ziel nicht mehr
    assert layer.position_of(flint) is None
    for frag in res.fragments:
        assert layer.position_of(frag) == (5, 5)
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9)


def test_strike_auf_das_andere_gehaltene_objekt():
    body, hands, layer = _setup(strength=0.5)
    hammer = make_object("granite", 1.0)
    flint = make_object("flint", 0.5)
    layer.add(hammer, (5, 5), source="spawned")
    layer.add(flint, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), hammer)
    do_grasp(body, hands, layer, (5, 5), flint)

    res = do_strike(body, hands, layer, (5, 5), hammer, flint, effort=0.9, rng=random.Random(7))
    assert res.ok and res.fragments
    assert hands.held == [hammer]  # zerschlagenes Ziel verlässt die Hand
    assert all(layer.position_of(f) == (5, 5) for f in res.fragments)


def test_strike_ungueltige_ziele_sind_noop_ohne_exert():
    body, hands, layer = _setup()
    hammer = make_object("granite", 1.0)
    layer.add(hammer, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), hammer)
    fern = make_object("flint", 0.5)
    layer.add(fern, (9, 9), source="spawned")

    assert not do_strike(body, hands, layer, (5, 5), hammer, fern, 1.0, random.Random(1)).ok
    assert not do_strike(body, hands, layer, (5, 5), hammer, hammer, 1.0, random.Random(1)).ok
    nicht_gehalten = make_object("granite", 1.0)
    layer.add(nicht_gehalten, (5, 5), source="spawned")
    ziel = make_object("flint", 0.5)
    layer.add(ziel, (5, 5), source="spawned")
    assert not do_strike(body, hands, layer, (5, 5), nicht_gehalten, ziel, 1.0, random.Random(1)).ok
    assert body.fatigue == 0.0  # kein ausgeführter Schlag → keine Ermüdung
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q -k strike`
Expected: FAIL — `ImportError: cannot import name 'do_strike'`.

- [ ] **Step 3: Implementieren** — in `actions.py` Import ergänzen:

```python
import random

from .processes import strike
```

und anhängen:

```python
def do_strike(
    body: Body,
    hands: Hands,
    layer,
    pos,
    striker_held: PhysObject,
    target: PhysObject,
    effort: float,
    rng: random.Random,
) -> ActionResult:
    """Schlag mit gehaltenem Schläger auf ein Boden-Objekt an eigener Position
    oder auf das andere gehaltene Objekt.

    Gelieferte Energie E = min(body.strike_energy_j(effort), ½·m_striker·V_MAX²)
    — die Kappung schließt den Kiesel-Exploit (leichter Schläger kann nicht
    mehr kinetische Energie tragen als ½mv²). body.exert_strike bindet
    verpflichtend: JEDER ausgeführte Schlag ermüdet, auch ohne Bruch.
    Ungültiges Ziel (nicht erreichbar / Schläger nicht in der Hand) ist ein
    No-op OHNE Exert — es wurde nicht geschlagen.
    """
    pos = (int(pos[0]), int(pos[1]))
    if striker_held not in hands.held:
        return ActionResult(ok=False, verb="strike", reason="striker_not_held")
    if target is striker_held:
        return ActionResult(ok=False, verb="strike", reason="target_is_striker")
    target_held = target in hands.held
    if not target_held and layer.position_of(target) != pos:
        return ActionResult(ok=False, verb="strike", reason="target_out_of_reach")

    effort = min(max(float(effort), 0.0), 1.0)
    impact_j = min(body.strike_energy_j(effort), 0.5 * striker_held.mass * V_MAX_STRIKE**2)
    body.exert_strike(impact_j)  # verpflichtende exert↔strike-Bindung (B4)
    energy_delta = -impact_j * WORK_SIM_ENERGY_PER_JOULE  # Arbeits-Metabolik (B5)

    result = strike(target, striker_held, impact_j, rng)
    if not result.fractured:
        return ActionResult(
            ok=True, verb="strike", reason="no_fracture",
            energy_delta_sim=energy_delta, work_j=impact_j,
        )
    # Ziel durch Fragmente ersetzen (Masse exakt erhalten, ledger-neutral);
    # Fragmente fallen zu Boden — 2–3 Bruchstücke passen in keine Hand.
    if target_held:
        hands.release(target)
    else:
        layer.remove(target)
    for frag in result.fragments:
        layer.add(frag, pos)
        layer.discovery.register(frag.props)
    return ActionResult(
        ok=True, verb="strike",
        energy_delta_sim=energy_delta, fragments=result.fragments, work_j=impact_j,
    )
```

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
git commit -m "feat(physics): do_strike — ½mv²-Kappung, verpflichtende exert-Bindung, Fragment-Ersatz im Layer"
```

---

### Task 10: `do_cut` — beidseitiger Klingen-Massen-Faktor, Effort-Ertrag, Schneidearbeit

**Files:**
- Modify: `artificial_society/environment/physics/actions.py` (anhängen; Import `cut` ergänzen)
- Test: `tests/environment/physics/test_actions.py` (erweitern)

**Interfaces:**
- Consumes: `cut(target, tool) -> CutResult` aus `.processes`; `BLADE_MASS_REF`,
  `BLADE_HANDLE_MAX`, `CUT_WORK_J_BASE`, `CUT_WORK_J_PER_EFFORT`, `WORK_SIM_ENERGY_PER_JOULE`
  (Task 3); `ActionResult` (Task 8)
- Produces:
  - `blade_factor(blade: PhysObject | None) -> float` — `1.0` für bloße Hand;
    sonst `clamp(sqrt(m/0.15), 0.2, 1.0) · clamp((2.0 − m)/1.0, 0.2, 1.0)`
  - `do_cut(body: Body, hands: Hands, layer, pos, blade_held: PhysObject | None, target: PhysObject, effort: float) -> ActionResult`
    — Ziel: Boden an eigener Position ODER gehalten; Ertrag skaliert mit
    `blade_factor · (0.5 + 0.5·effort)` (Spec B4 Rev. 4); `extracted` fällt zu Boden,
    `remainder` ersetzt das Ziel an dessen Ort (Hand bleibt Hand, Boden bleibt Boden);
    Masse exakt erhalten.

- [ ] **Step 1: Failing Tests schreiben** — Import ergänzen (`do_cut`, `blade_factor`,
  `CUT_WORK_J_BASE`, `CUT_WORK_J_PER_EFFORT` zur `actions`-Importliste; außerdem
  `from artificial_society.environment.physics.props import pv` und
  `from artificial_society.environment.physics.objects import PhysObject`), dann anhängen:

```python
def _klinge(mass: float) -> PhysObject:
    """Klinge mit FIXEN Props (Schärfe 0.8, Härte 0.7) — nur die Masse variiert,
    damit der Massen-Faktor isoliert messbar ist."""
    return PhysObject(props=pv(sharpness=0.8, hardness=0.7), mass=mass, kind="test_blade")


def test_blade_factor_beidseitig_begrenzt():
    """D2: 20-g-Splitter < 40 % eines 300-g-Abschlags UND 4-kg-Brocken < 300-g-Abschlag."""
    f20g = blade_factor(_klinge(0.02))
    f300g = blade_factor(_klinge(0.3))
    f4kg = blade_factor(_klinge(4.0))
    assert f20g < 0.4 * f300g
    assert f4kg < f300g
    assert f4kg == pytest.approx(0.2)  # das „8-kg-Skalpell“ ist zu
    assert blade_factor(None) == 1.0  # bloße Hand hat keinen Massen-Faktor


def test_cut_ertrag_skaliert_mit_klingen_faktor():
    # Effort konstant (0.5) — der Effort-Faktor kürzt sich aus den Verhältnissen heraus.
    for masse in (0.02, 0.3, 4.0):
        body, hands, layer = _setup()
        klinge = _klinge(masse)
        layer.add(klinge, (5, 5), source="spawned")
        do_grasp(body, hands, layer, (5, 5), klinge)
        kadaver = make_object("carcass", 20.0)
        layer.add(kadaver, (5, 5), source="spawned")
        res = do_cut(body, hands, layer, (5, 5), klinge, kadaver, effort=0.5)
        assert res.ok and res.extracted is not None
        if masse == 0.3:
            ertrag_300g = res.extracted.mass
        elif masse == 0.02:
            ertrag_20g = res.extracted.mass
        else:
            ertrag_4kg = res.extracted.mass
    assert ertrag_20g < 0.4 * ertrag_300g
    assert ertrag_4kg < ertrag_300g


def test_cut_ertrag_skaliert_mit_effort():
    """Spec B4 (Rev. 4): yield × (0.5 + 0.5·effort) — Effort 1.0 extrahiert exakt 2×
    soviel je Schnitt wie Effort 0.0, bei gleichem Werkzeug und gleichem Ziel."""
    ertraege = {}
    for effort in (0.0, 1.0):
        body, hands, layer = _setup()
        klinge = _klinge(0.3)  # blade_factor 1.0 → nur der Effort-Faktor wirkt
        layer.add(klinge, (5, 5), source="spawned")
        do_grasp(body, hands, layer, (5, 5), klinge)
        kadaver = make_object("carcass", 20.0)
        layer.add(kadaver, (5, 5), source="spawned")
        res = do_cut(body, hands, layer, (5, 5), klinge, kadaver, effort=effort)
        assert res.ok and res.extracted is not None
        ertraege[effort] = res.extracted.mass
    assert ertraege[1.0] == pytest.approx(2.0 * ertraege[0.0], rel=1e-9)


def test_cut_masse_exakt_erhalten_und_platzierung():
    body, hands, layer = _setup()
    klinge = _klinge(0.3)
    layer.add(klinge, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), klinge)
    kadaver = make_object("carcass", 20.0)
    layer.add(kadaver, (5, 5), source="spawned")

    res = do_cut(body, hands, layer, (5, 5), klinge, kadaver, effort=0.5)
    # Mikro-Invariante (D1): extracted + remainder == target, exakt
    boden = layer.objects_at((5, 5))
    remainder = [o for o in boden if o is not res.extracted]
    assert len(boden) == 2 and len(remainder) == 1
    assert math.isclose(res.extracted.mass + remainder[0].mass, 20.0, rel_tol=1e-9)
    assert layer.position_of(kadaver) is None  # Original ist ersetzt
    # Schneidearbeit über den Ermüdungspfad: CUT_WORK_J = 15 + 35·0.5 = 32.5 J
    erwartete_arbeit = CUT_WORK_J_BASE + CUT_WORK_J_PER_EFFORT * 0.5
    assert res.work_j == pytest.approx(erwartete_arbeit)
    assert body.fatigue == pytest.approx(erwartete_arbeit * FATIGUE_PER_JOULE)
    assert res.energy_delta_sim < 0.0
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9)


def test_cut_gehaltenes_ziel_remainder_bleibt_in_der_hand():
    body, hands, layer = _setup()
    fleisch = make_object("raw_meat", 2.0)
    layer.add(fleisch, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), fleisch)

    res = do_cut(body, hands, layer, (5, 5), None, fleisch, effort=0.2)  # bloße Hand
    assert res.ok and res.extracted is not None
    assert len(hands.held) == 1 and hands.held[0] is not fleisch  # remainder ersetzt Ziel
    assert layer.position_of(res.extracted) == (5, 5)  # Abschnitt fällt zu Boden
    assert math.isclose(hands.held[0].mass + res.extracted.mass, 2.0, rel_tol=1e-9)


def test_cut_granit_kein_ertrag_aber_arbeit():
    """Gestein ist nicht schneidbar (processes.cut) — Arbeit fällt trotzdem an."""
    body, hands, layer = _setup()
    klinge = _klinge(0.3)
    layer.add(klinge, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), klinge)
    granit = make_object("granite", 3.0)
    layer.add(granit, (5, 5), source="spawned")
    res = do_cut(body, hands, layer, (5, 5), klinge, granit, effort=1.0)
    assert res.ok and res.reason == "no_yield" and res.extracted is None
    assert granit.mass == 3.0 and layer.position_of(granit) == (5, 5)
    assert body.fatigue > 0.0
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q -k cut`
Expected: FAIL — `ImportError: cannot import name 'do_cut'`.

- [ ] **Step 3: Implementieren** — in `actions.py` Import `cut` ergänzen
  (`from .processes import cut, strike`) und anhängen:

```python
def blade_factor(blade: PhysObject | None) -> float:
    """Klingen-Massen-Faktor (B4), beidseitig begrenzt: zu leichte Splitter
    (< ~150 g) und unhandliche Brocken (> ~1 kg einhändig) schneiden schlecht.
    blade_factor = clamp(sqrt(m/BLADE_MASS_REF), 0.2, 1) · clamp((2·HANDLE_MAX − m)/HANDLE_MAX, 0.2, 1)
    Bloße Hand: 1.0 (kein Werkzeug, kein Massen-Faktor — der Hand-Malus steckt
    in BARE_HAND_SHARPNESS der Prozess-Physik)."""
    if blade is None:
        return 1.0
    leicht = min(max(math.sqrt(blade.mass / BLADE_MASS_REF), 0.2), 1.0)
    handlich = min(max((BLADE_HANDLE_MAX * 2.0 - blade.mass) / BLADE_HANDLE_MAX, 0.2), 1.0)
    return leicht * handlich


def do_cut(
    body: Body,
    hands: Hands,
    layer,
    pos,
    blade_held: PhysObject | None,
    target: PhysObject,
    effort: float,
) -> ActionResult:
    """Schneiden mit Klinge aus der Hand oder bloßer Hand (blade_held=None).

    Der Ertrag der Prozess-Physik (processes.cut) wird mit dem beidseitigen
    Klingen-Massen-Faktor UND dem Effort-Faktor (0.5 + 0.5·effort) skaliert
    (Spec B4 Rev. 4: Zerlegegeschwindigkeit skaliert mit aufgebrachter Kraft);
    Masse bleibt exakt erhalten (der nicht abgetrennte Anteil bleibt im
    remainder). Schneiden kostet Arbeit CUT_WORK_J = 15 + 35·effort über
    denselben Ermüdungspfad wie der Schlag. Platzierung: extracted fällt zu
    Boden; remainder ersetzt das Ziel an dessen Ort (Hand bleibt Hand, Boden
    bleibt Boden)."""
    pos = (int(pos[0]), int(pos[1]))
    if blade_held is not None and blade_held not in hands.held:
        return ActionResult(ok=False, verb="cut", reason="blade_not_held")
    if target is blade_held:
        return ActionResult(ok=False, verb="cut", reason="target_is_blade")
    target_held = target in hands.held
    if not target_held and layer.position_of(target) != pos:
        return ActionResult(ok=False, verb="cut", reason="target_out_of_reach")

    effort = min(max(float(effort), 0.0), 1.0)
    work_j = CUT_WORK_J_BASE + CUT_WORK_J_PER_EFFORT * effort
    body.exert_strike(work_j)  # derselbe Ermüdungspfad (FATIGUE_PER_JOULE)
    energy_delta = -work_j * WORK_SIM_ENERGY_PER_JOULE

    result = cut(target, blade_held)
    if result.extracted is None:
        return ActionResult(
            ok=True, verb="cut", reason="no_yield", energy_delta_sim=energy_delta, work_j=work_j
        )
    extracted, remainder = result.extracted, result.remainder
    # Ertrag = Prozess-Yield × Klingen-Massen-Faktor × Effort-Faktor (B4 Rev. 4);
    # die Kosten (CUT_WORK_J, Ermüdung) skalieren bereits oben mit Effort.
    faktor = blade_factor(blade_held) * (0.5 + 0.5 * effort)
    if faktor < 1.0:
        skaliert = extracted.mass * faktor
        if skaliert <= 1e-9:
            return ActionResult(
                ok=True, verb="cut", reason="no_yield",
                energy_delta_sim=energy_delta, work_j=work_j,
            )
        extracted.mass = skaliert
    remainder.mass = target.mass - extracted.mass  # Masse exakt erhalten

    if target_held:
        hands.held[hands.held.index(target)] = remainder  # Masse sinkt strikt → kein Budget-Check
    else:
        layer.remove(target)
        layer.add(remainder, pos)
    layer.add(extracted, pos)
    layer.discovery.register(extracted.props)
    return ActionResult(
        ok=True, verb="cut", energy_delta_sim=energy_delta, extracted=extracted, work_j=work_j
    )
```

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
git commit -m "feat(physics): do_cut — beidseitiger Klingen-Massen-Faktor, Schneidearbeit, exakte Massen-Erhaltung"
```

---

### Task 11: `do_eat` — Energie-Kopplung, Toxin-Schaden, Steinbeißen-No-op

**Files:**
- Modify: `artificial_society/environment/physics/actions.py` (anhängen)
- Test: `tests/environment/physics/test_actions.py` (erweitern)

**Interfaces:**
- Consumes: `BITE_MASS_KG`, `MIN_NUTRITION_EDIBLE`, `KCAL_PER_KG_PER_NUTRITION`,
  `SIM_ENERGY_PER_KCAL`, `TOX_DAMAGE_PER_KG` (Task 3); `IDX2` aus `.props`; `ActionResult` (Task 8)
- Produces: `do_eat(body: Body, hands: Hands, layer, pos, target: PhysObject) -> ActionResult`
  — Ziel gehalten oder am Boden an eigener Position; ein Biss pro Tick;
  `energy_delta_sim = nutrition · 4000 · bite_kg · SIM_ENERGY_PER_KCAL`,
  `health_delta = −(toxicity · bite_kg · TOX_DAMAGE_PER_KG)`; Objektmasse sinkt um `bite_kg`
  (`ledger["eaten"]`), vollständig verzehrte Objekte verschwinden aus Layer/Hand.

- [ ] **Step 1: Failing Tests schreiben** — Import `do_eat` und `TOX_DAMAGE_PER_KG`
  zur `actions`-Importliste ergänzen, dann anhängen:

```python
def test_eat_1kg_rohfleisch_ergibt_45_sim_energie():
    """D2 (Produkt-Test): 1 kg raw_meat ⇒ 45 ± 1 Sim-Energie über 4 Bisse (0.3+0.3+0.3+0.1)."""
    body, hands, layer = _setup()
    fleisch = make_object("raw_meat", 1.0)
    layer.add(fleisch, (5, 5), source="spawned")

    gesamt_energie = 0.0
    gesamt_schaden = 0.0
    bisse = 0
    while layer.position_of(fleisch) is not None:
        res = do_eat(body, hands, layer, (5, 5), fleisch)
        assert res.ok
        gesamt_energie += res.energy_delta_sim
        gesamt_schaden += res.health_delta
        bisse += 1
        assert bisse <= 10, "1 kg muss in ≤ 4 Bissen à 0.3 kg weg sein"
    assert bisse == 4
    assert abs(gesamt_energie - MEAT_ENERGY) <= 1.0  # 44.8 ≈ 45
    # Toxizität roh (0.15): −0.15·1.0·20 = −3.0 Health über die ganze Mahlzeit
    assert gesamt_schaden == pytest.approx(-0.15 * 1.0 * TOX_DAMAGE_PER_KG)
    assert math.isclose(layer.ledger["eaten"], 1.0, rel_tol=1e-9)
    lhs, rhs = layer.conservation_terms()
    assert math.isclose(lhs, rhs, rel_tol=1e-9)


def test_eat_steinbeissen_ist_noop():
    body, hands, layer = _setup()
    granit = make_object("granite", 2.0)
    layer.add(granit, (5, 5), source="spawned")
    res = do_eat(body, hands, layer, (5, 5), granit)
    assert not res.ok and res.reason == "not_edible"
    assert res.energy_delta_sim == 0.0 and res.health_delta == 0.0
    assert granit.mass == 2.0 and layer.ledger["eaten"] == 0.0


def test_eat_aus_der_hand_und_vollverzehr():
    body, hands, layer = _setup()
    stueck = make_object("raw_meat", 0.25)  # < BITE_MASS_KG → ein Biss, weg
    layer.add(stueck, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), stueck)

    res = do_eat(body, hands, layer, (5, 5), stueck)
    assert res.ok and res.bite_kg == pytest.approx(0.25)
    assert hands.held == []
    assert math.isclose(layer.ledger["eaten"], 0.25, rel_tol=1e-9)


def test_eat_ausser_reichweite_noop():
    body, hands, layer = _setup()
    fleisch = make_object("raw_meat", 1.0)
    layer.add(fleisch, (9, 9), source="spawned")
    assert not do_eat(body, hands, layer, (5, 5), fleisch).ok
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q -k eat`
Expected: FAIL — `ImportError: cannot import name 'do_eat'`.

- [ ] **Step 3: Implementieren** — in `actions.py` Import `IDX2` ergänzen
  (`from .props import IDX2`) und anhängen:

```python
def do_eat(body: Body, hands: Hands, layer, pos, target: PhysObject) -> ActionResult:
    """Ein Biss pro Tick: bite = min(BITE_MASS_KG, Restmasse). Energie-Kopplung
    (B5): energy += nutrition·4000·bite·SIM_ENERGY_PER_KCAL; Toxin-Schaden:
    health −= toxicity·bite·TOX_DAMAGE_PER_KG. Objekte mit nutrition ≤ 0.02
    sind wirkungslos (Steinbeißen = No-op). Gegessene Masse fließt bilanziert
    in ledger['eaten']; vollständig verzehrte Objekte verschwinden."""
    pos = (int(pos[0]), int(pos[1]))
    target_held = target in hands.held
    if not target_held and layer.position_of(target) != pos:
        return ActionResult(ok=False, verb="eat", reason="target_out_of_reach")
    nutrition = float(target.props[IDX2["nutrition"]])
    if nutrition <= MIN_NUTRITION_EDIBLE:
        return ActionResult(ok=False, verb="eat", reason="not_edible")

    bite = min(BITE_MASS_KG, target.mass)
    if target.mass - bite <= 1e-9:
        bite = target.mass  # Krümel-Reste mitessen statt Masse zu verlieren
    energy_gain = nutrition * KCAL_PER_KG_PER_NUTRITION * bite * SIM_ENERGY_PER_KCAL
    toxicity = float(target.props[IDX2["toxicity"]])
    health_delta = -(toxicity * bite * TOX_DAMAGE_PER_KG)

    if bite >= target.mass:  # vollständig verzehrt
        if target_held:
            hands.release(target)
        else:
            layer.remove(target)
    else:
        target.mass -= bite
    layer.ledger["eaten"] += bite
    return ActionResult(
        ok=True, verb="eat", energy_delta_sim=energy_gain, health_delta=health_delta, bite_kg=bite
    )
```

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
git commit -m "feat(physics): do_eat — kcal↔Sim-Energie-Kopplung, Toxin-Schaden, Ledger 'eaten'"
```

---

### Task 12: `physics_v2`-Flag + Body/Hands-Embodiment (Default-strength 0.5)

Das Flag wird `Simulation`-Kwarg und Agent-Attribut (gesetzt bei `spawn_*`). Im v2-Modus bekommt
jeder Agent `Body(body_mass=70.0, strength=0.5)` + `Hands()`; pro Tick laufen `carry_tick`/
`rest_tick` und der Überlast-Drop. Bei Flag aus ändert sich NICHTS (keine RNG-Draws, Golden grün).

**Files:**
- Modify: `artificial_society/agents/agent.py:1-50` (Import), `:103-176` (`ensure_fields`), `:249-303` (Dataclass-Felder), nach `:176` (`attach_body`-Helper), `:1060-1062` (`update`, nach `self._sleep_tick(mods)`)
- Modify: `simulation.py:101-155` (`__init__`-Kwarg + Flag), `:157-160` (`spawn_initial_population`), `:162-203` (`spawn_child_from_parent`), `:244-249` (`emergency_respawn`)
- Test: `tests/test_physics_v2_embodiment.py` (neu)

**Interfaces:**
- Consumes: `Body`, `Hands`, `BODY_MASS_DEFAULT_KG` aus `physics.body` (Task 2);
  `enforce_carry_budget` aus `physics.actions` (Task 8); `world.objects` (Task 5)
- Produces (von Tasks 13–16 konsumiert):
  - `Simulation(..., physics_v2: bool = False)`; `sim.physics_v2: bool`
  - Agent-Felder: `physics_v2: bool = False`, `body: object = None`, `hands: object = None`
  - `attach_body(agent) -> None` in `agent.py` (setzt Flag + Body/Hands)
  - `ensure_fields` reinstauriert `physics_v2/body/hands` für Checkpoint-Agenten

- [ ] **Step 1: Failing Tests schreiben** — neue Datei `tests/test_physics_v2_embodiment.py`:

```python
"""Plan 3a: physics_v2-Flag, Body/Hands-Embodiment, Überlast-Drop im Tick (Spec A, B4)."""

from __future__ import annotations

from artificial_society.agents.agent import Agent, attach_body, ensure_fields
from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG, Body, Hands
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def test_v2_sim_embodied_alle_agenten():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    assert sim.physics_v2 is True
    for a in sim.agents:
        assert a.physics_v2 is True
        assert isinstance(a.body, Body)
        assert a.body.body_mass == BODY_MASS_DEFAULT_KG and a.body.strength == 0.5
        assert isinstance(a.hands, Hands) and a.hands.held == []


def test_v1_sim_bleibt_unveraendert():
    sim = Simulation(seed=42, **_PARAMS)
    assert sim.physics_v2 is False
    for a in sim.agents:
        assert a.physics_v2 is False and a.body is None and a.hands is None


def test_ensure_fields_reinstauriert_embodiment():
    """Checkpoint-Agent (Flag gesetzt, Body fehlt) wird vollständig reinstauriert."""
    agent = Agent.spawn_random(1, 1)
    agent.physics_v2 = True
    agent.body = None
    agent.hands = None
    ensure_fields(agent)
    assert isinstance(agent.body, Body) and isinstance(agent.hands, Hands)


def test_ueberlast_drop_laeuft_im_sim_tick():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    schwer = make_object("granite", 6.0)
    leicht = make_object("granite", 5.0)
    agent.hands.held.extend([schwer, leicht])
    agent.body.fatigue = 1.0  # Kapazität 8.4 kg < 11 kg gehalten
    sim.step()
    assert schwer not in agent.hands.held, "schwerstes Objekt muss zwangsabgelegt werden"
    assert sim.world.objects.position_of(schwer) is not None


def test_respawns_werden_embodied():
    """Deckt den emergency_respawn-Pfad; spawn_child_from_parent nutzt dasselbe
    attach_body-Muster (Code-identisch, im Sozial-RNG schwer deterministisch erzwingbar)."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    sim.agents = sim.agents[:1]  # unter MIN_POPULATION → Respawn im nächsten Tick
    sim.step()
    assert len(sim.agents) > 1, "Respawn muss stattgefunden haben"
    for a in sim.agents:
        assert a.physics_v2 is True and isinstance(a.body, Body)
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_embodiment.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'physics_v2'`
bzw. `ImportError: cannot import name 'attach_body'`.

- [ ] **Step 3: Implementieren**

**`agent.py`** — Import-Block ergänzen (nach Zeile 17, bei den anderen `environment`-Imports):

```python
from artificial_society.environment.physics.actions import enforce_carry_budget
from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG, Body, Hands
```

Dataclass-Felder (nach `_recent_action_seq: list = field(default_factory=list)`, Zeile 303):

```python
    # Physik v2 (Plan 3a): Flag + Embodiment. Default False/None ⇒ v1 byte-gleich.
    physics_v2: bool = False
    body: object = None
    hands: object = None
```

Helper direkt nach `ensure_fields` (nach Zeile 176) einfügen:

```python
def attach_body(agent) -> None:
    """Physik-v2-Embodiment (Plan 3a): Body + Hände mit Default-Kraft 0.5.

    Das strength-Gen ersetzt den Default in Plan 3b; die Körpermasse ist real
    geankert (BODY_MASS_DEFAULT_KG, cal 'body_mass'). Zieht keine RNG.
    """
    agent.physics_v2 = True
    agent.body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.5)
    agent.hands = Hands()
```

In `ensure_fields` am Ende (nach dem `_disease_immunity`-Guard, Zeile 175–176) anhängen:

```python
    # --- Physik v2 (Plan 3a) ---
    if not hasattr(agent, "physics_v2"):
        agent.physics_v2 = False
    if not hasattr(agent, "body"):
        agent.body = None
    if not hasattr(agent, "hands"):
        agent.hands = None
    if agent.physics_v2 and agent.body is None:
        agent.body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.5)
    if agent.physics_v2 and agent.hands is None:
        agent.hands = Hands()
```

In `Agent.update`, direkt nach `self._sleep_tick(mods)` (Zeile 1060) einfügen:

```python
        if self.physics_v2 and self.body is not None:
            # Körper-Mechanik pro Tick (B4): Tragen ermüdet, Ruhe erholt;
            # Überlast (Ermüdung senkt die Kapazität) wirft zu Tick-Beginn das
            # jeweils schwerste gehaltene Objekt ab. Rein mechanisch, kein Reward.
            if self.hands.held:
                self.body.carry_tick(self.hands.carried_mass_kg())
            else:
                self.body.rest_tick()
            enforce_carry_budget(self.body, self.hands, world.objects, self.pos)
```

**`simulation.py`** — `__init__`-Signatur (Zeile 101–111) erweitern:

```python
    def __init__(
        self,
        width=1200,
        height=800,
        grid_w=60,
        grid_h=40,
        initial_population=36,
        headless=False,
        seed=None,
        load_checkpoint=True,
        physics_v2=False,
    ):
```

und direkt nach `self.seed = seed` (Zeile 114) einfügen:

```python
        # Physik v2 (Plan 3a): additive Objekt-Schicht. Default False = v1 byte-gleich.
        self.physics_v2 = bool(physics_v2)
```

Import ergänzen (Zeile 8): `attach_body` in die agent-Importliste aufnehmen:

```python
from artificial_society.agents.agent import (
    CORPSE_ENERGY,
    MAX_ENERGY,
    Agent,
    attach_body,
    ensure_fields,
)
```

`spawn_initial_population` (Zeile 157–160) ersetzen durch:

```python
    def spawn_initial_population(self, n):
        for _ in range(n):
            x, y = self.world.random_land_position()
            agent = Agent.spawn_random(x, y)
            if self.physics_v2:
                attach_body(agent)
            self.agents.append(agent)
```

In `spawn_child_from_parent`, nach `child.birth_tick = self.tick` (Zeile 173) einfügen:

```python
        if self.physics_v2:
            attach_body(child)
```

In `emergency_respawn` (Zeile 244–249), nach `a.birth_tick = self.tick` einfügen:

```python
            if self.physics_v2:
                attach_body(a)
```

- [ ] **Step 4: Tests + Determinismus-Kontrakt verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_embodiment.py tests/test_headless.py tests/test_regression_golden.py -q`
Expected: alle grün — Golden beweist: Flag aus = byte-gleich.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/agents/agent.py simulation.py tests/test_physics_v2_embodiment.py
git commit -m "feat(sim): physics_v2-Kwarg + Body/Hands-Embodiment (strength 0.5) + Ueberlast-Drop im Tick"
```

---

### Task 13: Registriertes System `"physics_v2_objects"` + Start-Seeding

Verwesung und Spawning hängen sich als registriertes System in den Tick — kein verstreutes
`if physics_v2:` in der Tick-Schleife (Muster `systems/registry.py` / `systems/_builtins.py`).

**Files:**
- Create: `artificial_society/systems/physics_v2.py`
- Modify: `simulation.py:151-155` (`__init__`, Frisch-Start-Zweig: Seeding)
- Test: `tests/test_physics_v2_system.py` (neu)

**Interfaces:**
- Consumes: `register_system` aus `systems.registry`; `seed_initial`, `tick_spawn`, `tick_decay`
  aus `environment.phys_objects` (Tasks 6/7); `sim.physics_v2`, `sim.world.objects`
- Produces: System `"physics_v2_objects"` (order 27 — nach world_regrowth 25, vor tribes 30),
  erreichbar als `sim.physics_v2_objects` und in `sim.systems`; tickt NUR bei `sim.physics_v2`.
  `Simulation.__init__` seedet die Objekt-Schicht beim Frisch-Start (nicht beim Checkpoint-Load).

- [ ] **Step 1: Failing Tests schreiben** — neue Datei `tests/test_physics_v2_system.py`:

```python
"""D2 Registry-Integrations-Smoke-Test: Objekt-Schicht läuft nachweislich im Sim-Tick."""

from __future__ import annotations

import pytest

from artificial_society.environment.physics.actions import DECAY_RATE
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=30, grid_h=20, initial_population=8)


def test_v2_sim_registriert_system_und_seedet_objekte():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    assert "physics_v2_objects" in sim.systems
    assert sim.world.objects.total_mass() > 0.0, "Start-Seeding muss Objekte liefern"
    assert sim.world.objects.ledger["spawned"] == pytest.approx(sim.world.objects.total_mass())
    for _ in range(3):
        sim.step()  # Spawning/Verwesung laufen im Tick, ohne zu crashen


def test_verwesung_laeuft_ueber_den_sim_tick():
    """Kein direkter Mechanik-Aufruf: der Massenverlust muss aus sim.step() kommen."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    kadaver = make_object("carcass", 50.0)
    sim.world.objects.add(kadaver, (5, 5), source="from_carcass")
    sim.step()
    assert kadaver.mass == pytest.approx(50.0 * (1.0 - DECAY_RATE))


def test_gegenprobe_flag_aus_system_wirkt_nicht():
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    assert "physics_v2_objects" in sim.systems  # registriert, aber inert
    assert sim.world.objects.total_mass() == 0.0  # kein Seeding
    kadaver = make_object("carcass", 50.0)
    sim.world.objects.add(kadaver, (5, 5), source="from_carcass")
    for _ in range(3):
        sim.step()
    assert kadaver.mass == 50.0  # keine Verwesung
    boden = [o for o, _ in sim.world.objects.all_objects()]
    assert boden == [kadaver]  # kein Spawning
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_system.py -q`
Expected: FAIL — `KeyError`/`AssertionError`: `"physics_v2_objects" in sim.systems` ist False.

- [ ] **Step 3: Implementieren**

Neue Datei `artificial_society/systems/physics_v2.py`:

```python
"""Physik-v2-Objektschicht als registriertes System (Plan 3a, Spec A/B2/B3.4).

Der Pro-Tick-Anteil der Objekt-Schicht (Verwesung, dann Spawning) läuft über
den System-Registry-Seam statt als if-Verzweigung in der Tick-Schleife. Bei
physics_v2=False ist der Tick ein sofortiger No-op (kein RNG-Draw — Golden).
"""

from __future__ import annotations

from artificial_society.environment.phys_objects import tick_decay, tick_spawn
from artificial_society.systems.registry import register_system


class PhysicsV2ObjectsSystem:
    """Zustandsloses System; die Objekt-Schicht selbst lebt an der Welt."""

    def tick(self, sim, tick: int) -> None:
        if not getattr(sim, "physics_v2", False):
            return
        layer = sim.world.objects
        tick_decay(layer)
        tick_spawn(layer, sim.world.biomes)


def _tick(sim, tick: int) -> None:
    sim.physics_v2_objects.tick(sim, tick)


# order 27: nach world_regrowth (25), vor tribes (30) — Objekte altern/erscheinen
# im selben Tick, in dem auch die Zell-Ökologie fortgeschrieben wurde.
register_system("physics_v2_objects", lambda sim: PhysicsV2ObjectsSystem(), order=27, tick=_tick)
```

In `simulation.py` `__init__`, den Frisch-Start-Zweig (Zeile 151–154) ersetzen durch:

```python
        if load_checkpoint and os.path.exists(CHECKPOINT_PATH):
            self._load_checkpoint()
        else:
            self.spawn_initial_population(initial_population)
            if self.physics_v2:
                # Start-Seeding NUR beim Frisch-Start (ein geladener Checkpoint
                # bringt seine Objekt-Schicht mit); zieht RNG nur bei Flag an.
                seed_initial(self.world.objects, self.world.biomes)
```

Import ergänzen (bei den anderen `environment`-Imports, nach Zeile 10):

```python
from artificial_society.environment.phys_objects import seed_initial
```

- [ ] **Step 4: Tests + Determinismus-Kontrakt verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_system.py tests/test_regression_golden.py tests/test_headless.py -q`
Expected: alle grün. (Falls `test_v2_sim_registriert_system_und_seedet_objekte` wegen fehlender
geeigneter Biome bei seed=42 kein Objekt seedet: anderen Seed wählen — deterministisch, einmal
grün = immer grün. 600 Zellen enthalten praktisch sicher Grasland/Wald.)

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/systems/physics_v2.py simulation.py tests/test_physics_v2_system.py
git commit -m "feat(systems): registriertes System physics_v2_objects (Verwesung+Spawning im Tick) + Start-Seeding"
```

---

### Task 14: Kadaver-Umleitung — `remove_dead()`, Pre-Filter-Aufhebung, Loot = 0

**Der Pfad ist heute real tot:** `step()` filtert Tote VOR `remove_dead()` heraus
(`simulation.py:488`, `self.agents = [a for a in self.agents if a.alive]`) — alle Update-Tode
(Verhungern, Krankheit, Alter, Attack-Kill) erreichen `remove_dead()` nie. Im v2-Modus entfällt
dieses Pre-Filtering; bei Flag aus bleibt es unverändert (Golden). Zusätzlich: Attack-Loot
(`agent.py:631-632`) = 0 im v2-Modus — sonst würde derselbe Tod doppelt gemünzt.

**Files:**
- Modify: `simulation.py:251-259` (`remove_dead` + neuer Helper `_spawn_carcass`), `:486-489` (`step`, Pre-Filter), Import-Block
- Modify: `artificial_society/agents/agent.py:629-633` (`_attack`, Loot-Zweig)
- Test: `tests/test_physics_v2_death.py` (neu)

**Interfaces:**
- Consumes: `make_object` aus `physics.objects`; `BODY_MASS_DEFAULT_KG` aus `physics.body`;
  `world.objects` (Task 5); `agent.physics_v2/body/hands` (Task 12)
- Produces: `Simulation._spawn_carcass(agent) -> None` — genau EIN Kadaver-Objekt
  (`kind="carcass"`, `mass = body.body_mass`) an der Todesposition, gehaltene Objekte fallen zu
  Boden (ledger-neutral), Ledger-Quelle `"from_carcass"`. v1-Zweig unverändert
  (`add_carcass(self.world, *agent.pos, CORPSE_ENERGY)`).

- [ ] **Step 1: Failing Tests schreiben** — neue Datei `tests/test_physics_v2_death.py`:

```python
"""D1: Kadaver-Einmal-Münzung über sim.step(), Verzweigungstest Flag aus, Kill ohne Doppel-Energie."""

from __future__ import annotations

import math

import pytest

from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _carcasses_at(sim, pos):
    return [o for o in sim.world.objects.objects_at(pos) if o.kind == "carcass"]


def test_v2_tod_im_step_muenzt_genau_einen_kadaver():
    """Der Test läuft über sim.step() (Agent stirbt im eigenen Update), NICHT über
    direkten remove_dead()-Aufruf — sonst testet er den per Pre-Filtering
    umgangenen Pfad grün (Spec B3/D1)."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    opfer = sim.agents[0]
    gehalten = [make_object("granite", 2.0), make_object("flint", 1.0)]
    opfer.hands.held.extend(gehalten)
    sim.world.objects.ledger["spawned"] += 3.0  # Handbestückung bilanzieren (Testaufbau)
    opfer.health = 0.01
    opfer.energy = 0.0  # verhungert im nächsten Update
    pos = opfer.pos
    pool_vorher = sim.world.get_cell(*pos)["carcasses"]

    sim.step()

    assert opfer not in sim.agents
    kadaver = _carcasses_at(sim, pos)
    assert len(kadaver) == 1, "genau EIN Kadaver-Objekt"
    # rel=1e-3: das registrierte System verwest den Kadaver im SELBEN Tick um
    # genau einen DECAY_RATE-Schritt (≈ 2.9e-4 relativ) — das ist korrekt.
    assert kadaver[0].mass == pytest.approx(BODY_MASS_DEFAULT_KG, rel=1e-3)
    assert sim.world.get_cell(*pos)["carcasses"] <= pool_vorher + 1e-9, (
        "kein v1-Zell-Credit im v2-Modus (Pool darf höchstens von selbst zerfallen)"
    )
    for obj in gehalten:  # gehaltene Objekte liegen am Boden
        assert sim.world.objects.position_of(obj) is not None
    haende = sum(a.hands.carried_mass_kg() for a in sim.agents if a.physics_v2)
    lhs, rhs = sim.world.objects.conservation_terms(held_mass_kg=haende)
    assert math.isclose(lhs, rhs, rel_tol=1e-9)


def test_verzweigung_flag_aus_v1_pfad_unveraendert():
    """D1: direkter Assert des v1-Zweigs (Golden beweist nur den Default-Pfad,
    nicht die Verzweigung) — daher direkter remove_dead()-Aufruf."""
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    toter = sim.agents[0]
    toter.alive = False
    pos = toter.pos
    pool_vorher = sim.world.get_cell(*pos)["carcasses"]

    sim.remove_dead()

    assert toter not in sim.agents
    assert sim.world.get_cell(*pos)["carcasses"] > pool_vorher  # CORPSE_ENERGY-Zell-Credit
    assert sim.world.objects.total_mass() == 0.0  # KEIN PhysObject


def _erzwungener_kill(sim):
    angreifer, ziel = sim.agents[0], sim.agents[1]
    ziel.pos = angreifer.pos  # adjazent (Radius 1 schließt dieselbe Zelle ein)
    angreifer.genes["aggression"] = 5.0  # threshold 4.7 → random.random() > 4.7 nie → Angriff sicher
    angreifer.trust[ziel.id] = -1.0
    ziel.health = 1.0
    rueckgabe = angreifer._attack(sim.agents, {})
    assert not ziel.alive, "Testaufbau: der Angriff muss töten"
    return angreifer, rueckgabe


def test_v2_kill_ohne_doppel_energie():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    energie_vorher = sim.agents[0].energy
    angreifer, rueckgabe = _erzwungener_kill(sim)
    assert angreifer.energy == pytest.approx(energie_vorher), "kein Loot im v2-Modus"
    assert rueckgabe == 0.0
    sim.remove_dead()  # Energie aus dem Kill gibt es ausschließlich über den Kadaver
    assert any(o.kind == "carcass" for o, _ in sim.world.objects.all_objects())


def test_v1_kill_vergibt_weiterhin_loot():
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    ziel_energie = sim.agents[1].energy
    energie_vorher = sim.agents[0].energy
    angreifer, rueckgabe = _erzwungener_kill(sim)
    erwartet = min(240.0, energie_vorher + ziel_energie * 0.3)  # MAX_ENERGY-Klemme
    assert angreifer.energy == pytest.approx(erwartet)
    assert rueckgabe == pytest.approx(ziel_energie * 0.3)
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_death.py -q`
Expected: FAIL — `test_v2_tod_im_step_muenzt_genau_einen_kadaver` findet 0 Kadaver
(Pre-Filter frisst den Toten) und `test_v2_kill_ohne_doppel_energie` sieht Loot.

- [ ] **Step 3: Implementieren**

**`simulation.py`** — Imports ergänzen (bei den bestehenden physics-Imports aus Task 12/13):

```python
from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG
from artificial_society.environment.physics.objects import make_object
```

`remove_dead` (Zeile 251–259) ersetzen durch:

```python
    def remove_dead(self):
        survivors = []
        for agent in self.agents:
            if agent.alive:
                survivors.append(agent)
                continue
            self._broadcast_death_knowledge(agent)
            if self.physics_v2:
                # v2 (Spec B3): der Tod münzt genau EIN Kadaver-Objekt —
                # kein add_carcass-Credit auf Zell-Pools, kein Loot.
                self._spawn_carcass(agent)
            else:
                add_carcass(self.world, *agent.pos, CORPSE_ENERGY)
        self.agents = survivors

    def _spawn_carcass(self, agent):
        """Kadaver-Objekt an der Todesposition; gehaltene Objekte fallen zu Boden.

        Ledger: Handmasse → Boden ist neutral (war schon in der Invariante),
        die Körpermasse fließt als 'from_carcass' zu (Erhaltung, Spec D1).
        """
        layer = self.world.objects
        hands = getattr(agent, "hands", None)
        if hands is not None:
            for obj in list(hands.held):
                hands.release(obj)
                layer.add(obj, agent.pos)
        body = getattr(agent, "body", None)
        body_mass = body.body_mass if body is not None else BODY_MASS_DEFAULT_KG
        layer.add(make_object("carcass", body_mass), agent.pos, source="from_carcass")
```

In `step()` das Pre-Filtering (Zeile 484–489) ersetzen durch:

```python
        # Flag aus: Pre-Filtering unverändert (Golden-Garantie) — Tote erreichen
        # remove_dead() dort wie bisher nicht. Im v2-Modus MÜSSEN Tote
        # remove_dead() erreichen (Kadaver-Objekt + Erhaltung, Spec B3): ohne
        # diesen Fix entstünden nie Kadaver — Massenleck im Ledger, und die
        # Kern-Kette Kadaver→Schneiden→Essen existierte nicht.
        if not self.physics_v2:
            self.agents = [a for a in self.agents if a.alive]
        self.remove_dead()
```

**`agent.py`** — in `_attack`, den Kill-Zweig (Zeile 629–633) ersetzen durch:

```python
        if target.health <= 0:
            target.alive = False
            if self.physics_v2:
                # v2 (Spec B3): Energie aus einem Kill gibt es AUSSCHLIESSLICH
                # über den Kadaver — Loot wäre Energie ohne Massen-Gegenwert
                # (Doppel-Münzung, für den Ledger unsichtbar).
                return 0.0
            loot = target.energy * 0.3
            self.energy = min(MAX_ENERGY, self.energy + loot)
            return loot
```

- [ ] **Step 4: Tests + Determinismus-Kontrakt verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_death.py tests/test_regression_golden.py tests/test_headless.py -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add simulation.py artificial_society/agents/agent.py tests/test_physics_v2_death.py
git commit -m "feat(sim): Kadaver-Umleitung in remove_dead + Pre-Filter-Aufhebung + Loot=0 im v2-Modus"
```

---

### Task 15: B6-Mechanik-Schalter im v2-Modus

B6 trennt pro System Mechanik/Belohnung. In 3a werden NUR die Mechanik-Schalter umgelegt
(Reward-Spalte = Plan 3b, Global Constraint 4): Zell-Fleisch/Aas-Foraging AUS (Fleisch existiert
nur noch als Kadaver-Objekt), v1-Erfindung AUS (beide Trigger-Pfade + Goal-Stack-Planner),
Kochen AUS, Hamilton-Energie-Umverteilung AUS. Ökologie, Kampf, Territorium, Kooperation,
Sprache, Stämme, Krankheit, Endokrin, Handel laufen unverändert weiter.

**Files:**
- Modify: `artificial_society/agents/agent.py:523` (`_forage`, Pflanzen-Zweig), `:1127` (Goal-Stack-Block), `:1201-1209` (need-Invention), `:1211-1216` (Zufalls-Invention), `:1218-1222` (Kochen)
- Modify: `simulation.py:303-311` (`_apply_hamilton_rewards`, Guard)
- Test: `tests/test_physics_v2_switches.py` (neu)

**Interfaces:**
- Consumes: `agent.physics_v2` (Task 12), `sim.physics_v2` (Task 12)
- Produces: keine neuen Symbole — Verhaltensschalter. Plan 3b baut die Reward-Spalte auf
  dieser Trennung auf.

- [ ] **Step 1: Failing Tests schreiben** — neue Datei `tests/test_physics_v2_switches.py`:

```python
"""B6-Mechanik-Spalte im v2-Modus: Fleisch-Zellpools, v1-Erfindung, Kochen, Goal-Stack,
Hamilton — AUS. Reward-Spalte bleibt unangetastet (Plan 3b)."""

from __future__ import annotations

import pytest

import artificial_society.agents.agent as agent_mod
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def test_v2_forage_laesst_fleisch_zellpools_unangetastet():
    """B6: Zell-Fleisch/Aas-Pools AUS — ein einziger Pfad für Fleischkalorien (Kadaver-Objekte)."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    agent.genes["diet_preference"] = 1.0  # Karnivor — würde in v1 zuerst Aas/Fleisch nehmen
    x, y = agent.pos
    sim.world.set_cell(x, y, "carcasses", 50.0)
    sim.world.set_cell(x, y, "meat_food", 50.0)
    sim.world.set_cell(x, y, "plant_food", 20.0)

    agent._forage(sim.world, {})

    cell = sim.world.get_cell(x, y)
    assert cell["carcasses"] == pytest.approx(50.0)
    assert cell["meat_food"] == pytest.approx(50.0)
    assert cell["plant_food"] < 20.0  # Pflanzen-Zell-Foraging bleibt AN


def test_v1_forage_unveraendert():
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    agent = sim.agents[0]
    agent.genes["diet_preference"] = 1.0
    x, y = agent.pos
    sim.world.set_cell(x, y, "carcasses", 50.0)
    agent._forage(sim.world, {})
    assert sim.world.get_cell(x, y)["carcasses"] < 50.0


def _spy(monkeypatch, name, rueckgabe=None):
    aufrufe = []

    def _zaehler(*args, **kwargs):
        aufrufe.append(args)
        return rueckgabe

    monkeypatch.setattr(agent_mod, name, _zaehler)
    return aufrufe


def test_v2_erfindung_kochen_goalstack_tot(monkeypatch):
    """B6: beide Invention-Trigger-Pfade, Kochen und der Goal-Stack-Planner feuern nie."""
    inv_need = _spy(monkeypatch, "agent_invent_from_need")
    inv_rand = _spy(monkeypatch, "agent_try_invention")
    kochen = _spy(monkeypatch, "agent_try_cook")
    # (None, 0.0) statt None: im Failing-Lauf (Schalter existiert noch nicht) wird
    # der Spy aufgerufen und sein Rückgabewert in agent.py:1127–1128 entpackt
    # (`goal_action, goal_shaping = ...`) — so schlägt unten sauber die
    # Spy-Assertion fehl statt eines Unpack-TypeError.
    goals = _spy(monkeypatch, "agent_tick_with_goals", rueckgabe=(None, 0.0))

    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    for _ in range(13):  # deckt tick%3- und tick%4-Fenster mehrfach ab
        sim.step()

    assert inv_need == [] and inv_rand == [] and kochen == [] and goals == []


def test_v2_hamilton_umverteilung_aus():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    a, b = sim.agents[0], sim.agents[1]
    a.tribe_id = b.tribe_id = 1
    a.energy, b.energy = 200.0, 10.0
    sim.tick = 20  # HAMILTON_TICK_INTERVAL
    sim._apply_hamilton_rewards()
    assert a.energy == 200.0 and b.energy == 10.0
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_switches.py -q`
Expected: FAIL mit `AssertionError` in allen vier Tests — v2-Forage frisst den Aas-Pool,
die Spies zählen Aufrufe (der goals-Spy gibt `(None, 0.0)` zurück, damit das Unpack in
`agent.py:1127–1128` im Failing-Lauf nicht mit einem TypeError crasht), Hamilton bewegt Energie.

- [ ] **Step 3: Implementieren**

**`agent.py`, `_forage`** — Zeile 523 (`if diet < 0:`) ersetzen durch:

```python
        if diet < 0 or self.physics_v2:
            # Herbivore (v1) bzw. Physik-v2-Modus: nur Pflanzen-Zell-Foraging.
            # v2 (B6): Zell-Fleisch/Aas-Pools sind AUS — Fleisch existiert nur
            # noch als Kadaver-OBJEKT (B3); ein einziger Pfad je Kalorienquelle.
```

(Der bestehende Herbivoren-Block darunter bleibt wörtlich unverändert; auch der
`else:`-Karnivoren-Zweig bleibt unverändert — er ist im v2-Modus schlicht unerreichbar.)

**`agent.py`, `update`** — Goal-Stack-Block, Zeile 1127 ersetzen:

```python
        if not self.physics_v2 and getattr(self, "goal_stack", None) is not None:
```

need-Invention-Block (Zeile 1201–1209) ersetzen durch:

```python
        if not self.physics_v2:
            # v2 (B6): v1-Erfindung AUS — beide Trigger-Pfade entfallen mitsamt
            # ihren Boni; Entdecken läuft künftig über die Objekt-Physik (3b).
            if self._need_inv_cooldown <= 0:
                compute_need_vector(self, current_cell)
                inv_result = agent_invent_from_need(self, world, *self.pos, tick)
                if inv_result:
                    reward += 0.5
                    self.endocrine.apply_discovery(1.0)
                self._need_inv_cooldown = NEED_INVENTION_INTERVAL
            else:
                self._need_inv_cooldown -= 1
```

Zufalls-Invention (Zeile 1211–1216): die `if`-Zeile ersetzen durch:

```python
        if not self.physics_v2 and tick % 3 == 0 and random.random() < inv_prob:
```

Kochen (Zeile 1218): die `if`-Zeile ersetzen durch:

```python
        if not self.physics_v2 and tick % 4 == 0 and random.random() < 0.18:
```

**`simulation.py`, `_apply_hamilton_rewards`** — direkt nach dem Docstring-Kommentar
(vor `if self.tick % HAMILTON_TICK_INTERVAL != 0:`, Zeile 310) einfügen:

```python
        if self.physics_v2:
            # v2 (B6/Architekturtabelle): keine Verwandten-Energie-Umverteilung.
            return
```

- [ ] **Step 4: Tests + Determinismus-Kontrakt verifizieren**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_switches.py tests/test_regression_golden.py -q`
Expected: alle grün (v1-Pfade wörtlich unverändert ⇒ Golden grün).

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/agents/agent.py simulation.py tests/test_physics_v2_switches.py
git commit -m "feat(agents): B6-Mechanik-Schalter im v2-Modus — Fleisch-Zellpools, v1-Erfindung, Kochen, Goal-Stack, Hamilton aus"
```

---

### Task 16: Checkpoint — `physics_v2` im Payload + Guard VOR dem broad-except

Heute würde ein werfender Guard vom broad-`except` in `_load_checkpoint` still verschluckt und
„frisch gestartet“ — stiller Datenverlust statt harter Schranke. Der Flag-Mismatch-Check läuft
deshalb über eine dedizierte Exception, die VOR dem broad-`except` gefangen und **re-raised** wird.
(`CHECKPOINT_FORMAT_VERSION = 2` kommt erst mit der 3b-Architekturänderung — Spec C5.)

**Files:**
- Modify: `simulation.py:343-360` (`_save_checkpoint`), `:362-383` (`_load_checkpoint`), Modul-Ebene (Exception-Klasse nach Zeile 53)
- Test: `tests/test_physics_v2_checkpoint.py` (neu)

**Interfaces:**
- Consumes: `sim.physics_v2` (Task 12), `world.objects`-Pickling (Task 4), `ensure_fields`-Embodiment (Task 12)
- Produces: `CheckpointIncompatibleError(RuntimeError)` in `artificial_society.simulation`;
  Checkpoint-Payload-Key `"physics_v2": bool`. Alte Checkpoints (ohne Key) laden mit
  `physics_v2=False` weiter (Default-Kompatibilität), kollidieren aber hart mit `physics_v2=True`.

- [ ] **Step 1: Failing Tests schreiben** — neue Datei `tests/test_physics_v2_checkpoint.py`:

```python
"""C5 (3a-Anteil): physics_v2 im Checkpoint-Payload; Mismatch re-raised VOR dem broad-except."""

from __future__ import annotations

import pickle

import pytest

import artificial_society.simulation as sim_mod
from artificial_society.simulation import CheckpointIncompatibleError, Simulation

_PARAMS = dict(headless=True, grid_w=20, grid_h=15, initial_population=8)


@pytest.fixture()
def checkpoint_path(tmp_path, monkeypatch):
    pfad = str(tmp_path / "checkpoint.pkl")
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", pfad)
    return pfad


def test_roundtrip_v2_erhaelt_flag_objekte_und_embodiment(checkpoint_path):
    sim1 = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim1.step()
    sim1.step()
    masse = sim1.world.objects.total_mass()
    sim1._save_checkpoint()

    sim2 = Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)
    assert sim2.tick == sim1.tick
    assert sim2.world.objects.total_mass() == pytest.approx(masse)
    assert all(a.physics_v2 and a.body is not None for a in sim2.agents)


def test_flag_mismatch_re_raised_statt_still_verschluckt(checkpoint_path):
    sim1 = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim1._save_checkpoint()
    with pytest.raises(CheckpointIncompatibleError):
        Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)


def test_alt_checkpoint_ohne_key_laedt_nur_mit_flag_aus(checkpoint_path):
    """Legacy-Payload (kein physics_v2-Key) ⇒ implizit False: lädt mit Flag aus,
    kollidiert hart mit Flag an."""
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"agents": [], "tick": 5}, f)
    sim = Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)
    assert sim.tick == 5
    with pytest.raises(CheckpointIncompatibleError):
        Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)


def test_kaputter_checkpoint_startet_weiterhin_frisch(checkpoint_path):
    """Der broad-except-Pfad (korrupte Datei) bleibt erhalten: frisch starten."""
    with open(checkpoint_path, "wb") as f:
        f.write(b"kein pickle")
    sim = Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)
    assert sim.tick == 0 and len(sim.agents) == 8
```

- [ ] **Step 2: Tests laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_checkpoint.py -q`
Expected: FAIL — `ImportError: cannot import name 'CheckpointIncompatibleError'`.

- [ ] **Step 3: Implementieren** — in `simulation.py`:

Nach den Modul-Konstanten (nach `SIDEBAR_W = 300`, Zeile 53) einfügen:

```python
class CheckpointIncompatibleError(RuntimeError):
    """Checkpoint passt nicht zur angeforderten Konfiguration (z. B. physics_v2-Mismatch).

    Wird in _load_checkpoint VOR dem broad-except re-raised: still verschlucken
    und „frisch starten“ wäre stiller Datenverlust (Spec C5).
    """
```

In `_save_checkpoint` (Zeile 343–360) das Payload-Dict um einen Key erweitern — nach
`"tick": self.tick,` einfügen:

```python
                        "physics_v2": self.physics_v2,
```

`_load_checkpoint` (Zeile 362–383) ersetzen durch:

```python
    def _load_checkpoint(self):
        try:
            with open(CHECKPOINT_PATH, "rb") as f:
                data = pickle.load(f)
            saved_flag = bool(data.get("physics_v2", False))
            if saved_flag != self.physics_v2:
                raise CheckpointIncompatibleError(
                    f"checkpoint physics_v2={saved_flag} != Simulation physics_v2="
                    f"{self.physics_v2} — Checkpoint löschen oder Flag angleichen"
                )
            self.agents = data.get("agents", [])
            self.tick = data.get("tick", 0)
            self.world = data.get("world", self.world)
            # Checkpoints from before the struct-of-arrays cell storage hold
            # plain dict cells — migrate them in place (no-op for new saves).
            self.world.ensure_array_storage()
            self.stats = data.get("stats", self.stats)
            self.tribes = data.get("tribes", self.tribes)
            self.technology = data.get("technology", self.technology)
            # Restore the global emergent registries; absent in pre-Phase-3 saves,
            # in which case the freshly-reset singletons are kept.
            _restore_singletons(data.get("registries", {}))
            for agent in self.agents:
                ensure_fields(agent)
            print(f"[checkpoint] loaded tick={self.tick}, agents={len(self.agents)}")
        except CheckpointIncompatibleError:
            raise  # harte Schranke — NICHT vom broad-except verschlucken lassen
        except Exception as e:
            print(f"[checkpoint] load failed: {e} — starting fresh")
            self.spawn_initial_population(self._initial_population)
```

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/test_physics_v2_checkpoint.py tests/test_regression_golden.py -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add simulation.py tests/test_physics_v2_checkpoint.py
git commit -m "feat(sim): physics_v2 im Checkpoint-Payload + harter Mismatch-Guard vor dem broad-except"
```

---

### Task 17: Totes Alt-Modul `systems/world_objects.py` → `archive/`

Das komplett unverdrahtete Alt-Modul (toter Vorläufer, referenziert nicht-existente Methoden,
keine Importe im Repo — verifiziert per grep) kollidiert namentlich/konzeptionell mit der neuen
ObjectLayer und wandert nach `archive/` (Spec A, „Aufräumen im Zuge von 3a“; der Umzug ist der
explizite Owner-Auftrag aus der Spec).

**Files:**
- Move: `artificial_society/systems/world_objects.py` → `archive/world_objects.py`

**Interfaces:**
- Consumes: — (keine Konsumenten)
- Produces: — (das Modul verschwindet aus dem Registry-Discovery-Scan von
  `artificial_society.systems`; `archive/` wird nie importiert)

- [ ] **Step 1: Verifizieren, dass niemand importiert**

Run: `grep -rn "world_objects" --include="*.py" artificial_society tests scripts | grep -v "artificial_society/systems/world_objects.py"`
Expected: keine Treffer (nur das Modul selbst existiert).

- [ ] **Step 2: Verschieben**

```bash
git mv artificial_society/systems/world_objects.py archive/world_objects.py
```

- [ ] **Step 3: Volle Suite laufen lassen**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest -q`
Expected: alle grün — insbesondere crasht `registry.discover()` nicht (Discovery skippt
fehlende/fehlerhafte Module ohnehin, und dieses hier ist schlicht weg).

- [ ] **Step 4: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add -A archive/ artificial_society/systems/
git commit -m "chore(systems): totes Alt-Modul world_objects.py nach archive/ verschoben (Kollision mit ObjectLayer)"
```

---

### Task 18: Metriken-Hooks (D4, nach 3a messbar)

Reines Logging, kein Verhalten: Fragmente (kumulativ; „je Tick“ = Delta zwischen Snapshots),
Schnitte mit Werkzeug vs. bloßer Hand, gegessene kcal je Quelle (`kind`-Label — reines
Diagnose-Label, nie Physik-Input), Ledger-Flüsse.

**Files:**
- Modify: `artificial_society/environment/phys_objects.py` (`ObjectLayer.__init__`, `__setstate__`, `metrics_snapshot`)
- Modify: `artificial_society/environment/physics/actions.py` (`do_strike`/`do_cut`/`do_eat`: je 1–3 Zähl-Zeilen)
- Test: `tests/environment/physics/test_actions.py` (erweitern)

**Interfaces:**
- Consumes: `ActionResult`-Pfade aus Tasks 9–11
- Produces:
  - `layer.metrics: dict` mit Keys `"fragments_total": int`, `"cuts_with_tool": int`,
    `"cuts_bare_hand": int`, `"kcal_eaten_by_kind": dict[str, float]`
  - `layer.metrics_snapshot() -> dict` — Kopie von Metrics + `"ledger"`-Kopie (Pilot-Logging, Plan 5)

- [ ] **Step 1: Failing Test schreiben** — in `tests/environment/physics/test_actions.py` anhängen:

```python
def test_metriken_hooks_zaehlen_ohne_verhalten():
    """D4 (nach 3a messbar): Fragmente, Schnitte Werkzeug vs. Hand, kcal je Quelle,
    Ledger-Snapshot — reine Zähler, kein Verhalten."""
    body, hands, layer = _setup(strength=0.5)
    hammer = make_object("granite", 0.5)
    flint = make_object("flint", 0.8)
    fleisch_a = make_object("raw_meat", 0.5)
    fleisch_b = make_object("raw_meat", 0.5)
    for obj in (hammer, flint, fleisch_a, fleisch_b):
        layer.add(obj, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), hammer)

    res = do_strike(body, hands, layer, (5, 5), hammer, flint, 0.8, random.Random(42))
    assert layer.metrics["fragments_total"] == len(res.fragments) > 0

    # ACHTUNG: do_cut ERSETZT das Ziel (remainder ist ein neues Objekt) —
    # deshalb zwei getrennte Fleisch-Objekte für die zwei Schnitte.
    schnitt_hand = do_cut(body, hands, layer, (5, 5), None, fleisch_a, effort=0.2)
    assert schnitt_hand.ok and schnitt_hand.extracted is not None
    klinge = _klinge(0.3)
    layer.add(klinge, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), klinge)
    schnitt_klinge = do_cut(body, hands, layer, (5, 5), klinge, fleisch_b, effort=0.2)
    assert schnitt_klinge.ok
    assert layer.metrics["cuts_bare_hand"] == 1
    assert layer.metrics["cuts_with_tool"] == 1

    biss = do_eat(body, hands, layer, (5, 5), schnitt_hand.extracted)
    assert biss.ok
    kcal = layer.metrics["kcal_eaten_by_kind"]
    assert kcal["raw_meat_piece"] == pytest.approx(0.35 * 4000.0 * biss.bite_kg)

    snap = layer.metrics_snapshot()
    assert snap["ledger"] == layer.ledger and snap["ledger"] is not layer.ledger
    assert snap["fragments_total"] == layer.metrics["fragments_total"]
```

- [ ] **Step 2: Test laufen lassen, FAIL verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q -k metriken`
Expected: FAIL — `AttributeError: 'ObjectLayer' object has no attribute 'metrics'`.

- [ ] **Step 3: Implementieren**

**`phys_objects.py`, `ObjectLayer.__init__`** — nach der `self.ledger = ...`-Zeile einfügen:

```python
        # D4-Metriken (reines Logging, kein Verhalten): kumulative Zähler;
        # „je Tick“ ergibt sich als Delta zwischen zwei metrics_snapshot()-Aufrufen.
        self.metrics: dict = {
            "fragments_total": 0,
            "cuts_with_tool": 0,
            "cuts_bare_hand": 0,
            "kcal_eaten_by_kind": {},
        }
```

**`phys_objects.py`, `__setstate__`** — am Ende der Methode ergänzen (alte Pickles ohne Metrics):

```python
        self.__dict__.setdefault(
            "metrics",
            {"fragments_total": 0, "cuts_with_tool": 0, "cuts_bare_hand": 0, "kcal_eaten_by_kind": {}},
        )
```

**`phys_objects.py`** — Methode in `ObjectLayer` ergänzen (nach `conservation_terms`):

```python
    def metrics_snapshot(self) -> dict:
        """Kopie aller D4-Zähler + Ledger-Flüsse (Pilot-Logging, Plan 5)."""
        snap = {
            k: (dict(v) if isinstance(v, dict) else v) for k, v in self.metrics.items()
        }
        snap["ledger"] = dict(self.ledger)
        return snap
```

**`actions.py`, `do_strike`** — im Bruch-Zweig, direkt vor dem abschließenden
`return ActionResult(...)`, einfügen:

```python
    layer.metrics["fragments_total"] += len(result.fragments)
```

**`actions.py`, `do_cut`** — im Erfolgs-Zweig, direkt vor dem abschließenden
`return ActionResult(...)`, einfügen:

```python
    layer.metrics["cuts_with_tool" if blade_held is not None else "cuts_bare_hand"] += 1
```

**`actions.py`, `do_eat`** — direkt vor `return ActionResult(...)` (nach der
`layer.ledger["eaten"] += bite`-Zeile) einfügen:

```python
    kcal_by_kind = layer.metrics["kcal_eaten_by_kind"]
    kcal_by_kind[target.kind] = (
        kcal_by_kind.get(target.kind, 0.0) + nutrition * KCAL_PER_KG_PER_NUTRITION * bite
    )
```

- [ ] **Step 4: Tests laufen lassen, PASS verifizieren**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/ -q`
Expected: alle grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add artificial_society/environment/phys_objects.py artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
git commit -m "feat(physics): D4-Metriken-Hooks — Fragmente, Schnitte Werkzeug/Hand, kcal je Quelle, Ledger-Snapshot"
```

---

### Task 19: Massen-Ledger-Fuzzer — 50 Seeds × 200 Aktionen (D1)

Die Erhaltungsaussage ist die härteste Garantie des Designs und darf nicht von der Seed-Wahl des
Autors abhängen. Kein Hypothesis im Repo — handgeschriebener Fuzzer mit `random.Random(seed)`.
Aktions-Menge (Spec D1): {spawn, grasp, release, strike, cut, eat, decay-tick, overload-drop,
agent-stirbt-mit-2-Objekten}; Invariante nach JEDER Aktion.

**Files:**
- Test: `tests/environment/test_conservation_fuzz.py` (neu; reine Test-Datei, kein Produktionscode)

**Interfaces:**
- Consumes: `ObjectLayer`, `tick_decay` (Tasks 4/7); `do_grasp/do_release/do_strike/do_cut/do_eat`,
  `enforce_carry_budget` (Tasks 8–11); `Body`, `Hands`, `BODY_MASS_DEFAULT_KG`; `make_object`
- Produces: — (Beweis-Test)

- [ ] **Step 1: Fuzzer schreiben** — neue Datei `tests/environment/test_conservation_fuzz.py`:

```python
"""D1-Massen-Ledger-Fuzzer: 50 Seeds × 200 zufällige Aktionen, Invariante nach JEDER Aktion.

Invariante: boden + hände + ledger[eaten] + ledger[decayed] == ledger[spawned] + ledger[from_carcass].
Sim-frei: ein geskripteter Körper (Body+Hands) treibt die do_*-Funktionen direkt (Spec Plan-Schnitt 3a).
"""

from __future__ import annotations

import math
import random

from artificial_society.environment.phys_objects import ObjectLayer, tick_decay
from artificial_society.environment.physics.actions import (
    do_cut,
    do_eat,
    do_grasp,
    do_release,
    do_strike,
    enforce_carry_budget,
)
from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG, Body, Hands
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.props import IDX2

_GRID = 8
_MATERIALIEN = ("granite", "flint", "dry_wood", "plant_fiber", "clay_moist", "carcass", "raw_meat")
_AKTIONEN = ("spawn", "grasp", "release", "strike", "cut", "eat", "decay", "overload", "die")


def _assert_invariante(layer: ObjectLayer, hands: Hands, seed: int, schritt: int, aktion: str):
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9), (
        f"Ledger-Leck: seed={seed} schritt={schritt} aktion={aktion}: {lhs} != {rhs}"
    )


def _fuzz_ein_seed(seed: int, n_aktionen: int = 200) -> None:
    rng = random.Random(seed)
    layer = ObjectLayer(_GRID, _GRID, rng=rng)
    body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.5)
    hands = Hands()

    for schritt in range(n_aktionen):
        pos = (rng.randrange(_GRID), rng.randrange(_GRID))
        aktion = rng.choice(_AKTIONEN)
        boden_nah = [o for o, _ in layer.objects_near(pos, 1)]
        boden_hier = layer.objects_at(pos)

        if aktion == "spawn":
            layer.add(
                make_object(rng.choice(_MATERIALIEN), rng.uniform(0.05, 8.0)),
                (rng.randrange(_GRID), rng.randrange(_GRID)),
                source="spawned",
            )
        elif aktion == "grasp" and boden_nah:
            do_grasp(body, hands, layer, pos, rng.choice(boden_nah))
        elif aktion == "release" and hands.held:
            do_release(body, hands, layer, pos, rng.choice(hands.held))
        elif aktion == "strike" and hands.held:
            striker = rng.choice(hands.held)
            ziele = boden_hier + [o for o in hands.held if o is not striker]
            if ziele:
                do_strike(body, hands, layer, pos, striker, rng.choice(ziele), rng.random(), rng)
        elif aktion == "cut":
            klinge = rng.choice(hands.held + [None]) if hands.held else None
            ziele = boden_hier + [o for o in hands.held if o is not klinge]
            if ziele:
                do_cut(body, hands, layer, pos, klinge, rng.choice(ziele), rng.random())
        elif aktion == "eat":
            ziele = boden_hier + list(hands.held)
            if ziele:
                do_eat(body, hands, layer, pos, rng.choice(ziele))
        elif aktion == "decay":
            tick_decay(layer)
        elif aktion == "overload":
            body.fatigue = min(1.0, body.fatigue + 0.5)
            enforce_carry_budget(body, hands, layer, pos)
        elif aktion == "die":
            # Agent stirbt mit (bis zu) 2 gehaltenen Objekten: Hände fallen zu
            # Boden (neutral), Körper wird als Kadaver gemünzt (from_carcass) —
            # exakt der remove_dead()-Pfad, sim-frei nachgestellt.
            for obj in list(hands.held):
                hands.release(obj)
                layer.add(obj, pos)
            layer.add(make_object("carcass", body.body_mass), pos, source="from_carcass")
            body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.5)  # „nächster“ Agent
            hands = Hands()

        _assert_invariante(layer, hands, seed, schritt, aktion)


def test_ledger_fuzzer_50_seeds_x_200_aktionen():
    for seed in range(50):
        _fuzz_ein_seed(seed)


def test_ledger_smoke_beispielkette():
    """Schneller Smoke-Check der Kern-Kette: spawn → grasp → strike → cut → eat → decay."""
    rng = random.Random(99)
    layer = ObjectLayer(_GRID, _GRID, rng=rng)
    body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.7)
    hands = Hands()
    hammer = make_object("granite", 1.0)
    flint = make_object("flint", 0.8)
    kadaver = make_object("carcass", 25.0)
    layer.add(hammer, (4, 4), source="spawned")
    layer.add(flint, (4, 4), source="spawned")
    layer.add(kadaver, (4, 4), source="from_carcass")

    do_grasp(body, hands, layer, (4, 4), hammer)
    schlag = do_strike(body, hands, layer, (4, 4), hammer, flint, 1.0, rng)
    assert schlag.fragments
    klinge = max(schlag.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))
    do_release(body, hands, layer, (4, 4), hammer)
    do_grasp(body, hands, layer, (4, 4), klinge)
    schnitt = do_cut(body, hands, layer, (4, 4), klinge, kadaver, 0.8)
    assert schnitt.extracted is not None
    assert do_eat(body, hands, layer, (4, 4), schnitt.extracted).ok
    tick_decay(layer)
    _assert_invariante(layer, hands, 99, -1, "beispielkette")
```

- [ ] **Step 2: Fuzzer laufen lassen**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_conservation_fuzz.py -q`
Expected: 2 passed (Laufzeit wenige Sekunden). Schlägt die Invariante fehl, nennt die
Assertion-Message Seed/Schritt/Aktion — den Bug IMMER in der Mechanik fixen
(Tasks 8–11/7), NIE die Toleranz aufweichen.

- [ ] **Step 3: Volle Suite als Regressionsschutz**

Run: `rm -f checkpoint.pkl && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../venv/bin/python -m pytest -q`
Expected: alle grün.

- [ ] **Step 4: Commit**

```bash
../venv/bin/ruff check --fix . && ../venv/bin/ruff format .
git add tests/environment/test_conservation_fuzz.py
git commit -m "test(physics): D1-Massen-Ledger-Fuzzer — 50 Seeds x 200 Aktionen, Invariante nach jeder Aktion"
```

---

### Task 20: Endabnahme — voller Gate-Lauf

Kein neuer Code; beweist, dass der Branch merge-fähig ist (Spiegel des CI-Gates).

**Files:** — (nur Verifikation)

**Interfaces:** —

- [ ] **Step 1: Stales Checkpoint entfernen und vollen Check fahren**

```bash
rm -f checkpoint.pkl
bash scripts/check.sh
```

Expected: `OK — check passed (predicts green CI).` — insbesondere:
Golden-Trajektorie grün (Flag aus = byte-gleich), Headless-Digest grün,
Realitäts-Gate grün (alle neuen Konstanten kalibriert + Doc synchron), Fuzzer grün.

- [ ] **Step 2: Doc-Sync gegenprüfen**

```bash
../venv/bin/python scripts/gen_kalibrierung.py && git status --short docs/physics/kalibrierung.md
```

Expected: keine Ausgabe von `git status` (Datei unverändert ⇒ committetes Artefakt ist synchron).

- [ ] **Step 3: End-to-End-Rauchprobe v2 (manuell, headless)**

```bash
rm -f checkpoint.pkl
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONHASHSEED=0 ../venv/bin/python -c "
from artificial_society.simulation import Simulation
sim = Simulation(headless=True, seed=42, grid_w=30, grid_h=20, initial_population=12,
                 load_checkpoint=False, physics_v2=True)
for _ in range(200):
    sim.step()
print('objekte:', sum(1 for _ in sim.world.objects.all_objects()))
print('snapshot:', sim.world.objects.metrics_snapshot())
"
rm -f checkpoint.pkl
```

Expected: läuft ohne Exception durch; `objekte:` > 0; Snapshot zeigt Ledger-Flüsse
(`spawned` > 0; `from_carcass` > 0, falls in 200 Ticks jemand starb). Das zweite
`rm -f checkpoint.pkl` räumt den bei Tick 500 zwar nicht erreichten, aber
grundsätzlich möglichen Auto-Save weg (Hygiene).

- [ ] **Step 4: Kein Commit** — Task 20 ändert nichts; der Branch ist damit review-bereit.
  Abschluss (Merge/PR) läuft über die superpowers:finishing-a-development-branch-Routine des
  Plan-Owners, nicht über diesen Plan.

---

## Selbst-Review (Spec-Abdeckung 3a)

| Spec-Punkt | Task(s) |
|---|---|
| A: v2 additive Schicht, Flag-Kwarg + Agent-Attribut | 12 |
| A: Registry-Integration statt Tick-Schleifen-Ifs | 13 |
| A: `world_objects.py` → `archive/` | 17 |
| B1: ObjectLayer-API, Validierung, Ledger, sparse, GPU-Garantie | 4, 5 |
| B1: per-Welt DiscoveryV2 + Singleton-Entfernung | 1, 4, 5 |
| B2: Spawning-Tabelle, Dichten/Raten, kind `"spawn"`, Gate-Erweiterung | 6 |
| B3: Kadaver-Umleitung, Pre-Filter-Aufhebung, Loot=0, gehaltene Objekte fallen | 14 |
| B3.4: Verwesung (Masse, nutrition↓, toxicity↑ Kappe 0.6, `ledger[decayed]`) | 7, 13 |
| B4: do_grasp/do_release (MAX_HELD, Budget) | 8 |
| B4: do_strike (½mv²-Kappung 14 m/s, exert-Bindung) | 9 |
| B4: do_cut (beidseitiger Klingen-Faktor, Effort-Ertrag yield×(0.5+0.5·effort), CUT_WORK_J = 15+35·effort) | 10 |
| B4: do_eat (BITE 0.3, nutrition ≤ 0.02 No-op) | 11 |
| B4: Überlast-Drop zu Tick-Beginn | 8, 12 |
| B5: SIM_ENERGY_PER_KCAL 0.032, Produkt-Test ±1, Kadaver 0.14/0.02, TOX_DAMAGE 20, MUSCLE_EFFICIENCY 0.25 | 3, 7, 11 |
| B6 Mechanik-Spalte (Fleisch-Zellpools, v1-Erfindung, Kochen, Goal-Stack, Hamilton) | 15 |
| C5-3a-Anteil: Checkpoint-Payload + Guard vor broad-except, re-raise | 16 |
| D1: Fuzzer 50×200, Mikro-Invarianten, Kadaver-Einmal-Münzung über `sim.step()`, Verzweigungstest Flag aus, Kill ohne Doppel-Energie | 19, 9, 10, 14 |
| D2: Kiesel-Kappung, Knapping-Positiv-Regression, Klingen-Faktor, exert-Bindung, Überlast, Essen 45±1/Granit, Verwesung, Gate-Erweiterung, Discovery pro Welt, Registry-Smoke inkl. Gegenprobe | 9, 10, 8, 11, 7, 3, 6, 5, 13 |
| D4 (nach 3a messbar): Fragmente, Schnitte Werkzeug/Hand, kcal je Quelle, Ledger-Flüsse | 18 |

**Bewusst NICHT in diesem Plan** (siehe Datei-Landkarte oben): Abschnitt C, D3,
B6-Reward-Spalte (Plan 3b), `strength`-Gen, `CHECKPOINT_FORMAT_VERSION = 2`,
GPU-Batching der Objekt-Schicht (Tier-5-Folgeaufgabe), Verwesung gehaltener Objekte
(das registrierte System wirkt auf den Boden-Layer; Spec definiert Verwesung über
dieses System — bekannte, dokumentierte Vereinfachung).

**Spec-Synchronisation (Spec Rev. 4, team-entschieden 2026-07-02):** Die ursprünglich
plan-internen Entscheidungen sind in die Spec zurückgeschrieben und damit normativ —
Spec und Plan sind deckungsgleich:
1. Kind `"action"` für `CALIBRATED_ACTION_PARAMS` (Spec B2 ↔ Tasks 2/3).
2. Arbeits-Metabolik ohne `×1000` (Spec B5 ↔ Task 3; Anker „200 Schläge ≈ 0.3“ per Test fixiert).
3. Kiesel-Präzisierung: max ~4.9 J < Schwelle typischer Knollen ≥ 0.55 kg; Winz-Knollen
   darf er brechen (Spec B4 ↔ Task-9-Test prüft Kappung + 0.8-kg-Flint).
4. Cut-Ertrag skaliert mit Effort: `yield × (0.5 + 0.5·effort)` (Spec B4 ↔ Task 3 cal-Text
   `cut_work` + Task 10 Implementierung/Tests inkl. 2×-Assert; Tasks 11/18/19 geprüft —
   deren Erwartungswerte sind ertragsdynamisch bzw. effort-konstant, keine Anpassung nötig).
5. Exert-Bindung normativ: ausgeführter Schlag = Exert (auch wirkungslos), unerreichbares/
   ungültiges Ziel = No-op ohne Exert (Spec B4 ↔ Task 9).
6. Verwesungs-Gate eigenschaftsbasiert `moisture ≥ DECAY_MOISTURE_MIN (0.5) ∧ nutrition > 0`,
   kalibrierte Konstante kind `action` (Spec B3 ↔ Tasks 3/7).

**Verbleibende plan-interne Entscheidung** (Golden-neutral, kein Spec-Widerspruch):
`World` trägt die (leere) ObjectLayer auch bei Flag aus — konstruktionsseitig RNG-frei.

## Ausführungs-Hinweise

- Reihenfolge einhalten: 1→2→3 sind Fundament (Kinds/Konstanten), 4→5 Welt, 6→7 Spawning/Verwesung,
  8→11 Aktionen, 12→16 Sim-Integration, 17–20 Aufräumen/Beweis. Kein Task überspringbar;
  Task N kompiliert nicht ohne N−1 (Imports).
- Zeilennummern beziehen sich auf den Stand `main @ f37f31d`; nach eigenen Edits verschieben sie
  sich — die zitierten Code-Anker (exakte Zeileninhalte) sind maßgeblich.
- Bei jedem Task, der `Simulation` konstruiert: vorher `rm -f checkpoint.pkl` (Global Constraint 6).








