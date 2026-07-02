# Embodiment v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Die realen Körper-Grenzen (Spec §4: Tragkraft, zwei Hände, Kraft→Schlagenergie,
Ermüdung) als standalone-Modul im Physik-v2-Paket — so kalibriert, dass „Schneiden mit
Kante ≫ mit bloßer Hand" und „ganzer Kadaver untragbar → zerteilen lohnt" messbare,
getestete Tatsachen sind (der Erfindungsdruck).

**Architecture:** Neues Modul `artificial_society/environment/physics/body.py` — der
Körper ist die physische Hülle des Agenten und gehört zur Weltphysik (das Gehirn bleibt
in `agents/`). `Body` (Masse, Kraft, Ermüdung → Tragkapazität, Schlagenergie) und `Hands`
(max. 2 gehaltene `PhysObject`s, Masse-Budget) sind reine, deterministische Datenklassen
ohne Sim-Integration (die kommt im Lern-Kopplungs-Plan). Die Kalibrierungstabelle bekommt
ein viertes Kind `"body"`; das Realitäts-Gate erzwingt Anker auch für Körper-Parameter.

**Tech Stack:** Python 3.9, numpy (nur transitiv), pytest, ruff. Kein torch, kein pygame.

## Global Constraints

- **Basis-Branch:** `main` @ `97c468f` (Physik-v2-Kern gemerged). Arbeits-Branch: `feat/environment-body-v1` (eigener Worktree).
- **Hot-File-Contract:** `simulation.py`, `world.py`, `agents/agent.py`, `agents/brain.py`, `environment/materials.py`, `systems/registry.py` werden NICHT angefasst. Geändert werden nur: `environment/physics/calibration.py` (kein Hot File), neues `environment/physics/body.py`, der Gate-Test, neue Testdateien, `docs/physics/kalibrierung.md` (generiert).
- **Golden Trajectory bleibt unberührt** — nichts hier wird von der Sim importiert.
- **Determinismus:** Body/Hands sind vollständig deterministisch — KEIN Zufall, kein Modul-level RNG-State.
- **Realitäts-Gate:** Jeder neue Körper-Parameter braucht einen `cal("body", ...)`-Eintrag mit realem Anker + Quelle. **Jede Task, die `cal()`-Einträge hinzufügt, MUSS danach `../venv/bin/python scripts/gen_kalibrierung.py` laufen lassen und das regenerierte `docs/physics/kalibrierung.md` mitcommitten** — sonst wird `test_kalibrierung_doc_is_in_sync` rot.
- **Python 3.9:** `from __future__ import annotations` in jeder Datei; keine 3.10+-Runtime-Syntax.
- **Stil:** ruff-clean; nach dem Implementieren `../venv/bin/python -m ruff check --fix <dateien>` und `../venv/bin/python -m ruff format <dateien>` laufen lassen (der Auto-Hook greift in Subagenten nicht zuverlässig).
- **Tests laufen mit:** `../venv/bin/python -m pytest <datei> -q` vom Worktree-Root.
- **Commit-Stil:** `env(body-v1): <beschreibung>` + Zeile `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Spec-§4-Abdeckung:** Tragkraft, Greifen (2 Hände), Kraft→Schlagenergie und Ermüdung baut
dieser Plan. *Reichweite* (nur aktuelle Zelle) gilt in der Sim bereits — kein Standalone-Baustein
nötig. *Verletzbarkeit* wird bewusst in den Lern-Kopplungs-Plan verschoben (die Sim hat bereits
`health`; realistisch anbinden lässt sie sich erst dort).

**Kalibrierte Referenzwerte, auf denen alle Testzahlen beruhen** (aus dem gemergten
Physik-Kern, nicht ändern): Feuerstein-Kern 0.8 kg bricht ab 7.2 J; Granit 0.8 kg ab
28.8 J; Klinge-auf-Kadaver-Ertrag pro Schnitt 7.8–13.0 % der Masse, bloße Hand 0.89 %.

---

### Task 1: Body-Kern + Kalibrierungs-Kind „body"

**Files:**
- Create: `artificial_society/environment/physics/body.py`
- Modify: `artificial_society/environment/physics/calibration.py` (3 Stellen: `VALID_KINDS`, `_KIND_TITLES`, Import in `render_markdown`)
- Modify: `tests/environment/physics/test_reality_gate.py` (Body-Vollständigkeit + Orphan-Zweig)
- Modify: `docs/physics/kalibrierung.md` (regeneriert)
- Test: `tests/environment/physics/test_body.py`

**Interfaces:**
- Consumes: `cal(kind, name, anchor, source)` aus `calibration.py`
- Produces: `Body(body_mass: float, strength: float, fatigue: float = 0.0)` (dataclass, `__post_init__` validiert `body_mass > 0`, `strength`/`fatigue` ∈ [0,1], sonst `ValueError`); `Body.carry_capacity_kg() -> float`; `Body.strike_energy_j(effort: float) -> float`; Konstanten `CARRY_FRACTION_SUSTAINED = 0.30`, `STRIKE_ENERGY_MIN_J = 5.0`, `STRIKE_ENERGY_MAX_J = 50.0`; `CALIBRATED_BODY_PARAMS: tuple` (wächst in Tasks 2/3); `VALID_KINDS` enthält `"body"`.

- [ ] **Step 1: Failing Tests schreiben**

`tests/environment/physics/test_body.py`:

```python
"""Body v1: Tragkapazität und Schlagenergie aus Masse, Kraft, Ermüdung."""

from __future__ import annotations

import pytest

from artificial_society.environment.physics.body import (
    CARRY_FRACTION_SUSTAINED,
    STRIKE_ENERGY_MAX_J,
    Body,
)


def test_carry_capacity_baseline():
    # 70-kg-Körper, volle Kraft, ausgeruht: 30 % des Körpergewichts.
    assert Body(body_mass=70.0, strength=1.0).carry_capacity_kg() == pytest.approx(21.0)
    assert CARRY_FRACTION_SUSTAINED == 0.30


def test_carry_capacity_scales_with_strength_and_fatigue():
    weak = Body(body_mass=70.0, strength=0.5)
    assert weak.carry_capacity_kg() == pytest.approx(16.8)  # 21 * (0.6+0.4*0.5)
    tired = Body(body_mass=70.0, strength=1.0, fatigue=1.0)
    assert tired.carry_capacity_kg() == pytest.approx(10.5)  # 21 * 0.5


def test_strike_energy_baseline_and_anchor_range():
    # Kräftiger Schlag eines starken, ausgeruhten Körpers = oberes Anker-Ende (50 J).
    strong = Body(body_mass=70.0, strength=1.0)
    assert strong.strike_energy_j(effort=1.0) == pytest.approx(STRIKE_ENERGY_MAX_J)
    # Minimaler Effort bleibt über 0 (5 J * Kraftfaktor).
    assert strong.strike_energy_j(effort=0.0) == pytest.approx(5.0)


def test_strike_energy_scales_and_clamps_effort():
    body = Body(body_mass=70.0, strength=0.7)
    assert body.strike_energy_j(effort=1.0) == pytest.approx(42.5)  # 50 * 0.85
    assert body.strike_energy_j(effort=2.0) == body.strike_energy_j(effort=1.0)
    assert body.strike_energy_j(effort=-1.0) == body.strike_energy_j(effort=0.0)


def test_fatigue_dampens_strike_energy():
    exhausted = Body(body_mass=70.0, strength=1.0, fatigue=1.0)
    assert exhausted.strike_energy_j(effort=1.0) == pytest.approx(20.0)  # 50 * 0.4


def test_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        Body(body_mass=0.0, strength=0.5)
    with pytest.raises(ValueError):
        Body(body_mass=70.0, strength=1.5)
    with pytest.raises(ValueError):
        Body(body_mass=70.0, strength=0.5, fatigue=-0.1)
```

Zusätzlich in `tests/environment/physics/test_reality_gate.py`:
1. Import ergänzen: `from artificial_society.environment.physics.body import CALIBRATED_BODY_PARAMS`
2. Neue Testfunktion (nach `test_every_process_is_calibrated`):

```python
def test_every_body_param_is_calibrated():
    for name in CALIBRATED_BODY_PARAMS:
        _assert_calibrated("body", name)
```

3. Im Orphan-Test `test_no_orphan_calibration_entries` einen Zweig ergänzen:

```python
        elif kind == "body":
            assert name in CALIBRATED_BODY_PARAMS, f"verwaister body-Eintrag: {name}"
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_body.py tests/environment/physics/test_reality_gate.py -q`
Expected: FAIL/ERROR mit `ModuleNotFoundError: No module named 'artificial_society.environment.physics.body'`

- [ ] **Step 3: Implementierung**

`artificial_society/environment/physics/body.py`:

```python
"""Körper v1 — die physische Hülle des Agenten (Spec §4, Embodiment).

Der Körper gehört zur Weltphysik: reale Grenzen (Tragkraft, Kraft, Ermüdung)
erzeugen den Erfindungsdruck — ein Werkzeug lohnt sich genau deshalb, weil der
nackte Körper an diesen Grenzen scheitert. Vollständig deterministisch, kein
Zufall. Sim-Integration (Gene→strength, Energie-Kopplung) kommt im
Lern-Kopplungs-Plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import cal

# Körper-Physik (reale Anker: siehe cal()-Einträge unten)
CARRY_FRACTION_SUSTAINED = 0.30
STRIKE_ENERGY_MIN_J = 5.0
STRIKE_ENERGY_MAX_J = 50.0

# Vom Realitäts-Gate geprüfte Körper-Parameter (wächst mit Tasks 2/3).
CALIBRATED_BODY_PARAMS = ("carry_capacity", "strike_energy")


@dataclass
class Body:
    body_mass: float  # kg
    strength: float  # [0,1] relative Kraft (1 ≈ kräftiger Erwachsener)
    fatigue: float = 0.0  # [0,1] (0 = ausgeruht)

    def __post_init__(self) -> None:
        if self.body_mass <= 0:
            raise ValueError("body_mass muss > 0 sein")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength muss in [0,1] liegen")
        if not 0.0 <= self.fatigue <= 1.0:
            raise ValueError("fatigue muss in [0,1] liegen")

    def carry_capacity_kg(self) -> float:
        """Dauerhaft tragbare Masse: ~30 % des Körpergewichts, skaliert mit
        Kraft, gedämpft durch Ermüdung."""
        return (
            CARRY_FRACTION_SUSTAINED
            * self.body_mass
            * (0.6 + 0.4 * self.strength)
            * (1.0 - 0.5 * self.fatigue)
        )

    def strike_energy_j(self, effort: float) -> float:
        """Aufprallenergie eines Handschlags mit Werkzeugstein (5–50 J),
        skaliert mit Kraft, gedämpft durch Ermüdung."""
        effort = min(max(effort, 0.0), 1.0)
        base = STRIKE_ENERGY_MIN_J + (STRIKE_ENERGY_MAX_J - STRIKE_ENERGY_MIN_J) * effort
        return base * (0.5 + 0.5 * self.strength) * (1.0 - 0.6 * self.fatigue)


cal(
    "body",
    "carry_capacity",
    "Dauer-Tragfähigkeit ≈ 30 % des Körpergewichts (Trekking-Richtwert 20–25 %, "
    "militärisches Marschgepäck 30–45 % mit Ermüdungsfolgen); skaliert mit Kraft, "
    "gedämpft durch Ermüdung",
    "Ergonomie-/Militär-Richtwerte zum Lastentragen",
)
cal(
    "body",
    "strike_energy",
    "Schlagenergie eines Handschlags mit Werkzeugstein 5–50 J (deckungsgleich mit dem "
    "Anker des Prozesses strike: kräftiger Handschlag 10–50 J); skaliert mit Kraft, "
    "gedämpft durch Ermüdung",
    "Biomechanik des Hammerschlags; experimentelle Archäologie",
)
```

`artificial_society/environment/physics/calibration.py` — drei chirurgische Änderungen:
1. Zeile 14: `VALID_KINDS = ("dim", "material", "process")` → `VALID_KINDS = ("dim", "material", "process", "body")`
2. Im `_KIND_TITLES`-Dict (ab Zeile 127) den Eintrag ergänzen: `"body": "Körper-Parameter",`
3. In `render_markdown()` (Zeile 145) den Import erweitern:
   `from . import materials_v2, processes  # noqa: F401` → `from . import body, materials_v2, processes  # noqa: F401`

Dann Doku regenerieren:

Run: `../venv/bin/python scripts/gen_kalibrierung.py`
Expected: `wrote .../docs/physics/kalibrierung.md` (neue Sektion „Körper-Parameter" mit 2 Zeilen)

- [ ] **Step 4: Tests laufen lassen — müssen grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: `44 passed` (37 bestehende + 6 Body + 1 Gate-Body-Test)

- [ ] **Step 5: ruff + Commit**

```bash
../venv/bin/python -m ruff check --fix artificial_society/environment/physics/ tests/environment/physics/ && ../venv/bin/python -m ruff format artificial_society/environment/physics/ tests/environment/physics/
git add artificial_society/environment/physics/ tests/environment/physics/ docs/physics/kalibrierung.md
git commit -m "env(body-v1): Body-Kern — Tragkraft + Schlagenergie, Kalibrierungs-Kind body

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Ermüdung (exert / rest / carry)

**Files:**
- Modify: `artificial_society/environment/physics/body.py` (Konstanten + 3 Methoden + cal-Eintrag + `CALIBRATED_BODY_PARAMS` erweitern)
- Modify: `docs/physics/kalibrierung.md` (regeneriert)
- Test: `tests/environment/physics/test_body.py` (Tests anhängen)

**Interfaces:**
- Consumes: `Body` aus Task 1
- Produces: `Body.exert_strike(energy_j: float) -> None` (fatigue += energy·`FATIGUE_PER_JOULE`, geklemmt auf ≤ 1); `Body.rest_tick() -> None` (fatigue −= `FATIGUE_RECOVERY_PER_TICK`, geklemmt auf ≥ 0); `Body.carry_tick(carried_mass_kg: float) -> None` (fatigue += `CARRY_FATIGUE_PER_TICK_AT_CAPACITY` · Auslastung); Konstanten `FATIGUE_PER_JOULE = 0.0001`, `FATIGUE_RECOVERY_PER_TICK = 0.02`, `CARRY_FATIGUE_PER_TICK_AT_CAPACITY = 0.002`; `CALIBRATED_BODY_PARAMS = ("carry_capacity", "strike_energy", "fatigue")`.

- [ ] **Step 1: Failing Tests anhängen**

An `tests/environment/physics/test_body.py` anhängen (Import oben ergänzen: `FATIGUE_RECOVERY_PER_TICK` mit in die bestehende Import-Liste aufnehmen):

```python
def test_exert_strike_accumulates_and_clamps():
    body = Body(body_mass=70.0, strength=1.0)
    body.exert_strike(45.0)
    assert body.fatigue == pytest.approx(0.0045)
    for _ in range(1000):
        body.exert_strike(45.0)
    assert body.fatigue == 1.0  # geklemmt


def test_two_hundred_strong_strikes_reach_exhaustion_anchor():
    # Anker: ~200 kräftige Schläge (45 J) bis deutliche Erschöpfung (~0.9).
    body = Body(body_mass=70.0, strength=1.0)
    for _ in range(200):
        body.exert_strike(45.0)
    assert body.fatigue == pytest.approx(0.9)


def test_rest_recovers_and_clamps_at_zero():
    body = Body(body_mass=70.0, strength=1.0, fatigue=1.0)
    body.rest_tick()
    assert body.fatigue == pytest.approx(1.0 - FATIGUE_RECOVERY_PER_TICK)
    for _ in range(200):
        body.rest_tick()
    assert body.fatigue == 0.0


def test_carry_tick_scales_with_load():
    body = Body(body_mass=70.0, strength=1.0)
    body.carry_tick(body.carry_capacity_kg())  # Volllast
    assert body.fatigue == pytest.approx(0.002)
    fresh = Body(body_mass=70.0, strength=1.0)
    fresh.carry_tick(body.carry_capacity_kg() / 2)  # Halblast — Achtung: Kapazität des ermüdeten body
    assert 0.0 < fresh.fatigue < 0.002
    idle = Body(body_mass=70.0, strength=1.0)
    idle.carry_tick(0.0)
    assert idle.fatigue == 0.0
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_body.py -q`
Expected: FAIL mit `AttributeError: ... 'Body' object has no attribute 'exert_strike'` (bzw. ImportError für `FATIGUE_RECOVERY_PER_TICK`)

- [ ] **Step 3: Implementierung**

In `body.py` — Konstanten zu den bestehenden, Methoden in die `Body`-Klasse, cal-Eintrag ans Dateiende, `CALIBRATED_BODY_PARAMS` ERSETZEN durch `("carry_capacity", "strike_energy", "fatigue")`:

```python
FATIGUE_PER_JOULE = 0.0001  # ~200 kräftige Schläge (45 J) bis fatigue ~0.9
FATIGUE_RECOVERY_PER_TICK = 0.02  # volle Erholung nach ~50 Ruhe-Ticks
CARRY_FATIGUE_PER_TICK_AT_CAPACITY = 0.002  # Dauerlast an der Traggrenze: ~500 Ticks bis Erschöpfung
```

```python
    def exert_strike(self, energy_j: float) -> None:
        """Ein ausgeführter Schlag ermüdet proportional zur aufgewandten Energie."""
        self.fatigue = min(1.0, self.fatigue + max(energy_j, 0.0) * FATIGUE_PER_JOULE)

    def rest_tick(self) -> None:
        """Ein Tick Ruhe baut Ermüdung ab."""
        self.fatigue = max(0.0, self.fatigue - FATIGUE_RECOVERY_PER_TICK)

    def carry_tick(self, carried_mass_kg: float) -> None:
        """Ein Tick Tragen ermüdet proportional zur Auslastung der Tragkapazität."""
        if carried_mass_kg <= 0.0:
            return
        load = carried_mass_kg / self.carry_capacity_kg()
        self.fatigue = min(1.0, self.fatigue + CARRY_FATIGUE_PER_TICK_AT_CAPACITY * load)
```

```python
cal(
    "body",
    "fatigue",
    "Ermüdung/Erholung, Größenordnungen: ~200 kräftige Schläge bis deutliche Erschöpfung "
    "(geübte Steinschläger arbeiten stundenlang); Dauerlast an der Traggrenze über "
    "Hunderte Ticks tragbar; Erholung in Ruhe über Dutzende Ticks. [Zeitskala "
    "Sim-Tick↔Realzeit bewusst qualitativ, bis die Sim-Integration sie fixiert]",
    "Arbeitsphysiologie (Ermüdung/Erholung beim Lastentragen und repetitiver Arbeit)",
)
```

Dann Doku regenerieren: `../venv/bin/python scripts/gen_kalibrierung.py`

- [ ] **Step 4: Tests laufen lassen — müssen grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: `48 passed` (44 + 4 neue)

- [ ] **Step 5: ruff + Commit**

```bash
../venv/bin/python -m ruff check --fix artificial_society/environment/physics/body.py tests/environment/physics/test_body.py && ../venv/bin/python -m ruff format artificial_society/environment/physics/body.py tests/environment/physics/test_body.py
git add artificial_society/environment/physics/body.py tests/environment/physics/test_body.py docs/physics/kalibrierung.md
git commit -m "env(body-v1): Ermüdung — Schläge und Traglast ermüden, Ruhe erholt

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Hände (Greifen, 2-Objekt-Limit, Masse-Budget)

**Files:**
- Modify: `artificial_society/environment/physics/body.py` (Hands-Klasse + `MAX_HELD` + cal-Eintrag + `CALIBRATED_BODY_PARAMS` erweitern)
- Modify: `docs/physics/kalibrierung.md` (regeneriert)
- Test: `tests/environment/physics/test_hands.py`

**Interfaces:**
- Consumes: `Body` (Tasks 1–2), `PhysObject`/`make_object` aus `objects.py`
- Produces: `Hands()` (dataclass, `held: list` leer initialisiert); `Hands.carried_mass_kg() -> float`; `Hands.can_grasp(obj: PhysObject, body: Body) -> bool` (frei Hand UND Gesamtmasse ≤ Kapazität); `Hands.grasp(obj, body) -> bool`; `Hands.release(obj) -> None` (`ValueError` wenn nicht gehalten — via `list.remove`); Konstante `MAX_HELD = 2`; `CALIBRATED_BODY_PARAMS = ("carry_capacity", "strike_energy", "fatigue", "hands")`.

- [ ] **Step 1: Failing Tests schreiben**

`tests/environment/physics/test_hands.py`:

```python
"""Hände v1: zwei Hände, Masse-Budget — der Transport-Engpass vor der Behälter-Erfindung."""

from __future__ import annotations

import pytest

from artificial_society.environment.physics.body import MAX_HELD, Body, Hands
from artificial_society.environment.physics.objects import make_object


def _body():
    return Body(body_mass=70.0, strength=0.7)  # Kapazität 18.48 kg


def test_whole_carcass_exceeds_capacity():
    # DER Erfindungsdruck: ein 25-kg-Kadaver ist als Ganzes untragbar.
    hands = Hands()
    assert not hands.can_grasp(make_object("carcass", 25.0), _body())


def test_light_pieces_are_graspable_up_to_two_hands():
    hands = Hands()
    body = _body()
    assert hands.grasp(make_object("raw_meat", 3.0), body)
    assert hands.grasp(make_object("flint", 0.8), body)
    assert hands.carried_mass_kg() == pytest.approx(3.8)
    # Dritte Hand gibt es nicht — auch für ein federleichtes Objekt.
    assert not hands.can_grasp(make_object("plant_fiber", 0.05), body)
    assert MAX_HELD == 2


def test_mass_budget_counts_total_load():
    hands = Hands()
    body = _body()
    assert hands.grasp(make_object("granite", 15.0), body)
    # Zweite Hand frei, aber 15 + 5 > 18.48 → nein.
    assert not hands.can_grasp(make_object("granite", 5.0), body)


def test_release_frees_the_hand():
    hands = Hands()
    body = _body()
    stone = make_object("granite", 15.0)
    assert hands.grasp(stone, body)
    hands.release(stone)
    assert hands.carried_mass_kg() == 0.0
    assert hands.grasp(make_object("carcass", 3.0), body)


def test_release_unheld_object_raises():
    with pytest.raises(ValueError):
        Hands().release(make_object("flint", 0.8))
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_hands.py -q`
Expected: FAIL mit `ImportError: cannot import name 'Hands'`

- [ ] **Step 3: Implementierung**

In `body.py` — nach der `Body`-Klasse; `CALIBRATED_BODY_PARAMS` ERSETZEN durch
`("carry_capacity", "strike_energy", "fatigue", "hands")`; Import oben ergänzen:
`from dataclasses import dataclass, field`:

```python
MAX_HELD = 2  # zwei Hände; je Hand ein gehaltenes Objekt


@dataclass
class Hands:
    """Was der Körper ohne erfundene Behälter transportieren kann: höchstens
    zwei gehaltene Objekte, deren Gesamtmasse in die Tragkapazität passt."""

    held: list = field(default_factory=list)

    def carried_mass_kg(self) -> float:
        return sum(obj.mass for obj in self.held)

    def can_grasp(self, obj, body: Body) -> bool:
        if len(self.held) >= MAX_HELD:
            return False
        return self.carried_mass_kg() + obj.mass <= body.carry_capacity_kg()

    def grasp(self, obj, body: Body) -> bool:
        if not self.can_grasp(obj, body):
            return False
        self.held.append(obj)
        return True

    def release(self, obj) -> None:
        self.held.remove(obj)
```

cal-Eintrag ans Dateiende:

```python
cal(
    "body",
    "hands",
    "Zwei Hände, je Hand ein gehaltenes Objekt; Gesamtlast innerhalb der Tragkapazität. "
    "Ohne erfundene Behälter ist Transport damit auf 2 Objekte pro Weg begrenzt — der "
    "reale Druck, aus dem Behälter/Bündel entstanden sind",
    "Menschliche Anatomie; Archäologie früher Trage-/Behältertechnik",
)
```

Dann Doku regenerieren: `../venv/bin/python scripts/gen_kalibrierung.py`

- [ ] **Step 4: Tests laufen lassen — müssen grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: `53 passed` (48 + 5 neue)

- [ ] **Step 5: ruff + Commit**

```bash
../venv/bin/python -m ruff check --fix artificial_society/environment/physics/body.py tests/environment/physics/test_hands.py && ../venv/bin/python -m ruff format artificial_society/environment/physics/body.py tests/environment/physics/test_hands.py
git add artificial_society/environment/physics/body.py tests/environment/physics/test_hands.py docs/physics/kalibrierung.md
git commit -m "env(body-v1): Hände — 2-Objekt-Limit + Masse-Budget als Transport-Engpass

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Werkzeug-Delta-Kette (Integrationstests) + Paket-API + Gesamt-Gate

**Files:**
- Modify: `artificial_society/environment/physics/__init__.py` (Body/Hands/Konstanten exportieren)
- Test: `tests/environment/physics/test_tool_pressure.py`

**Interfaces:**
- Consumes: alles aus Tasks 1–3 plus `strike`/`cut`/`make_object` aus dem gemergten Kern
- Produces: Paket-Export `from artificial_society.environment.physics import Body, Hands, MAX_HELD` — und den getesteten Beweis des Erfindungsdrucks (Spec §4 Kalibrierungsziel).

- [ ] **Step 1: Failing Tests schreiben**

`tests/environment/physics/test_tool_pressure.py`:

```python
"""Das Kalibrierungsziel der Spec (§4): Körper-Grenzen machen Werkzeuge messbar wertvoll.

Volle Kette ohne jede Sim-Integration: Body → Schlagenergie → Feuerstein schlagen →
Klinge → Kadaver schneiden vs. bloße Hand; plus der Transport-Druck (ganzer Kadaver
untragbar → Zerteilen lohnt) und der Ermüdungs-Effekt auf die Schlagfähigkeit.
"""

from __future__ import annotations

import random

from artificial_society.environment.physics import Body, Hands, cut, make_object, strike
from artificial_society.environment.physics.props import IDX2


def _blade(rng_seed: int = 42):
    result = strike(
        make_object("flint", 0.8),
        make_object("granite", 1.0),
        Body(body_mass=70.0, strength=0.7).strike_energy_j(effort=1.0),  # 42.5 J
        random.Random(rng_seed),
    )
    assert result.fractured
    return max(result.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))


def test_tool_delta_full_chain_body_to_blade():
    # Ein realer Körper kann mit selbst geschlagener Klinge ≫ mehr Fleisch gewinnen.
    blade = _blade()
    hand_cut = cut(make_object("carcass", 25.0), None)
    blade_cut = cut(make_object("carcass", 25.0), blade)
    assert hand_cut.extracted is not None and blade_cut.extracted is not None
    assert blade_cut.extracted.mass > 6 * hand_cut.extracted.mass


def test_transport_pressure_forces_cutting():
    # Ganzer Kadaver: untragbar. Geschnittenes Stück: tragbar. Zerteilen LOHNT.
    body = Body(body_mass=70.0, strength=0.7)
    hands = Hands()
    assert not hands.can_grasp(make_object("carcass", 25.0), body)
    piece = cut(make_object("carcass", 25.0), _blade()).extracted
    assert piece is not None
    assert hands.grasp(piece, body)


def test_fatigue_erodes_knapping_capability():
    # Nach 200 kräftigen Schlägen: Feuerstein geht noch, Granit nicht mehr.
    body = Body(body_mass=70.0, strength=1.0)
    for _ in range(200):
        body.exert_strike(45.0)
    weak_energy = body.strike_energy_j(effort=1.0)  # 50 * (1 - 0.6*0.9) = 23.0 J
    hammer = make_object("granite", 1.2)
    assert not strike(make_object("granite", 0.8), hammer, weak_energy, random.Random(1)).fractured
    assert strike(make_object("flint", 0.8), hammer, weak_energy, random.Random(1)).fractured
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_tool_pressure.py -q`
Expected: FAIL mit `ImportError: cannot import name 'Body' from 'artificial_society.environment.physics'`

- [ ] **Step 3: Paket-API erweitern**

In `artificial_society/environment/physics/__init__.py`: Import-Block ergänzen um

```python
from .body import (
    CARRY_FRACTION_SUSTAINED,
    MAX_HELD,
    STRIKE_ENERGY_MAX_J,
    STRIKE_ENERGY_MIN_J,
    Body,
    Hands,
)
```

und in `__all__` diese sechs Namen ergänzen: `"CARRY_FRACTION_SUSTAINED"`, `"MAX_HELD"`, `"STRIKE_ENERGY_MAX_J"`, `"STRIKE_ENERGY_MIN_J"`, `"Body"`, `"Hands"`.

- [ ] **Step 4: Tests + Gesamt-Gate — müssen grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: `56 passed` (53 + 3 neue)

Run: `bash scripts/check.sh`
Expected: komplett grün (rc=0), inkl. Golden-Trajectory + Headless-Digest — dieser Plan integriert weiterhin nichts in die Sim; ist Golden rot → STOPPEN und melden, keinen Test anfassen.

- [ ] **Step 5: ruff + Commit**

```bash
../venv/bin/python -m ruff check --fix artificial_society/environment/physics/__init__.py tests/environment/physics/test_tool_pressure.py && ../venv/bin/python -m ruff format artificial_society/environment/physics/__init__.py tests/environment/physics/test_tool_pressure.py
git add artificial_society/environment/physics/__init__.py tests/environment/physics/test_tool_pressure.py
git commit -m "env(body-v1): Werkzeug-Delta bewiesen — Klinge, Transport-Druck, Ermüdung end-to-end

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Danach (NICHT Teil dieses Plans)

Laut Spec §6, je eigener Plan: **3. Lern-Kopplung** (Policy wählt Kombination; Causal-Model-
Prototyp aus `archive/` reaktivieren; Shaping-Boni raus; Wahrnehmung über Eigenschaftsvektoren;
Hot Files `agents/brain.py`/`agents/agent.py`; dort auch Gene→`strength`, Sim-Energie↔Body,
Verletzbarkeit) → **4. Kultur-Korrektur** (genetischer Prior, lamarckistische Pfade raus)
→ **5. Meilenstein-1-Pilot** (Emergenz-A/B, GPU-PC). Offene Integrations-Checkliste aus dem
Kern-Review: Energie-Kosten für cut/strike, DiscoveryV2 pro Welt, Runtime-Props-Validierung,
Registry-Threshold.

**Zusätzliche Checklisten-Punkte aus dem Final-Review dieses Branches (für die Lern-Kopplung
deliberat zu entscheiden, nicht wiederzuentdecken):** (a) Schlagstein-Masse ist von der
Schlagenergie entkoppelt — ein 0.05-kg-Kiesel „liefert" 50 J; `strike_energy_j` braucht eine
Kappung über die Schlagstein-Masse (~½·m·v², Handgeschwindigkeit max ~12–15 m/s), Spec §3 nennt
beide Faktoren. (b) Mini-Klingen-Exploit: Fragment-Schärfe und Schneide-Ertrag sind energie- und
werkzeugmasse-unabhängig — 1-J-Tipper auf 0.1-kg-Kiesel liefern Spitzen-Klingen; Kandidat:
Masse-/Größenterm in `effective_sharpness` bzw. Ertrag (Kern-Physik, Entscheidung im Pilot-Lane).
(c) `exert_strike` ist nicht an `strike()` gebunden — im Action-Layer MUSS jeder Schlag ermüden,
sonst ist Arbeit gratis. (d) Überlast-Semantik: Kapazität wird nur beim Greifen geprüft;
Fallenlassen bei geschrumpfter Kapazität gehört der Action-Schicht.
