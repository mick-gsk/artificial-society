# Physik v2 Kern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Das real kalibrierte Physik-Fundament (Schnitt 1 der Spec
`docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md`): reale
Eigenschaftsdimensionen, reale Startmaterialien, die Prozesse **Schlagen** und **Schneiden**
als reine Physik-Funktionen, plus das **Realitäts-Gate** (Kalibrierungstabelle als
Merge-Bedingung).

**Architecture:** Neues, in sich geschlossenes Paket
`artificial_society/environment/physics/` — reine Funktionen/Datenklassen ohne
Simulations-Integration (die kommt in Folgeplänen: Embodiment, Gehirn-Kopplung).
Objekte = intensiver Eigenschaftsvektor (13 real geankerte Dimensionen, [0,1]-normiert)
+ extensive Masse in kg. Prozesse leiten Wirkung ausschließlich aus Eigenschaften ab
(nie aus Labels) und erhalten Masse exakt. Eine eigene kleine Discovery-Registry vergibt
stabile IDs für neuartige Vektoren. Die Kalibrierungstabelle lebt als Code
(`calibration.py`, SSOT), wird nach `docs/physics/kalibrierung.md` gerendert und von
einem dauerhaften Gate-Test erzwungen.

**Tech Stack:** Python 3.9, numpy, pytest, ruff. Kein torch, kein pygame.

## Global Constraints

- **Basis-Branch:** `core/foundation` (enthält Phase-4/5-Integration + Golden-Regen + Spec). Arbeits-Branch: `feat/environment-physics-v2` (eigener Worktree empfohlen).
- **Hot-File-Contract:** `simulation.py`, `world.py`, `agents/agent.py`, `agents/brain.py`, `environment/materials.py`, `systems/registry.py` werden in diesem Plan NICHT angefasst. Nur neue Dateien + `docs/roadmap.md` (Task 8, kein Hot File).
- **Golden Trajectory bleibt unberührt** — keine Änderung an bestehendem Verhalten; alles hier ist neuer, unintegrierter Code.
- **Determinismus:** Zufall NUR über einen explizit übergebenen `random.Random`-Parameter. Kein globales `random`/`numpy`-Seeding, kein RNG-State auf Modulebene.
- **Realitäts-Gate:** Jede neue Dimension / jedes Material / jeder Prozess braucht einen `cal(...)`-Eintrag mit realem Anker + Quelle (Spec §2). Task 7 erzwingt das dauerhaft per Test.
- **Python 3.9:** `from __future__ import annotations` in jeder Datei; typisiert; dataclasses; keine 3.10+-Syntax (kein `match`, keine `X | Y`-Typen außerhalb von Annotations).
- **Stil:** ruff-clean (ein PostToolUse-Hook formatiert automatisch); double quotes; Kommentare/Docstrings deutsch (fachbegriffe englisch ok) — konsistent innerhalb des Pakets.
- **Tests laufen mit:** `../venv/bin/python -m pytest <datei> -q` vom Repo-Root (`/Users/moritzbecker/projekt/artificial-society`).
- **Commit-Stil:** `env(physics-v2): <beschreibung>` (+ Co-Authored-By-Zeile wie im Repo üblich).

---

### Task 1: Paket-Skelett, Eigenschafts-Schema, Kalibrierungs-Mechanik

**Files:**
- Create: `artificial_society/environment/physics/__init__.py`
- Create: `artificial_society/environment/physics/props.py`
- Create: `artificial_society/environment/physics/calibration.py`
- Create: `tests/environment/physics/__init__.py`
- Test: `tests/environment/physics/test_props_schema.py`

**Interfaces:**
- Consumes: —
- Produces: `PROP_DIMS_V2: list[str]` (13 Namen), `IDX2: dict[str, int]`, `N_PROPS_V2: int`, `pv(**kwargs: float) -> np.ndarray` (float32, shape `(13,)`); `calibration.cal(kind, name, anchor, source) -> None`, `calibration.entry_for(kind, name) -> CalEntry | None`, `calibration.CALIBRATION: dict[tuple[str, str], CalEntry]`, `CalEntry(kind, name, anchor, source)` (frozen dataclass). Alle 13 Dimensionen sind nach Import von `calibration` bereits als `("dim", <name>)` kalibriert.

- [ ] **Step 1: Test-Verzeichnis + failing Test schreiben**

`tests/environment/physics/__init__.py` — leere Datei.

`tests/environment/physics/test_props_schema.py`:

```python
"""Schema-Tests Physik v2: Dimensionsliste, Index-Map, Vektor-Builder."""

from __future__ import annotations

import numpy as np
import pytest

from artificial_society.environment.physics.calibration import cal, entry_for
from artificial_society.environment.physics.props import IDX2, N_PROPS_V2, PROP_DIMS_V2, pv


def test_schema_has_13_unique_dims():
    assert N_PROPS_V2 == 13
    assert len(set(PROP_DIMS_V2)) == 13
    assert IDX2 == {name: i for i, name in enumerate(PROP_DIMS_V2)}


def test_pv_builds_float32_vector_with_zero_defaults():
    v = pv(hardness=0.7, moisture=0.1)
    assert v.dtype == np.float32
    assert v.shape == (N_PROPS_V2,)
    assert v[IDX2["hardness"]] == pytest.approx(0.7)
    assert v[IDX2["moisture"]] == pytest.approx(0.1)
    assert float(v.sum()) == pytest.approx(0.8)


def test_pv_rejects_unknown_dimension():
    with pytest.raises(KeyError):
        pv(mana=1.0)


def test_every_dim_has_calibration_entry():
    for dim in PROP_DIMS_V2:
        entry = entry_for("dim", dim)
        assert entry is not None, f"Dimension '{dim}' ohne Kalibrierungs-Eintrag"
        assert entry.anchor.strip() and entry.source.strip()


def test_cal_rejects_empty_anchor_and_bad_kind():
    with pytest.raises(ValueError):
        cal("dim", "x", "", "quelle")
    with pytest.raises(ValueError):
        cal("spell", "x", "anker", "quelle")
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_props_schema.py -q`
Expected: FAIL / ERROR mit `ModuleNotFoundError: No module named 'artificial_society.environment.physics'`

- [ ] **Step 3: Implementierung schreiben**

`artificial_society/environment/physics/__init__.py`:

```python
"""Physik v2 — real kalibrierte Material- und Prozessphysik.

Spec: docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md.
Öffentliche API wird in Task 7 vervollständigt.
"""
```

`artificial_society/environment/physics/props.py`:

```python
"""Physik v2 — reale Eigenschaftsdimensionen.

Jede Dimension ist auf [0, 1] normiert über einen dokumentierten realen Anker
(z.B. hardness = Mohs / 10). Die Normierungsformeln stehen in calibration.py
(SSOT) und im generierten docs/physics/kalibrierung.md.

`sharpness` ist eine abgeleitete Größe: KEIN Startmaterial darf sie > 0 setzen —
Schärfe entsteht ausschließlich als Prozessergebnis (Bruch). Ein Gate-Test
erzwingt das.
"""

from __future__ import annotations

import numpy as np

PROP_DIMS_V2 = [
    "hardness",
    "density",
    "brittleness",
    "tensile_strength",
    "sharpness",
    "flammability",
    "ignition_temp",
    "melting_point",
    "thermal_conductivity",
    "nutrition",
    "toxicity",
    "moisture",
    "grain_fineness",
]
IDX2 = {name: i for i, name in enumerate(PROP_DIMS_V2)}
N_PROPS_V2 = len(PROP_DIMS_V2)  # 13


def pv(**kwargs: float) -> np.ndarray:
    """Eigenschaftsvektor aus Keyword-Args; fehlende Dimensionen = 0."""
    v = np.zeros(N_PROPS_V2, dtype=np.float32)
    for k, val in kwargs.items():
        v[IDX2[k]] = float(val)
    return v
```

`artificial_society/environment/physics/calibration.py`:

```python
"""Kalibrierungstabelle — das Realitäts-Gate der Physik v2 (Spec §2).

Jede Dimension, jedes Startmaterial und jeder Prozess MUSS hier einen Eintrag
mit realem Anker + Quelle haben. tests/environment/physics/test_reality_gate.py
erzwingt das als Merge-Bedingung: was keinen realen Anker hat, kommt nicht in
die Welt. docs/physics/kalibrierung.md wird aus dieser Datei generiert
(scripts/gen_kalibrierung.py) — nie von Hand editieren.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_KINDS = ("dim", "material", "process")


@dataclass(frozen=True)
class CalEntry:
    kind: str
    name: str
    anchor: str  # realer Anker inkl. Normierungsformel/Skala
    source: str  # Herkunft des Ankers


CALIBRATION: dict = {}


def cal(kind: str, name: str, anchor: str, source: str) -> None:
    """Kalibrierungs-Eintrag registrieren; leerer Anker/Quelle ist ein Fehler."""
    if kind not in VALID_KINDS:
        raise ValueError(f"unbekannter Kalibrierungs-Typ: {kind!r}")
    if not anchor.strip() or not source.strip():
        raise ValueError(f"Kalibrierung {kind}:{name} braucht Anker UND Quelle")
    CALIBRATION[(kind, name)] = CalEntry(kind=kind, name=name, anchor=anchor, source=source)


def entry_for(kind: str, name: str):
    return CALIBRATION.get((kind, name))


# ---------------------------------------------------------------------------
# Dimensionen (Normierungsanker)
# ---------------------------------------------------------------------------
cal(
    "dim",
    "hardness",
    "Ritzhärte, Mohs-Skala / 10 (Talk 1 → 0.1, Quarz/Feuerstein 7 → 0.7, Diamant 10 → 1.0)",
    "Mohs-Härteskala (Mineralogie-Standard)",
)
cal(
    "dim",
    "density",
    "Rohdichte ρ / 5000 kg/m³, geklemmt auf [0,1] (Wasser 1000 → 0.2, Granit 2700 → 0.54)",
    "CRC Handbook / Gesteinskunde-Standardwerte",
)
cal(
    "dim",
    "brittleness",
    "Sprödigkeit: 0 = zäh/duktil (Sehne, frisches Holz) … 1 = ideal spröde mit muscheligem "
    "Bruch (Glas/Obsidian ≈ 0.95, Feuerstein ≈ 0.9)",
    "Bruchmechanik spröder vs. duktiler Werkstoffe; Lithik (Feuersteinschlagen)",
)
cal(
    "dim",
    "tensile_strength",
    "Zugfestigkeit / 1000 MPa (Hanffaser ≈ 600 MPa → 0.6, Holz längs ≈ 100 → 0.1, Fels ≈ 10 → 0.01)",
    "Werkstoffkunde-Tabellenwerte",
)
cal(
    "dim",
    "sharpness",
    "Kantenschärfe: 0 = stumpf … 1 ≈ frisch geschlagene Obsidianklinge. NUR als "
    "Prozessergebnis (Bruch) erzeugbar — kein Startmaterial hat sharpness > 0",
    "Experimentelle Archäologie: Schärfe geschlagener Steinwerkzeuge",
)
cal(
    "dim",
    "flammability",
    "Entflammbarkeit 0..1 qualitativ: 0 = nicht brennbar (Stein), 0.8 = trockenes Holz, 1 = Zunder",
    "Brandverhalten von Naturstoffen (Forst-/Brandschutzliteratur)",
)
cal(
    "dim",
    "ignition_temp",
    "Zündtemperatur / 1000 °C (Holz ≈ 300 °C → 0.3); 1.0 = praktisch nicht entzündbar",
    "Zündtemperatur-Tabellen (Holz 280–340 °C)",
)
cal(
    "dim",
    "melting_point",
    "Schmelz-/Sinterpunkt / 2000 °C (Ton sintert ≈ 1000 → 0.5, Quarz ≈ 1670 → 0.84); "
    "1.0 = schmilzt praktisch nicht bzw. zersetzt sich vorher",
    "Keramik-/Petrologie-Standardwerte",
)
cal(
    "dim",
    "thermal_conductivity",
    "Wärmeleitfähigkeit / 10 W/(m·K), geklemmt (Granit ≈ 2.8 → 0.28, Holz ≈ 0.15 → 0.02)",
    "CRC Handbook, Wärmeleitfähigkeiten",
)
cal(
    "dim",
    "nutrition",
    "verwertbare Energie / 400 kcal pro 100 g (mageres Rohfleisch ≈ 140 → 0.35, Beeren ≈ 50 → 0.13)",
    "Nährwerttabellen (USDA)",
)
cal(
    "dim",
    "toxicity",
    "akute Schädlichkeit bei rohem Verzehr, 0..1 qualitativ (rohes Aas ≈ 0.15 wegen Keimbelastung)",
    "Lebensmittelhygiene: Keimbelastung roher Tierprodukte",
)
cal(
    "dim",
    "moisture",
    "Wasseranteil 0..1 (Frischfleisch ≈ 0.7, lufttrockenes Holz ≈ 0.12, Wasser = 1.0)",
    "Holzfeuchte-/Lebensmitteltabellen",
)
cal(
    "dim",
    "grain_fineness",
    "Gefügefeinheit: 0 = grobkristallin (Granit ≈ 0.15) … 1 = kryptokristallin (Feuerstein ≈ 0.95). "
    "Bestimmt muscheligen Bruch und erreichbare Kantenschärfe",
    "Petrologie: Kryptokristallinität von Silex — Grundlage des Feuersteinschlagens",
)
```

- [ ] **Step 4: Test laufen lassen — muss grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_props_schema.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add artificial_society/environment/physics/ tests/environment/physics/
git commit -m "env(physics-v2): Eigenschafts-Schema (13 reale Dimensionen) + Kalibrierungs-Mechanik

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Startmaterialien (real kalibriert)

**Files:**
- Create: `artificial_society/environment/physics/materials_v2.py`
- Test: `tests/environment/physics/test_materials_v2.py`

**Interfaces:**
- Consumes: `pv`, `IDX2`, `N_PROPS_V2` aus `props.py`; `cal` aus `calibration.py`
- Produces: `MATERIALS_V2: dict[str, np.ndarray]` mit genau diesen 9 Keys: `granite`, `flint`, `dry_wood`, `plant_fiber`, `clay_moist`, `water`, `berries`, `raw_meat`, `carcass`. Nach Import sind alle 9 als `("material", <name>)` kalibriert.

- [ ] **Step 1: Failing Test schreiben**

`tests/environment/physics/test_materials_v2.py`:

```python
"""Startmaterial-Tests: Normierung, keine Start-Schärfe, Kalibrierungspflicht."""

from __future__ import annotations

from artificial_society.environment.physics.calibration import entry_for
from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.props import IDX2, N_PROPS_V2

EXPECTED = {
    "granite",
    "flint",
    "dry_wood",
    "plant_fiber",
    "clay_moist",
    "water",
    "berries",
    "raw_meat",
    "carcass",
}


def test_expected_seed_material_set():
    assert set(MATERIALS_V2) == EXPECTED


def test_vectors_have_v2_shape_and_are_normalized():
    for name, vec in MATERIALS_V2.items():
        assert vec.shape == (N_PROPS_V2,), name
        assert (vec >= 0.0).all() and (vec <= 1.0).all(), name


def test_no_seed_material_starts_sharp():
    # Schärfe ist NUR Prozessergebnis — auch Feuerstein ist roh nicht scharf.
    for name, vec in MATERIALS_V2.items():
        assert vec[IDX2["sharpness"]] == 0.0, name


def test_flint_is_the_knappable_one():
    flint = MATERIALS_V2["flint"]
    granite = MATERIALS_V2["granite"]
    assert flint[IDX2["brittleness"]] > 0.8
    assert flint[IDX2["grain_fineness"]] > 0.9
    assert granite[IDX2["grain_fineness"]] < 0.3


def test_every_seed_material_is_calibrated():
    for name in MATERIALS_V2:
        entry = entry_for("material", name)
        assert entry is not None, f"Material '{name}' ohne Kalibrierungs-Eintrag"
        assert entry.anchor.strip() and entry.source.strip()
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_materials_v2.py -q`
Expected: FAIL mit `ModuleNotFoundError` (materials_v2 existiert nicht)

- [ ] **Step 3: Implementierung schreiben**

`artificial_society/environment/physics/materials_v2.py`:

```python
"""Startmaterialien der Physik v2 — reale Stoffe, real kalibrierte Werte.

Kein Startmaterial ist ein Rezept-Endprodukt, keines startet mit sharpness > 0
(Schärfe entsteht nur durch Prozesse, z.B. Feuerstein schlagen). Werte sind
[0,1]-normiert nach den Ankern in calibration.py.
"""

from __future__ import annotations

from .calibration import cal
from .props import pv

MATERIALS_V2: dict = {
    "granite": pv(
        hardness=0.65,
        density=0.54,
        brittleness=0.45,
        tensile_strength=0.01,
        ignition_temp=1.0,
        melting_point=0.62,
        thermal_conductivity=0.28,
        moisture=0.02,
        grain_fineness=0.15,
    ),
    "flint": pv(
        hardness=0.70,
        density=0.52,
        brittleness=0.90,
        tensile_strength=0.01,
        ignition_temp=1.0,
        melting_point=0.84,
        thermal_conductivity=0.14,
        moisture=0.01,
        grain_fineness=0.95,
    ),
    "dry_wood": pv(
        hardness=0.25,
        density=0.13,
        brittleness=0.20,
        tensile_strength=0.10,
        flammability=0.80,
        ignition_temp=0.30,
        melting_point=1.0,
        thermal_conductivity=0.02,
        moisture=0.12,
        grain_fineness=0.30,
    ),
    "plant_fiber": pv(
        hardness=0.05,
        density=0.06,
        brittleness=0.05,
        tensile_strength=0.60,
        flammability=0.85,
        ignition_temp=0.25,
        melting_point=1.0,
        thermal_conductivity=0.01,
        nutrition=0.01,
        moisture=0.15,
        grain_fineness=0.50,
    ),
    "clay_moist": pv(
        hardness=0.10,
        density=0.40,
        brittleness=0.15,
        ignition_temp=1.0,
        melting_point=0.50,
        thermal_conductivity=0.10,
        toxicity=0.02,
        moisture=0.35,
        grain_fineness=0.80,
    ),
    "water": pv(
        density=0.20,
        ignition_temp=1.0,
        thermal_conductivity=0.06,
        moisture=1.0,
    ),
    "berries": pv(
        hardness=0.02,
        density=0.21,
        brittleness=0.10,
        flammability=0.05,
        ignition_temp=0.40,
        melting_point=1.0,
        thermal_conductivity=0.05,
        nutrition=0.13,
        toxicity=0.05,
        moisture=0.85,
        grain_fineness=0.60,
    ),
    "raw_meat": pv(
        hardness=0.05,
        density=0.21,
        brittleness=0.15,
        tensile_strength=0.02,
        flammability=0.05,
        ignition_temp=0.60,
        melting_point=1.0,
        thermal_conductivity=0.05,
        nutrition=0.35,
        toxicity=0.15,
        moisture=0.70,
        grain_fineness=0.40,
    ),
    "carcass": pv(
        hardness=0.10,
        density=0.20,
        brittleness=0.10,
        tensile_strength=0.10,
        flammability=0.05,
        ignition_temp=0.60,
        melting_point=1.0,
        thermal_conductivity=0.05,
        nutrition=0.30,
        toxicity=0.10,
        moisture=0.65,
        grain_fineness=0.30,
    ),
}

cal(
    "material",
    "granite",
    "Granit: Mohs ≈ 6.5, ρ ≈ 2700 kg/m³, grobkristallin, mäßig spröde — splittert unter "
    "starkem Schlag in stumpfe Bruchstücke",
    "Gesteinskunde-Standardwerte (Granit)",
)
cal(
    "material",
    "flint",
    "Feuerstein/Silex: Mohs ≈ 7, ρ ≈ 2600, kryptokristallin → muscheliger Bruch; klassisches "
    "Ausgangsmaterial für Klingen; im Rohzustand NICHT scharf",
    "Petrologie Silex; experimentelle Archäologie Feuersteinschlagen",
)
cal(
    "material",
    "dry_wood",
    "Lufttrockenes Holz: ρ ≈ 650, Zündtemp ≈ 300 °C, Zugfestigkeit längs ≈ 100 MPa, zäh "
    "(splittert nicht muschelig)",
    "Holztechnik-Tabellenwerte",
)
cal(
    "material",
    "plant_fiber",
    "Bastfaser (Hanf/Lein): Zugfestigkeit ≈ 600 MPa, sehr leicht, gut brennbar",
    "Werkstoffkunde Naturfasern",
)
cal(
    "material",
    "clay_moist",
    "Feuchter Ton: weich/plastisch, sintert ab ≈ 1000 °C, sehr feines Gefüge",
    "Keramik-Grundlagen",
)
cal(
    "material",
    "water",
    "Wasser: ρ = 1000 kg/m³ — Referenzstoff der Dichte- und Feuchteskala",
    "CRC Handbook",
)
cal(
    "material",
    "berries",
    "Beeren: ≈ 50 kcal/100 g, wasserreich, weich; leichte Rest-Toxizität (Wildsammlung)",
    "USDA Nährwerttabelle (Beeren)",
)
cal(
    "material",
    "raw_meat",
    "Rohes Muskelfleisch: ≈ 140 kcal/100 g, ≈ 70 % Wasser, zäh (niedrige Sprödigkeit), roh "
    "leicht keimbelastet",
    "USDA Nährwerttabellen; Lebensmittelhygiene",
)
cal(
    "material",
    "carcass",
    "Tierkadaver: Verbund aus Haut/Sehnen/Fleisch/Knochen — sehr zäh; Nährwert praktisch nur "
    "durch Zerteilen (Schneiden) erschließbar",
    "Zoologie/Jagdpraxis: Zerwirken",
)
```

- [ ] **Step 4: Test laufen lassen — muss grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_materials_v2.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add artificial_society/environment/physics/materials_v2.py tests/environment/physics/test_materials_v2.py
git commit -m "env(physics-v2): 9 reale Startmaterialien mit Kalibrierungs-Ankern

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Physische Objekte (Eigenschaften + Masse)

**Files:**
- Create: `artificial_society/environment/physics/objects.py`
- Test: `tests/environment/physics/test_objects.py`

**Interfaces:**
- Consumes: `MATERIALS_V2`, `N_PROPS_V2`
- Produces: `PhysObject(props: np.ndarray, mass: float, kind: str = "unknown")` (dataclass; validiert shape `(13,)` float32 und `mass > 0` in `__post_init__`); `make_object(kind: str, mass: float) -> PhysObject` (kopiert den Materialvektor).

- [ ] **Step 1: Failing Test schreiben**

`tests/environment/physics/test_objects.py`:

```python
"""PhysObject: intensiver Eigenschaftsvektor + extensive Masse (kg)."""

from __future__ import annotations

import numpy as np
import pytest

from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.objects import PhysObject, make_object
from artificial_society.environment.physics.props import N_PROPS_V2


def test_make_object_copies_material_vector():
    obj = make_object("flint", 0.8)
    assert obj.kind == "flint"
    assert obj.mass == pytest.approx(0.8)
    obj.props[0] = 0.123
    assert MATERIALS_V2["flint"][0] != np.float32(0.123)  # Original unverändert


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        PhysObject(props=np.zeros(3, dtype=np.float32), mass=1.0)


def test_rejects_non_positive_mass():
    with pytest.raises(ValueError):
        PhysObject(props=np.zeros(N_PROPS_V2, dtype=np.float32), mass=0.0)
    with pytest.raises(ValueError):
        make_object("granite", -1.0)
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_objects.py -q`
Expected: FAIL mit `ModuleNotFoundError` (objects existiert nicht)

- [ ] **Step 3: Implementierung schreiben**

`artificial_society/environment/physics/objects.py`:

```python
"""Physische Objekte der Physik v2.

Ein Objekt = intensiver Eigenschaftsvektor (props, [0,1]-normiert laut
Kalibrierung) + extensive Masse in kg. Masse ist die Erhaltungsgröße aller
Prozesse. `kind` ist reines Herkunfts-/Diagnoselabel und darf NIE Physik-Input
sein — Wirkung folgt ausschließlich aus props (Spec §3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials_v2 import MATERIALS_V2
from .props import N_PROPS_V2


@dataclass
class PhysObject:
    props: np.ndarray
    mass: float
    kind: str = "unknown"

    def __post_init__(self) -> None:
        self.props = np.asarray(self.props, dtype=np.float32)
        if self.props.shape != (N_PROPS_V2,):
            raise ValueError(f"props braucht shape ({N_PROPS_V2},), hat {self.props.shape}")
        if self.mass <= 0:
            raise ValueError("mass muss > 0 sein")


def make_object(kind: str, mass: float) -> PhysObject:
    """Objekt eines Startmaterials erzeugen (kopiert den Materialvektor)."""
    return PhysObject(props=MATERIALS_V2[kind].copy(), mass=mass, kind=kind)
```

- [ ] **Step 4: Test laufen lassen — muss grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_objects.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add artificial_society/environment/physics/objects.py tests/environment/physics/test_objects.py
git commit -m "env(physics-v2): PhysObject — Eigenschaften intensiv, Masse extensiv

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Prozess „Schlagen" (Bruchphysik)

**Files:**
- Create: `artificial_society/environment/physics/processes.py`
- Test: `tests/environment/physics/test_strike.py`

**Interfaces:**
- Consumes: `PhysObject`, `make_object`, `IDX2`, `cal`
- Produces: `strike(target: PhysObject, striker: PhysObject, impact_energy_j: float, rng: random.Random) -> StrikeResult`; `StrikeResult(fractured: bool, fragments: list[PhysObject])`; `fracture_threshold_j(target: PhysObject) -> float`; Konstanten `FRACTURE_BASE_J_PER_KG = 60.0`, `MIN_STRIKER_HARDNESS = 0.5`, `MIN_HARDNESS_MARGIN = -0.15`; `PROCESSES: dict[str, callable]` (enthält `"strike"`). Prozess ist als `("process", "strike")` kalibriert.

**Kalibrierte Erwartungswerte (zur Orientierung des Implementierers):**
Bruchschwelle = `60 · (1.05 − brittleness) · max(mass, 0.1)` J. Feuerstein-Kern 0.8 kg → 7.2 J (Handschlag ~20 J bricht ihn); Granit 0.8 kg → 28.8 J (erst kräftiger Schlag ~45 J); trockenes Holz 0.8 kg → 40.8 J (Handschlag bricht es nicht). Fragment-Schärfe = `brittleness · grain_fineness · (0.6 + 0.4·u)`: Feuerstein ∈ [0.513, 0.855] (scharf), Granit ∈ [0.04, 0.07] (stumpf) — die Testschwellen 0.5 / 0.15 sind für JEDEN rng-Zug sicher.

- [ ] **Step 1: Failing Test schreiben**

`tests/environment/physics/test_strike.py`:

```python
"""Bruchphysik: nur spröde+feinkörnige Ziele liefern scharfe Fragmente; Masse erhalten."""

from __future__ import annotations

import math
import random

from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.processes import strike
from artificial_society.environment.physics.props import IDX2

HAND_STRIKE_J = 20.0  # kräftiger Handschlag mit Schlagstein (Anker: 10–50 J)
STRONG_STRIKE_J = 45.0


def test_flint_strike_yields_sharp_fragments():
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(42))
    assert result.fractured
    assert len(result.fragments) >= 2
    assert max(f.props[IDX2["sharpness"]] for f in result.fragments) > 0.5


def test_granite_fragments_are_dull():
    result = strike(make_object("granite", 0.8), make_object("granite", 1.2), STRONG_STRIKE_J, random.Random(42))
    assert result.fractured
    assert max(f.props[IDX2["sharpness"]] for f in result.fragments) < 0.15


def test_tough_wood_does_not_shatter_under_hand_strike():
    result = strike(make_object("dry_wood", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(42))
    assert not result.fractured


def test_insufficient_energy_no_fracture():
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), 2.0, random.Random(42))
    assert not result.fractured


def test_soft_striker_cannot_knap():
    result = strike(make_object("flint", 0.8), make_object("dry_wood", 1.0), STRONG_STRIKE_J, random.Random(42))
    assert not result.fractured


def test_mass_is_conserved():
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(7))
    assert result.fractured
    assert math.isclose(sum(f.mass for f in result.fragments), 0.8, rel_tol=1e-6)


def test_deterministic_given_same_rng_seed():
    a = strike(make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(3))
    b = strike(make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(3))
    assert [f.mass for f in a.fragments] == [f.mass for f in b.fragments]


def test_striking_creates_no_nutrition():
    # Kein Zauber: ein Schlag erzeugt keinen Nährwert.
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(42))
    for f in result.fragments:
        assert f.props[IDX2["nutrition"]] == 0.0
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_strike.py -q`
Expected: FAIL mit `ModuleNotFoundError` (processes existiert nicht)

- [ ] **Step 3: Implementierung schreiben**

`artificial_society/environment/physics/processes.py`:

```python
"""Prozesse der Physik v2 — Wirkung folgt IMMER aus Eigenschaften, nie aus Labels.

Erhaltung: die Masse aller Outputs eines Prozesses ist exakt die Masse der
Inputs (Tests sichern das). Zufall läuft ausschließlich über einen explizit
übergebenen random.Random (Determinismus-Kontrakt des Repos).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .calibration import cal
from .objects import PhysObject
from .props import IDX2

# Schlag-Physik (realer Anker: siehe cal()-Eintrag unten)
FRACTURE_BASE_J_PER_KG = 60.0
MIN_STRIKER_HARDNESS = 0.5
MIN_HARDNESS_MARGIN = -0.15  # Schlagstein darf höchstens 0.15 weicher sein als das Ziel


@dataclass
class StrikeResult:
    fractured: bool
    fragments: list = field(default_factory=list)


def fracture_threshold_j(target: PhysObject) -> float:
    """Nötige Aufprallenergie: zähere Ziele und mehr Masse brauchen mehr Energie."""
    brittleness = float(target.props[IDX2["brittleness"]])
    return FRACTURE_BASE_J_PER_KG * (1.05 - brittleness) * max(target.mass, 0.1)


def strike(
    target: PhysObject,
    striker: PhysObject,
    impact_energy_j: float,
    rng: random.Random,
) -> StrikeResult:
    """Hart-Hammer-Perkussion: Ziel zerbricht, wenn Schläger hart genug und
    Energie über der Bruchschwelle liegt. Fragment-Schärfe ergibt sich aus
    Sprödigkeit × Gefügefeinheit (muscheliger Bruch), nie aus einem Label."""
    striker_hardness = float(striker.props[IDX2["hardness"]])
    target_hardness = float(target.props[IDX2["hardness"]])
    if striker_hardness < MIN_STRIKER_HARDNESS:
        return StrikeResult(fractured=False)
    if striker_hardness - target_hardness < MIN_HARDNESS_MARGIN:
        return StrikeResult(fractured=False)
    if impact_energy_j < fracture_threshold_j(target):
        return StrikeResult(fractured=False)

    n_fragments = rng.randint(2, 3)
    weights = [0.5 + rng.random() for _ in range(n_fragments)]
    total = sum(weights)
    brittleness = float(target.props[IDX2["brittleness"]])
    grain = float(target.props[IDX2["grain_fineness"]])
    fragments = []
    for w in weights:
        props = target.props.copy()
        props[IDX2["sharpness"]] = brittleness * grain * (0.6 + 0.4 * rng.random())
        fragments.append(
            PhysObject(props=props, mass=(w / total) * target.mass, kind=f"{target.kind}_fragment")
        )
    return StrikeResult(fractured=True, fragments=fragments)


PROCESSES: dict = {"strike": strike}

cal(
    "process",
    "strike",
    "Hart-Hammer-Perkussion: nur spröde, feinkörnige Gesteine (muscheliger Bruch) liefern "
    "scharfe Abschläge; nötige Schlagenergie im Bereich eines kräftigen Handschlags (10–50 J); "
    "zähe Stoffe (Holz, Fleisch) zersplittern so nicht",
    "Experimentelle Archäologie: Feuersteinschlagen/Lithik; Bruchmechanik spröder Stoffe",
)
```

- [ ] **Step 4: Test laufen lassen — muss grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_strike.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add artificial_society/environment/physics/processes.py tests/environment/physics/test_strike.py
git commit -m "env(physics-v2): Prozess Schlagen — Bruchphysik mit Massenerhaltung

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Prozess „Schneiden" (Werkzeugnutzen)

**Files:**
- Modify: `artificial_society/environment/physics/processes.py` (Funktionen anhängen, `PROCESSES` erweitern)
- Test: `tests/environment/physics/test_cut.py`

**Interfaces:**
- Consumes: `strike` (in Tests, um reale Klingen-Fragmente zu erzeugen), `PhysObject`, `IDX2`, `cal`
- Produces: `cut(target: PhysObject, tool: PhysObject | None) -> CutResult`; `CutResult(extracted: PhysObject | None, remainder: PhysObject)`; `effective_sharpness(tool: PhysObject | None) -> float`; Konstanten `BARE_HAND_SHARPNESS = 0.05`, `MAX_CUT_FRACTION = 0.9`. `PROCESSES` enthält danach `"strike"` UND `"cut"`; `("process", "cut")` ist kalibriert.

**Kalibrierte Erwartungswerte:** Ertragsanteil = `eff_sharpness · max(0, 1.15 − Zähigkeit)`, Zähigkeit = `1 − brittleness`. Kadaver (Zähigkeit 0.9): bloße Hand 1.25 % pro Aktion; Feuerstein-Klinge (Schärfe 0.51–0.86, Härte 0.7) 10.9–18.2 % → Faktor ≈ 9–15× (Test-Schwelle 6× ist für jeden rng-Zug sicher); stumpfes Granit-Fragment ≈ 0.8–1.4 % (kaum besser als die Hand — realistisch).

- [ ] **Step 1: Failing Test schreiben**

`tests/environment/physics/test_cut.py`:

```python
"""Schneidephysik: scharfe Klinge schlägt bloße Hand um ein Vielfaches; Masse erhalten."""

from __future__ import annotations

import math
import random

from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.processes import cut, strike
from artificial_society.environment.physics.props import IDX2


def _sharpest_flint_fragment():
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), 20.0, random.Random(42))
    return max(result.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))


def test_blade_beats_bare_hand_by_wide_margin():
    hand = cut(make_object("carcass", 20.0), None)
    blade = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    assert hand.extracted is not None and blade.extracted is not None
    assert blade.extracted.mass > 6 * hand.extracted.mass


def test_cut_conserves_mass():
    result = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    assert math.isclose(result.extracted.mass + result.remainder.mass, 20.0, rel_tol=1e-6)


def test_cut_does_not_change_properties():
    # Schneiden macht nichts nahrhafter oder giftiger — nur kleiner.
    result = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    assert (result.extracted.props == MATERIALS_V2["carcass"]).all()
    assert (result.remainder.props == MATERIALS_V2["carcass"]).all()


def test_dull_granite_fragment_cuts_barely_better_than_hand():
    dull_result = strike(make_object("granite", 0.8), make_object("granite", 1.2), 45.0, random.Random(42))
    dull = max(dull_result.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))
    blade_cut = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    dull_cut = cut(make_object("carcass", 20.0), dull)
    assert blade_cut.extracted.mass > 4 * dull_cut.extracted.mass
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_cut.py -q`
Expected: FAIL mit `ImportError: cannot import name 'cut'`

- [ ] **Step 3: Implementierung anhängen**

In `artificial_society/environment/physics/processes.py` — Konstanten zu den bestehenden dazu, Code nach `strike()` anhängen, `PROCESSES`-Zeile ERSETZEN:

```python
# Schneide-Physik (realer Anker: siehe cal()-Eintrag unten)
BARE_HAND_SHARPNESS = 0.05  # Hände/Zähne/Reißen
MAX_CUT_FRACTION = 0.9


@dataclass
class CutResult:
    extracted: PhysObject | None
    remainder: PhysObject


def effective_sharpness(tool: PhysObject | None) -> float:
    """Ohne Werkzeug: Hände/Zähne. Mit Werkzeug: Schärfe, getragen von Härte —
    eine scharfe, aber weiche Kante verformt sich beim Schnitt."""
    if tool is None:
        return BARE_HAND_SHARPNESS
    return float(tool.props[IDX2["sharpness"]]) * (0.5 + 0.5 * float(tool.props[IDX2["hardness"]]))


def cut(target: PhysObject, tool: PhysObject | None) -> CutResult:
    """Ein Schneidevorgang trennt einen Massenanteil ab. Ertrag steigt mit
    effektiver Schärfe des Werkzeugs, sinkt mit Zähigkeit des Ziels.
    Eigenschaften bleiben unverändert (nur kleiner) — Masse exakt erhalten."""
    toughness = 1.0 - float(target.props[IDX2["brittleness"]])
    yield_fraction = min(effective_sharpness(tool) * max(0.0, 1.15 - toughness), MAX_CUT_FRACTION)
    extracted_mass = yield_fraction * target.mass
    if extracted_mass <= 1e-9:
        return CutResult(extracted=None, remainder=target)
    extracted = PhysObject(
        props=target.props.copy(), mass=extracted_mass, kind=f"{target.kind}_piece"
    )
    remainder = PhysObject(
        props=target.props.copy(), mass=target.mass - extracted_mass, kind=target.kind
    )
    return CutResult(extracted=extracted, remainder=remainder)


PROCESSES: dict = {"strike": strike, "cut": cut}

cal(
    "process",
    "cut",
    "Schneiden: Ertrag steigt mit Kantenschärfe × Härte des Werkzeugs und sinkt mit der "
    "Zähigkeit des Ziels; einen Kadaver mit bloßer Hand zu zerwirken ist nahezu unmöglich, "
    "mit Steinklinge effizient",
    "Experimentelle Archäologie: Zerwirken mit Steinklingen",
)
```

Hinweis: die alte Zeile `PROCESSES: dict = {"strike": strike}` aus Task 4 wird durch die neue ersetzt (nur EINE `PROCESSES`-Definition am Dateiende, nach beiden Funktionen).

- [ ] **Step 4: Tests laufen lassen — muss grün sein (inkl. Task-4-Tests)**

Run: `../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: alle Tests passed, 0 failed

- [ ] **Step 5: Commit**

```bash
git add artificial_society/environment/physics/processes.py tests/environment/physics/test_cut.py
git commit -m "env(physics-v2): Prozess Schneiden — Klinge schlägt Hand mechanistisch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Discovery-Registry v2 (neuartige Vektoren → stabile IDs)

**Files:**
- Create: `artificial_society/environment/physics/discovery.py`
- Test: `tests/environment/physics/test_discovery_v2.py`

**Interfaces:**
- Consumes: `strike`/`make_object` (nur im Test)
- Produces: `DiscoveryV2(similarity_threshold: float = 0.08)` mit `register(vector, discoverer_id=-1, tick=0) -> str` (IDs `pmat_0000`, `pmat_0001`, …; ähnliche Vektoren < Schwelle → gleiche ID), `get_vector(mat_id) -> np.ndarray | None`, `known_ids() -> list[str]`, `reset()`, `state_dict()`, `load_state_dict(data)`; Modul-Instanz `DISCOVERY_V2`.

**Warum eigene Klasse statt Reuse:** die v1-`DiscoveryRegistry` (`environment/materials.py`) printet beim Registrieren v1-Dimensionsnamen über v1-Indizes (`edibility`, `heat_emission`, `scent`) — auf 13-dim-v2-Vektoren wären das falsch beschriftete Werte; außerdem ist materials.py eingefrorenes Hot File. Die Minimal-Kopie ist bewusst.

- [ ] **Step 1: Failing Test schreiben**

`tests/environment/physics/test_discovery_v2.py`:

```python
"""DiscoveryV2: stabile IDs für neuartige v2-Eigenschaftsvektoren."""

from __future__ import annotations

import random

import numpy as np

from artificial_society.environment.physics.discovery import DiscoveryV2
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.processes import strike


def test_same_vector_same_id_new_vector_new_id():
    reg = DiscoveryV2()
    v1 = np.zeros(13, dtype=np.float32)
    v2 = v1.copy()
    v2[0] = 0.5
    id_a = reg.register(v1)
    assert reg.register(v1.copy()) == id_a  # identisch → gleiche ID
    assert reg.register(v2) != id_a  # deutlich anders → neue ID
    assert reg.known_ids() == [id_a, "pmat_0001"]


def test_fragments_from_strike_get_registered():
    reg = DiscoveryV2()
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), 20.0, random.Random(42))
    ids = [reg.register(f.props, discoverer_id=7, tick=100) for f in result.fragments]
    assert all(i.startswith("pmat_") for i in ids)
    vec = reg.get_vector(ids[0])
    assert vec is not None and vec.shape == (13,)


def test_state_roundtrip_and_reset():
    reg = DiscoveryV2()
    reg.register(np.ones(13, dtype=np.float32))
    snapshot = reg.state_dict()
    reg2 = DiscoveryV2()
    reg2.load_state_dict(snapshot)
    assert reg2.known_ids() == reg.known_ids()
    reg2.reset()
    assert reg2.known_ids() == []
    assert reg2.get_vector("pmat_0000") is None
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_discovery_v2.py -q`
Expected: FAIL mit `ModuleNotFoundError` (discovery existiert nicht)

- [ ] **Step 3: Implementierung schreiben**

`artificial_society/environment/physics/discovery.py`:

```python
"""Discovery-Registry der Physik v2: neuartige Eigenschaftsvektoren → stabile IDs.

Bewusst eigene Minimal-Implementierung statt Reuse der v1-Registry aus
environment/materials.py: die loggt v1-Dimensionsnamen über v1-Indizes (auf
v2-Vektoren falsch) und ist Teil des eingefrorenen Hot-File-Contracts.
Kein Print — Logging entscheidet später die Sim-Integration.
"""

from __future__ import annotations

import numpy as np


class DiscoveryV2:
    def __init__(self, similarity_threshold: float = 0.08):
        self.entries: list = []
        self.threshold = similarity_threshold

    def reset(self) -> None:
        self.entries.clear()

    def state_dict(self) -> dict:
        return {"entries": self.entries}

    def load_state_dict(self, data: dict) -> None:
        self.entries = list(data.get("entries", []))

    def register(self, vector: np.ndarray, discoverer_id: int = -1, tick: int = 0) -> str:
        vec = np.asarray(vector, dtype=np.float32)
        for entry in self.entries:
            if float(np.linalg.norm(vec - entry["vector"])) < self.threshold:
                return entry["id"]
        new_id = f"pmat_{len(self.entries):04d}"
        self.entries.append(
            {"id": new_id, "vector": vec.copy(), "discovered_by": discoverer_id, "tick": tick}
        )
        return new_id

    def get_vector(self, mat_id: str):
        for entry in self.entries:
            if entry["id"] == mat_id:
                return entry["vector"].copy()
        return None

    def known_ids(self) -> list:
        return [e["id"] for e in self.entries]


DISCOVERY_V2 = DiscoveryV2()
```

- [ ] **Step 4: Test laufen lassen — muss grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_discovery_v2.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add artificial_society/environment/physics/discovery.py tests/environment/physics/test_discovery_v2.py
git commit -m "env(physics-v2): DiscoveryV2 — stabile IDs für neuartige Vektoren

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Realitäts-Gate + Kalibrierungs-Doku + öffentliche API

**Files:**
- Modify: `artificial_society/environment/physics/calibration.py` (Render-Funktion anhängen)
- Modify: `artificial_society/environment/physics/__init__.py` (öffentliche API)
- Create: `scripts/gen_kalibrierung.py`
- Create: `docs/physics/kalibrierung.md` (generiert + committet)
- Test: `tests/environment/physics/test_reality_gate.py`

**Interfaces:**
- Consumes: alles aus Tasks 1–6
- Produces: `calibration.render_markdown() -> str`; Paket-API `from artificial_society.environment.physics import MATERIALS_V2, PhysObject, make_object, strike, cut, ...`; dauerhafter Gate-Test als CI-Merge-Bedingung.

- [ ] **Step 1: Failing Test schreiben**

`tests/environment/physics/test_reality_gate.py`:

```python
"""Realitäts-Gate (Spec §2): kein Physik-Baustein ohne realen Kalibrierungs-Anker.

Dieser Test ist eine dauerhafte MERGE-BEDINGUNG: Wer eine Dimension, ein
Material oder einen Prozess hinzufügt, muss einen cal()-Eintrag mit realem
Anker + Quelle mitliefern — sonst rot.
"""

from __future__ import annotations

from pathlib import Path

from artificial_society.environment.physics import MATERIALS_V2, PROP_DIMS_V2
from artificial_society.environment.physics.calibration import (
    CALIBRATION,
    entry_for,
    render_markdown,
)
from artificial_society.environment.physics.processes import PROCESSES

REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_calibrated(kind: str, name: str) -> None:
    entry = entry_for(kind, name)
    assert entry is not None, f"{kind} '{name}' ohne Kalibrierungs-Eintrag (Realitäts-Gate)"
    assert entry.anchor.strip(), f"{kind} '{name}': leerer Anker"
    assert entry.source.strip(), f"{kind} '{name}': leere Quelle"


def test_every_dim_is_calibrated():
    for dim in PROP_DIMS_V2:
        _assert_calibrated("dim", dim)


def test_every_material_is_calibrated():
    for name in MATERIALS_V2:
        _assert_calibrated("material", name)


def test_every_process_is_calibrated():
    for name in PROCESSES:
        _assert_calibrated("process", name)


def test_no_orphan_calibration_entries():
    # Ein Eintrag ohne zugehörigen Baustein = Umbenennung ohne Tabellenpflege.
    for kind, name in CALIBRATION:
        if kind == "dim":
            assert name in PROP_DIMS_V2, f"verwaister dim-Eintrag: {name}"
        elif kind == "material":
            assert name in MATERIALS_V2, f"verwaister material-Eintrag: {name}"
        elif kind == "process":
            assert name in PROCESSES, f"verwaister process-Eintrag: {name}"


def test_kalibrierung_doc_is_in_sync():
    doc = REPO_ROOT / "docs" / "physics" / "kalibrierung.md"
    assert doc.exists(), "docs/physics/kalibrierung.md fehlt — scripts/gen_kalibrierung.py laufen lassen"
    assert doc.read_text(encoding="utf-8") == render_markdown(), (
        "kalibrierung.md ist veraltet — scripts/gen_kalibrierung.py neu laufen lassen"
    )
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_reality_gate.py -q`
Expected: FAIL — `ImportError` (`MATERIALS_V2` ist noch nicht in `__init__` exportiert / `render_markdown` existiert nicht)

- [ ] **Step 3: Implementierung**

An `artificial_society/environment/physics/calibration.py` anhängen:

```python
_KIND_TITLES = {"dim": "Eigenschafts-Dimensionen", "material": "Startmaterialien", "process": "Prozesse"}

_DOC_HEADER = (
    "# Kalibrierungstabelle Physik v2\n\n"
    "> GENERIERT aus `artificial_society/environment/physics/calibration.py` via\n"
    "> `scripts/gen_kalibrierung.py` — nicht von Hand editieren.\n"
    "> Realitäts-Gate: Spec `docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md` §2.\n"
)


def render_markdown() -> str:
    """Kalibrierungstabelle als Markdown (SSOT = diese Datei)."""
    # Import hier, damit alle cal()-Registrierungen der Schwester-Module feuern
    # (kein Import-Zyklus: die Schwestern importieren nur cal aus diesem Modul).
    from . import materials_v2, processes  # noqa: F401

    lines = [_DOC_HEADER]
    for kind in VALID_KINDS:
        lines.append(f"\n## {_KIND_TITLES[kind]}\n")
        lines.append("| Name | Realer Anker | Quelle |")
        lines.append("|---|---|---|")
        entries = sorted((e for e in CALIBRATION.values() if e.kind == kind), key=lambda e: e.name)
        for e in entries:
            lines.append(f"| `{e.name}` | {e.anchor} | {e.source} |")
    return "\n".join(lines) + "\n"
```

`artificial_society/environment/physics/__init__.py` ERSETZEN durch:

```python
"""Physik v2 — real kalibrierte Material- und Prozessphysik.

Spec: docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md.
Realitäts-Gate: jede Dimension / jedes Material / jeder Prozess braucht einen
Kalibrierungs-Eintrag (calibration.cal) — tests/environment/physics/
test_reality_gate.py erzwingt das als Merge-Bedingung.
"""

from .calibration import CALIBRATION, CalEntry, cal, entry_for, render_markdown
from .discovery import DISCOVERY_V2, DiscoveryV2
from .materials_v2 import MATERIALS_V2
from .objects import PhysObject, make_object
from .processes import (
    PROCESSES,
    CutResult,
    StrikeResult,
    cut,
    effective_sharpness,
    fracture_threshold_j,
    strike,
)
from .props import IDX2, N_PROPS_V2, PROP_DIMS_V2, pv

__all__ = [
    "CALIBRATION",
    "CalEntry",
    "cal",
    "entry_for",
    "render_markdown",
    "DISCOVERY_V2",
    "DiscoveryV2",
    "MATERIALS_V2",
    "PhysObject",
    "make_object",
    "PROCESSES",
    "CutResult",
    "StrikeResult",
    "cut",
    "effective_sharpness",
    "fracture_threshold_j",
    "strike",
    "IDX2",
    "N_PROPS_V2",
    "PROP_DIMS_V2",
    "pv",
]
```

`scripts/gen_kalibrierung.py`:

```python
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
```

Dann die Doku generieren:

Run: `../venv/bin/python scripts/gen_kalibrierung.py`
Expected: `wrote /Users/moritzbecker/projekt/artificial-society/docs/physics/kalibrierung.md`

- [ ] **Step 4: Gesamte Physik-Suite laufen lassen — muss grün sein**

Run: `../venv/bin/python -m pytest tests/environment/physics/ -q`
Expected: alle Tests passed (Schema 5, Materialien 5, Objekte 3, Schlagen 8, Schneiden 4, Discovery 3, Gate 5)

- [ ] **Step 5: Commit**

```bash
git add artificial_society/environment/physics/ scripts/gen_kalibrierung.py docs/physics/kalibrierung.md tests/environment/physics/test_reality_gate.py
git commit -m "env(physics-v2): Realitäts-Gate als Merge-Bedingung + generierte Kalibrierungstabelle

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Roadmap-Abgleich (Tools zuerst, laut freigegebener Spec) + Gesamt-Gate

**Files:**
- Modify: `docs/roadmap.md` (§4b Reihenfolge + Backlog-Zeile O)

**Interfaces:**
- Consumes: Spec `docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md`
- Produces: Roadmap konsistent mit der freigegebenen Spec (Werkzeuge vor Sprache); Backlog verweist auf diesen Plan.

**Kontext:** Die Konsolidierung hat §4b mit Reihenfolge „1. Language, 2. Tools, 3. Building" geschrieben. Die vom User freigegebene Spec legt fest: **Werkzeuge zuerst** (Sprache emergiert erst sinnvoll, wenn es Kommunikationswürdiges gibt). Nur Reihenfolge/Verweise ändern — Prinzipien-Absatz („never given a capability") bleibt unverändert.

- [ ] **Step 1: §4b-Liste umsortieren**

In `docs/roadmap.md`, §4b: die nummerierte Liste so umstellen, dass gilt (Wortlaut der bestehenden Einträge übernehmen, nur Nummern/Reihenfolge + der eine neue Verweis):

1. **Tools / crafting.** (bisheriger Punkt 2, unverändert) — am Ende ergänzen: `Spec + laufender Plan: docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md, docs/superpowers/plans/2026-07-02-physik-v2-kern.md.`
2. **Building / environment shaping.** (bisheriger Punkt 3, unverändert)
3. **Language / communication.** (bisheriger Punkt 1, unverändert) — am Ende ergänzen: `Bewusst nach Tools: Sprache braucht erst etwas Kommunikationswürdiges (Spec 2026-07-02, §9).`
4. **Learning machinery.** (unverändert Punkt 4)

- [ ] **Step 2: Backlog-Zeile O ersetzen**

Die Zeile
`| O | **Capability slice 1: language/communication** (§4b — spec first) | systems + agents | regen | — |`
ersetzen durch:
`| O | **Capability slice 1: tools/crafting — Physik v2 Kern** (Spec + Plan 2026-07-02) | environment | neutral | — |`

- [ ] **Step 3: Gesamtes Gate laufen lassen**

Run: `bash scripts/check.sh`
Expected: ruff clean, kompletter pytest grün inkl. Golden-Trajectory + Headless-Digest (dieser Plan integriert nichts in die Sim — Golden MUSS unverändert grün sein; ist sie rot, hat ein Task versehentlich Bestehendes berührt → stoppen und prüfen).

- [ ] **Step 4: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): Capability-Reihenfolge an freigegebene Spec angeglichen — Tools zuerst

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Danach (NICHT Teil dieses Plans)

Folgepläne laut Spec §6, jeweils mit eigenem Plan-Dokument, erst nach Merge dieses Plans:
1. **Embodiment v1** (Tragkraft/Greifen/Kraft/Ermüdung — Hot File `agents/agent.py`, seriell via core-lead).
2. **Lern-Kopplung** (Policy wählt Kombination; Causal-Model-Prototyp aus `archive/` reaktivieren; Shaping-Boni raus — Hot Files `agents/brain.py`, `agents/agent.py`).
3. **Kultur-Korrektur** (genetischer Prior; lamarckistische Vererbungspfade entfernen — `simulation.py`).
4. **Meilenstein-1-Pilot** (Emergenz-A/B, GPU-PC).
