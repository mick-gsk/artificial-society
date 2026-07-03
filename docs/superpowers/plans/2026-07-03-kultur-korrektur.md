# Kultur-Korrektur (Plan 4) — Implementierungsplan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lamarckistische Geburts-Vererbung (Gelerntes Elter→Kind) für den v2-Pfad entfernen — Kinder starten mit frisch zufallsinitialisiertem Netz und leeren Lern-Stores; nur Gene werden vererbt.

**Architecture:** Die real feuernden Vererbungspfade (A1/A7/A8/A9 in `simulation.spawn_child_from_parent`, A10 in `evolution.make_child`) werden bei `physics_v2` übersprungen. Gating über `self.physics_v2` (simulation) bzw. `parent.physics_v2` (evolution — child-Flag ist zum Zeitpunkt noch False). Tote Pfade (A3–A6, latente inherit_from-Methoden) werden per Test verriegelt statt gegated. v1 bleibt byte-identisch (kein Golden-Bruch).

**Tech Stack:** Python 3.9, torch, pytest. Spec: `docs/superpowers/specs/2026-07-03-kultur-korrektur-design.md`.

## Global Constraints

- **v1 byte-identisch:** jeder Gate der Form `if not self.physics_v2:` / `if not parent.physics_v2:` lässt den v1-Zweig unverändert (die Bedingung selbst zieht keine RNG). Golden-Trajektorie + Headless-Digest MÜSSEN grün bleiben. Niemals einen Determinismus-Test anpassen.
- **v2 hat keinen Golden-Kontrakt** (Spec §7): der durch Überspringen verschobene v2-RNG-Strom (torch + random) ist zulässig.
- **Reihenfolge unangetastet:** `attach_body(child)` (frisches v2-Brain) läuft weiter VOR dem gegateten A1; A1 wird nur übersprungen, nicht verschoben.
- **Kein neuer Prior:** kein Gewichts-/Init-Prior-Mechanismus (User-Entscheidung: reiner Zufalls-Init).
- **ruff NUR auf geänderte Dateien.** Volle Suite: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest -q` — exakte Zahl reporten; maßgeblich ist: kein Test rot, Zahl monoton wachsend (Summary-Zeile wird in dieser Sandbox oft unterdrückt → per `--junitxml` gegenprüfen).
- **Hot-Files:** simulation.py ist eingefroren (core-lane); hier auf eigenem Branch seriell bearbeitet — ok. evolution.py ist systems-lane, reproduktions-nah.
- Commit-Trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Basis: main @ 3664a41, Suite-Start 310.

---

### Task 1: A1/A7/A8/A9 in `spawn_child_from_parent` v2-gaten

**Files:**
- Modify: `artificial_society/simulation.py:209-244` (+ Kommentar bei `_broadcast_death_knowledge:247`)
- Test: `tests/test_physics_v2_culture.py` (NEU)

**Interfaces:**
- Consumes: `Simulation(physics_v2=True)`, `spawn_child_from_parent(parent, genes)`, `attach_body` (unverändert), `Brain.inherit_weights_from` (bleibt als Methode — nur der Geburts-Aufruf entfällt bei v2).
- Produces: v2-Kinder haben frisches Brain (Gewichte ≠ Elter), `causal_memory is None`, leeres `remedy_knowledge`, keine `mat_`-Einträge aus dem Elter.

- [ ] **Step 1: Failing Test schreiben**

Neue Datei `tests/test_physics_v2_culture.py`:

```python
"""Plan 4: v2 vererbt kein Gelerntes bei Geburt — nur Gene. v1 byte-identisch (Golden)."""

from __future__ import annotations

import torch

from artificial_society.simulation import CausalMemory, Simulation


def _v2_sim():
    return Simulation(
        headless=True, load_checkpoint=False, physics_v2=True,
        seed=7, grid_w=20, grid_h=15, initial_population=8,
    )


def _brain_weight_snapshot(brain):
    return [p.detach().clone() for _, p in brain.named_parameters()]


def test_v2_kind_erbt_keine_brain_gewichte():
    sim = _v2_sim()
    parent = sim.agents[0]
    # Elter-Netz von der Zufallsinit wegtrainieren, damit Elter ≠ frisch
    with torch.no_grad():
        for _, p in parent.brain.named_parameters():
            p.add_(torch.randn_like(p) * 0.5)
    parent_w = _brain_weight_snapshot(parent.brain)
    child = sim.spawn_child_from_parent(parent, dict(parent.genes))
    child_w = _brain_weight_snapshot(child.brain)
    paare = [(c, p) for c, p in zip(child_w, parent_w) if c.shape == p.shape]
    assert paare, "keine form-gleichen Tensoren gefunden"
    child_flat = torch.cat([c.flatten() for c, _ in paare])
    parent_flat = torch.cat([p.flatten() for _, p in paare])
    cos = float(torch.dot(child_flat, parent_flat) / (child_flat.norm() * parent_flat.norm()))
    # Kopiertes Netz (A1, strength≈0.5) korreliert stark mit dem gestörten Elter
    # (gemessen cos≈0.99); ein frisches Netz ist unkorreliert (cos≈0.00).
    # max-abs-Abweichung trennt NICHT (beide Fälle > 0.1) — Kosinus schon.
    assert cos < 0.5, f"Kind-Brain korreliert mit Elter (cos={cos:.3f}) — A1 nicht gegated"


def test_v2_kind_erbt_keine_lern_stores():
    sim = _v2_sim()
    parent = sim.agents[0]
    parent.causal_memory = CausalMemory(capacity=32)
    parent.causal_memory.receive_transmitted(("strike", "mat_flint", "mat_stone"), fidelity=0.9)
    parent.remedy_knowledge = {"cough": ["herb_b"]}
    parent.material_inventory = {"mat_flint": 2.0, "water": 1.0}
    child = sim.spawn_child_from_parent(parent, dict(parent.genes))
    assert getattr(child, "causal_memory", None) is None or not child.causal_memory.sequences
    assert not child.remedy_knowledge
    child_mats = getattr(child, "material_inventory", {})
    assert not any(m.startswith("mat_") for m in child_mats), "mat_-Discovery vererbt (A9)"
```

- [ ] **Step 2: Tests laufen — RED**

Run: `../venv/bin/python -m pytest tests/test_physics_v2_culture.py -q`
Expected: beide FAIL (Kind erbt heute Gewichte + Stores).

- [ ] **Step 3: Gating implementieren**

In `simulation.py`, `spawn_child_from_parent`, den Block Zeile 216–244 (die vier Lamarck-Pfade A1/A7/A8/A9) in `if not self.physics_v2:` einrücken. Der v1-Zweig läuft danach byte-identisch; v2 überspringt A1/A7/A8/A9. `attach_body`-Aufruf (Z.215) bleibt VOR diesem Block, unverändert:

```python
        if self.physics_v2:
            # strength wird über den eigenen v2-Pfad vererbt (inherit_genes
            # überspringt es — Golden), DANN baut attach_body den Body daraus.
            # ALIAS beachten: `inherit_strength` ist in dieser Funktion bereits
            # die lokale Gewichts-Vererbungsstärke — daher inherit_strength_gene.
            inherit_strength_gene(child.genes, parent, other_parent)
            attach_body(child)
        if not self.physics_v2:
            # Plan 4 (Kultur-Korrektur): im v2-Pfad wird KEIN Gelerntes vererbt —
            # Kind startet mit frischem Netz (attach_body) + leeren Lern-Stores.
            # Nur Gene (inkl. strength) gehen ans Kind. Kultur überlebt allein
            # über soziales Lernen zu Lebzeiten.
            inherit_strength = max(
                0.20, min(0.75, 0.75 - (child.genes["plasticity"] - 0.3) / (1.8 - 0.3) * 0.55)
            )
            child.brain.inherit_weights_from(parent.brain, strength=inherit_strength)
            if other_parent is not None:
                child.brain.inherit_weights_from(other_parent.brain, strength=inherit_strength * 0.4)
            parent_mem = getattr(parent, "causal_memory", None)
            if parent_mem:
                child.causal_memory = CausalMemory(capacity=32)
                for seq in list(parent_mem.sequences.keys())[:INHERIT_SEQUENCES]:
                    child.causal_memory.receive_transmitted(seq, fidelity=INHERIT_FIDELITY)
            if parent.remedy_knowledge:
                for disease, herbs in parent.remedy_knowledge.items():
                    if random.random() < INHERIT_FIDELITY:
                        child.remedy_knowledge[disease] = list(herbs)
            parent_inv = getattr(parent, "material_inventory", {})
            parent_discoveries = {
                m: q
                for m, q in parent_inv.items()
                if m.startswith("mat_") and q >= DEATH_MATERIAL_MIN_QTY
            }
            if parent_discoveries:
                child_inv = getattr(child, "material_inventory", {})
                for mat_id, qty in parent_discoveries.items():
                    if random.random() < INHERIT_FIDELITY:
                        child_inv[mat_id] = (
                            child_inv.get(mat_id, 0.0) + qty * DEATH_MATERIAL_TRANSFER_RATIO
                        )
                child.material_inventory = child_inv
        return child
```

Zusätzlich ein Kommentar direkt über `_broadcast_death_knowledge` (Z.247), der die bewusste Scope-Entscheidung festhält:

```python
    # Plan 4 (bekannte Abweichung, bewusst belassen): Todes-Broadcast ist KEIN
    # Geburts-Erbgang, sondern horizontaler Transfer an anwesende Sippe — aber
    # beobachtungsfrei (schwächer als der reguläre Sozial-Lern-Kanal). Backlog:
    # später über Fidelity/Trust/Beobachtung routen. Nicht in diesem Schnitt.
    def _broadcast_death_knowledge(self, agent):
```

- [ ] **Step 4: Tests laufen — GREEN**

Run: `../venv/bin/python -m pytest tests/test_physics_v2_culture.py -q` → PASS.
Dann volle Suite (v1-Golden byte-identisch): `rm -f checkpoint.pkl && ../venv/bin/python -m pytest -q` — kein Test rot, Golden + Digest grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/python -m ruff check --fix artificial_society/simulation.py tests/test_physics_v2_culture.py
../venv/bin/python -m ruff format artificial_society/simulation.py tests/test_physics_v2_culture.py
git add artificial_society/simulation.py tests/test_physics_v2_culture.py
git commit -m "feat(plan4): v2 vererbt keine Brain-Gewichte/Lern-Stores bei Geburt (A1/A7/A8/A9 gegated)"
```

---

### Task 2: A10 in `evolution.make_child` v2-gaten (`parent.physics_v2`)

**Files:**
- Modify: `artificial_society/systems/evolution.py:39-46`
- Test: `tests/test_physics_v2_culture.py` (anhängen)

**Interfaces:**
- Consumes: `make_child(parent, x, y, genes, other_parent)`, `parent.physics_v2` (v2-Agenten tragen es; v1-Agenten haben es False oder nicht — daher `getattr(parent, "physics_v2", False)`).
- Produces: v2-Kind hat leeres `memory.resource_memory`.

- [ ] **Step 1: Failing Test schreiben** — an `tests/test_physics_v2_culture.py` anhängen:

```python
def test_v2_kind_erbt_kein_resource_memory():
    sim = _v2_sim()
    parent = sim.agents[0]
    parent.memory.resource_memory = [(2, 2), (3, 3), (4, 4)]
    child = sim.spawn_child_from_parent(parent, dict(parent.genes))
    assert child.memory.resource_memory == [], "resource_memory vererbt (A10 nicht gegated)"
```

- [ ] **Step 2: RED** — `../venv/bin/python -m pytest tests/test_physics_v2_culture.py::test_v2_kind_erbt_kein_resource_memory -q` → FAIL.

- [ ] **Step 3: Gating** — in `evolution.py` den Block Z.39-46 in `if not getattr(parent, "physics_v2", False):` einrücken. **`parent`, nicht `child`/`self`** (child.physics_v2 ist hier noch False; attach_body läuft erst später):

```python
        if not getattr(parent, "physics_v2", False):
            # Plan 4: v2-Kinder erben keine gelernten Ressourcen-Erinnerungen.
            # Gate auf parent.physics_v2 — child.physics_v2 wird erst durch das
            # nachgelagerte attach_body() in spawn_child_from_parent True.
            if parent.memory.resource_memory:
                child.memory.resource_memory = parent.memory.resource_memory[-3:]
            if other_parent and other_parent.memory.resource_memory:
                extra = other_parent.memory.resource_memory[-2:]
                for mem in extra:
                    if mem not in child.memory.resource_memory:
                        child.memory.resource_memory.append(mem)
```

Der `trust`-Prior-Block (Z.48-51) bleibt UNVERÄNDERT und AUSSERHALB des Gates (fester Verwandtschafts-Prior, kein Lamarck).

- [ ] **Step 4: GREEN** — Task-Test PASS; volle Suite grün, Golden byte-identisch.

- [ ] **Step 5: Commit**

```bash
../venv/bin/python -m ruff check --fix artificial_society/systems/evolution.py tests/test_physics_v2_culture.py
../venv/bin/python -m ruff format artificial_society/systems/evolution.py tests/test_physics_v2_culture.py
git add artificial_society/systems/evolution.py tests/test_physics_v2_culture.py
git commit -m "feat(plan4): v2-Kind erbt kein resource_memory (A10 auf parent.physics_v2 gegated)"
```

---

### Task 3: Tote Pfade verriegeln + Positiv-Kontrollen + Determinismus-Smoke

**Files:**
- Test: `tests/test_physics_v2_culture.py` (anhängen) — reine Tests, KEIN Produktionscode.

**Interfaces:**
- Consumes: `EvolutionSystem.make_child`, `Agent.spawn_child`, `genetics.inherit_genes`, `Brain.imitate_from`.
- Produces: nichts — Härtungs- und Kontroll-Tests.

- [ ] **Step 1: Verriegelungs-Tests (tote Pfade A3–A6 + latente Methoden)**

```python
import ast
import inspect
import textwrap

from artificial_society.systems.evolution import EvolutionSystem


def test_spawn_child_wird_nie_mit_parent_aufgerufen():
    """A3–A6 (ToM/Knowledge/EmotionalMemory/world_memory) stehen in
    Agent.spawn_child unter `if parent is not None`; make_child ruft es ohne
    parent. Schlüge jemand parent= durch, würden diese Pfade schlagartig live
    und bräuchten ein parent.physics_v2-Gate. (Substring-Check scheidet aus:
    die Signatur enthält `other_parent=None`, und "parent=" ist Substring von
    "other_parent=" → AST-basiert prüfen.)"""
    src = textwrap.dedent(inspect.getsource(EvolutionSystem.make_child))
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "spawn_child"
    ]
    assert calls, "make_child ruft Agent.spawn_child nicht mehr auf?"
    for call in calls:
        assert "parent" not in {kw.arg for kw in call.keywords}, (
            "make_child reicht parent= durch — A3–A6 wären live!"
        )
        assert len(call.args) <= 6  # parent ist der 7. Positionsparameter


def test_latente_inherit_from_methoden_feuern_nicht_bei_geburt():
    """StrategySystem.inherit_from / EpisodicStrategyMemory.inherit_from sind
    uncalled — Agenten tragen die Attribute nicht. Wächter gegen versehentliches
    Verdrahten."""
    sim = _v2_sim()
    a = sim.agents[0]
    assert not hasattr(a, "strategy")
    assert not hasattr(a, "strategy_memory")
    assert not hasattr(a, "episodic_strategy")
```

- [ ] **Step 2: Positiv-Kontrollen (Genetik + Lebzeit-Lernen bleiben)**

```python
def test_v2_kind_erbt_weiter_gene_und_trust_prior():
    sim = _v2_sim()
    parent = sim.agents[0]
    parent.tribe_id = 1  # frisches Sim hat tribe_id=None → trust-Zweig feuerte nie
    # strength aus den übergebenen Genen entfernen → prüft den v2-Pfad
    # (inherit_strength_gene) echt, statt tautologisch die Dict-Kopie
    genes = dict(parent.genes)
    genes.pop("strength", None)
    child = sim.spawn_child_from_parent(parent, genes)
    assert "strength" in child.genes, "strength-Gen wurde nicht via v2-Pfad ergänzt"
    # Verwandtschafts-Prior bleibt (fester Wert, kein Lamarck)
    assert child.trust.get(parent.id) == 0.4


def test_lebzeit_imitation_bleibt_intakt():
    """A2 (Brain.imitate_from) ist der Lebzeit-Kanal — vom A1-Gating unberührt."""
    sim = _v2_sim()
    a, b = sim.agents[0], sim.agents[1]
    before = [p.detach().clone() for _, p in a.brain.named_parameters()]
    a.brain.imitate_from(b.brain, strength=0.5)
    after = [p.detach().clone() for _, p in a.brain.named_parameters()]
    diffs = [float((x - y).abs().max()) for x, y in zip(after, before) if x.shape == y.shape]
    assert max(diffs) > 0.0, "imitate_from wirkt nicht mehr — Lebzeit-Kanal beschädigt"
```

- [ ] **Step 3: Determinismus-/Erhaltungs-Smoke (v2 mit Geburten)**

```python
import math


def test_v2_geburten_smoke_nan_frei():
    """Kurzer v2-Lauf mit garantierter Geburt bleibt NaN-frei (Sanity nach dem Umbau)."""
    sim = _v2_sim()
    # Geburt erzwingen (der reguläre Caller extendet sim.agents selbst; hier manuell)
    sim.agents.append(sim.spawn_child_from_parent(sim.agents[0], dict(sim.agents[0].genes)))
    for _ in range(40):
        sim.step()
    assert len(sim.agents) > 0
    assert all(math.isfinite(float(a.last_reward)) for a in sim.agents)
    assert all(not torch.isnan(a.hidden_state).any() for a in sim.agents)
```

- [ ] **Step 4: Alle Tests laufen — GREEN**

Run: `../venv/bin/python -m pytest tests/test_physics_v2_culture.py -q` → alle PASS.
Volle Suite: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest -q` → kein Test rot, Golden grün.

- [ ] **Step 5: Commit**

```bash
../venv/bin/python -m ruff check --fix tests/test_physics_v2_culture.py
../venv/bin/python -m ruff format tests/test_physics_v2_culture.py
git add tests/test_physics_v2_culture.py
git commit -m "test(plan4): tote Vererbungspfade verriegelt + Genetik/Lebzeit-Positivkontrollen + v2-Geburten-Smoke"
```

---

## Self-Review

- **Spec-Coverage:** A1/A7/A8/A9 (Task 1), A10 (Task 2), tote Pfade A3–A6 + latente Methoden (Task 3 Verriegelung), Genetik-Positivliste + trust-Prior + Lebzeit-Kanal (Task 3 Kontrollen), Todes-Broadcast (Task 1 Kommentar + Design-Backlog), v1-Golden (jede Task-Suite). Reiner Zufalls-Init: durch A1-Gating erfüllt (attach_body-Netz bleibt frisch). ✓
- **Kein Placeholder:** alle Test- und Prod-Blöcke ausgeschrieben. ✓
- **Flag-Konsistenz:** Task 1 `self.physics_v2`, Task 2 `getattr(parent, "physics_v2", False)` — bewusst verschieden (child-Flag-Timing), im Plan begründet. ✓
- **Massenerhaltung (Design §5):** der v2-Geburten-Smoke prüft NaN-Freiheit; „massenerhaltend" ist über die bestehenden Erhaltungstests der vollen Suite (`tests/test_energy_conservation.py`, `tests/test_physics_v2_death.py`) abgedeckt, die je Task mitlaufen. ✓
- **Plan-Review (fable) eingearbeitet:** P1 (Brain-Test → Kosinus-Diskriminator, max-abs trennte nicht), P2 (Verriegelungstest → AST statt Substring, `parent=`⊂`other_parent=`), E1 (ALIAS-Kommentar behalten), E2 (Positiv-Kontrolle: tribe_id=1 + strength-pop), E3 (Geburt im Smoke erzwungen), E4 (Massenerhaltung dokumentiert). Prod-Code als v1-byte-identisch + Flag-Timing bestätigt. ✓
