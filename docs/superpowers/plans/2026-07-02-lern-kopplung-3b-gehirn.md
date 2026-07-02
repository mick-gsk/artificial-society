# Plan 3b — Gehirn (Lern-Kopplung) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-07-02-lern-kopplung-design.md` (Rev. 4) — Scope: **Abschnitt C komplett (C1–C5), D3, D4-„erst nach 3b"-Metriken** plus die Welt-Nacharbeiten F4/F5/F6 aus dem 3a-Final-Review. Abschnitt B ist gebaut (Plan 3a, gemergt in main @ faf0825).

**Goal:** Die Physik-v2-Welt (Objekte, verkörperte Aktionen) an das lernende Gehirn koppeln: Attention-Wahrnehmung über Objekt-Slots, neue Aktionsköpfe mit kategorialer Slot-Auswahl, reiner Überlebens+Neugier-Reward (C3), drei-Quellen-Neugier (C4) mit torch-Causal-Model, `strength`-Gen, PPO-Anpassungen, Checkpoint-Versionierung — alles ausschließlich im v2-Modus, v1 bleibt byte-identisch.

**Architecture:** Der v2-Modus ist eine additive Schicht im bestehenden `Brain`/`Agent`/`Simulation`-Code, umgeschaltet über das existierende `physics_v2`-Flag. Neue Netz-Teile (Slot-Encoder, Attention, neue Köpfe, Slot-Prädiktions-Head) existieren NUR im v2-Brain; das v1-Brain bleibt form-identisch (gleiche Module, Shapes und Init-RNG-Reihenfolge wie heute). Wahrnehmung/Masken liefert das neue `agents/perception_v2.py`; das Causal Model ist eine torch-Neuimplementierung in `systems/causal_model.py` (pro Agent, KEIN registriertes Sim-System).

**Tech Stack:** Python 3.9, PyTorch (CPU-Default via `AS_BRAIN_DEVICE`), NumPy, pytest. Venv: `../venv/bin/python` (liegt eine Ebene über dem Worktree).

## Architektur-Vorentscheidung Brain v1/v2 (bindend, aus der Spec)

- `Brain(physics_v2: bool = False)` — das Flag bestimmt die Modulliste:
  - **v1 (`physics_v2=False`):** exakt heutige Module in exakt heutiger Konstruktions-Reihenfolge
    (encoder → gru(128→96) → policy_mean(96→7) → policy_logstd(7) → value_head → world_fc(103→128→128)
    → next_obs_head → reward_head). Dadurch bleiben Init-RNG-Draws, `state_dict`-Keys und alle
    Shapes identisch ⇒ Golden grün, und Legacy-v1-Checkpoints (Format 1, auch ohne Version-Key)
    laden im v1-Modus unverändert weiter (Task 16).
    `forward`, `act`, `evaluate_actions`, `maybe_train`-Kern, `plan_action`, `imagine_rollout`,
    `intrinsic_reward` werden für v1 NICHT verändert.
  - **v2 (`physics_v2=True`):** `action_size = 29`, `gru = GRUCell(192 → 96)`, ZUSÄTZLICH (nach den
    v1-Modulen konstruiert): `slot_embed` (Linear(17→32)+LayerNorm), `attn_query` (Linear(96→32)),
    `w_sel` (Linear(32→8, bias=False)), `next_slots_head` (Linear(128→170)), Neugier-Statistik-Buffer.
    Eigene Methoden `forward_v2`/`act_v2`/`evaluate_actions_v2`/`predict_world_v2`/`store_transition_v2`/
    `_train_v2`/`finalize_terminal` — die v1-Methoden bleiben unangetastet.
- **`research_drive` (Dim 6)** bleibt in BEIDEN Modi als Kopf bestehen (Form-Kompatibilität), wird
  im v2-Modus aber aus der log-prob-Summe ausgenommen (totes Rauschen im PPO-Ratio; Spec C2).
- **Planner:** `plan_action`/`imagine_rollout` werden im v2-Pfad schlicht nie aufgerufen
  (`act_v2` sampelt immer direkt); v1 unverändert. `inherit_weights_from` überspringt
  Shape-Mismatches still — beim Architektur-Wechsel gewollt, wird im Code kommentiert (Spec C5).
- **Agent-Seite:** `attach_body` (v2-Spawn-Pfad) baut das Brain als `Brain(physics_v2=True)` neu
  auf (bevor `inherit_weights_from` vom v2-Eltern-Brain läuft — Shapes passen dann). `ensure_fields`
  rüstet Checkpoint-geladene v2-Agenten idempotent nach.

## Global Constraints

Jede Task-Anforderung schließt diese Punkte implizit ein:

1. **`physics_v2=False` = byte-identisches v1-Verhalten.** Jede Gehirn-/Agent-/Sim-Änderung muss im
   v1-Pfad exakt das alte Verhalten liefern: gleiche Modul-Konstruktion (Init-RNG-Reihenfolge!),
   gleiche RNG-Draw-Folge im Tick, kein neuer Draw außerhalb von `if physics_v2:`. Golden
   (`tests/test_regression_golden.py`) und Headless-Digest müssen grün bleiben. NIE einen
   Determinismus-Test anpassen, um ihn grün zu bekommen.
2. **Volle Repo-Suite vor JEDEM Commit:** `cd /Users/moritzbecker/projekt/as-lern-kopplung-3b && ../venv/bin/python -m pytest -q` — die **exakte Testzahl reporten** (Stand vor Task 1: **234 Tests**, alle grün). Immer `../venv/bin/python` benutzen.
3. **`rm -f checkpoint.pkl`** im Worktree-Root vor jedem Sim-Lauf (stales root-`checkpoint.pkl` wird sonst auto-geladen).
4. **ruff NUR auf geänderte Dateien** (`../venv/bin/python -m ruff check --fix <dateien> && ../venv/bin/python -m ruff format <dateien>`) — das Repo ist repo-weit ruff-kontaminiert, NIE repo-weit linten/formatieren.
5. **Alle Zahlen aus der Spec verbatim** (Tabelle unten) — keine „ähnlichen" Werte.
6. **Sampling, nie Argmax:** kontinuierliche Köpfe via `rsample`+tanh, Slot-Auswahl via `Categorical.sample()`. (Die Auflösung „höchstes Verb > 0.5 gewinnt" ist deterministisches Mapping eines GESAMPELTEN Vektors — erlaubt.)
7. **Keine Shaping-Boni:** v2-Reward = exakt die C3-Terme (`r_energy + r_health + r_deficit [+ r_death terminal] + 0.3·curiosity`), sonst nichts — kein `cognition_mult`, kein Event-Bonus, kein Makro-Bonus.
8. **Arbeitsverzeichnis:** AUSSCHLIESSLICH `/Users/moritzbecker/projekt/as-lern-kopplung-3b` (Branch `feat/lern-kopplung-3b`). Den Haupt-Checkout `/Users/moritzbecker/projekt/artificial-society` NICHT lesen/anfassen.
9. **Determinismus:** neue Randomness nur über die von `artificial_society.rng.seed_all` geseedeten globalen Ströme (`random`-Modul, torch); keine eigenen Seeds/Generatoren zur Laufzeit.
10. **Hot Files:** `brain.py`, `agent.py`, `simulation.py` sind Hot Files — dieser Plan editiert sie absichtlich (Spec weist sie 3b zu, Branch ist die 3b-Lane); minimal-invasiv, jede Änderung v1-pfad-neutral.

## Spec-Zahlen (verbatim, SSOT für alle Tasks)

| Bereich | Wert |
|---|---|
| Slots | K = 8 Boden (Chebyshev r ≤ 4) + 2 Hand = **10**; **17** Features/Slot: 13 Props + `mass_kg/25` (Kappe 1) + `dx/4` + `dy/4` + `held` |
| Encoder | `embed = LayerNorm(Linear(17→32))`; Query `Linear(96→32)` aus `h_prev`; masked softmax mit `/√32`; **obj_ctx = attn-Pool ⊕ masked-Max-Pool = 64**; GRU-Input **128+64=192**; All-Masked ⇒ `obj_ctx = 0` |
| Köpfe v2 | 5 Verben (grasp, release, strike, cut, eat) + 1 Effort + 8 Target-Query + 8 Tool-Query = 22 neue kont. Dims ⇒ **29 kontinuierlich gesamt** + **2 kategoriale** Auswahlen; Verb-Aktivierung > **0.5**, höchstes gewinnt; **eine** verkörperte Aktion/Tick |
| Slot-Auswahl | Logits = `query · W_sel(embed_i) / √8` über zulässige Slots; Ziehung kategorial; log-prob in Policy-log-prob; leere Menge ⇒ No-op ohne log-prob-Anteil; Transition speichert Maske + Slot-Index |
| PPO v2 | GAMMA **0.99**, PPO_EPOCHS **4**, Minibatches **4×32** (128er-Buffer), KL-Early-Stop **0.02**, log_std-Floor **−0.8** (neue Köpfe), Entropy **0.004** (alte Gruppe) / **0.01** (neue Gruppe inkl. Kategorial), Init-Bias **+0.2** auf Verb-Means |
| Reward (C3) | `r_energy = 0.6·ΔE/45.0`; `r_health = 0.6·ΔH/50.0`; `r_deficit = −0.3·max(0,(60−E)/60)`; `r_death = −3.0` (echte Terminal-Transition); `r_int = 0.3·curiosity` |
| Neugier (C4) | `curiosity = 0.25·nextslot_err + 0.50·causal_epistemic + 0.25·novelty`, geclampt **[0, 2]**; ausgeschlossene Obs-Dims: **26** (last_reward), **34–36** (Causal-Memory), **37–48** (Episodic), **49–56** (Hormone); Normalisierung per Dim (laufendes Mean+Std); `causal_epistemic = max(0, mean(z²) − 1)` (**Hinge außen**), log σ-Floor **≥ −2**; novelty = `1/√n` über Buckets `floor(p/0.25)` (13 Props), pro Agent, LRU **4096** |
| Causal Model | Input `detach(gru_h 96) ⊕ Aktions-Vektor (22) ⊕ detach(Slot-Embed 32)` = 150 → (μ, log σ) über **17** Features des getrackten Objekts im nächsten Tick; NLL; eigener Optimizer, LR × Plastizitäts-Gen; `err_ema` fürs Logging; bei strike-Zerstörung: massereichstes Fragment; aus Sicht gefallen: maskiert (kein Loss) |
| Gene | `strength` (16. Gen), Mutations-σ **0.012**-Klasse; speist `Body(strength=...)` statt Default 0.5 |
| Checkpoint | `CHECKPOINT_FORMAT_VERSION = 2` beim Speichern; Laden: fehlender Key ⇒ Version 1 (Legacy lädt NUR mit Flag aus — 3a-Garantie bleibt), Version ∉ {1, 2} ⇒ harter Fehler; v2-Brain-Formen-Guard VOR `ensure_fields`; alle Guards VOR dem broad-except re-raised |
| F4/F5 | ε-Cull: `mass < 1e-6` ⇒ Rest nach `ledger["decayed"]`, remove; gehaltene Objekte verwesen mit (tick_decay über Hände lebender Agenten) |

## File-Struktur (was entsteht/ändert sich)

| Datei | Verantwortung |
|---|---|
| `artificial_society/environment/phys_objects.py` (ändern) | F4 ε-Cull, F5 Hand-Verwesung (`tick_decay(layer, hands_list=())`) |
| `artificial_society/systems/physics_v2.py` (ändern) | Hand-Liste der lebenden Agenten an `tick_decay` durchreichen |
| `artificial_society/environment/physics/actions.py` (ändern, additiv) | `ActionResult.remainder`-Feld (Causal-Target-Fortschreibung bei cut) |
| `artificial_society/agents/genetics.py` (ändern) | `strength`-Gen: `GENE_RANGES`-Eintrag + `ensure_strength_gene`/`inherit_strength` (v2-only-Draws), `inherit_genes` überspringt `strength` |
| `artificial_society/agents/perception_v2.py` (NEU) | `SlotView`/`build_slots`/`admissible_masks`/`resolve_slot_of`/`NoveltyBuckets` — Slots, Maskierung, Identitäts-Tracking, Count-Novelty |
| `artificial_society/agents/brain.py` (Hot, ändern) | v2-Konstruktor-Flag + Module, `forward_v2`, `act_v2`, `evaluate_actions_v2`, `predict_world_v2`, `nextslot_error`, `store_transition_v2`, `_train_v2`, `finalize_terminal`; v1-Pfad byte-identisch |
| `artificial_society/systems/causal_model.py` (NEU) | `CausalModelV2` (torch, NLL, Hinge-Epistemik, σ-Floor, err_ema) — pro Agent, kein registriertes System |
| `artificial_society/agents/agent.py` (Hot, ändern) | v2: Slot-Wahrnehmung, `act_v2`-Aufruf, Verb→`do_*`-Mapping (`_execute_embodied`), C3-Reward, Neugier-Assemblierung, Causal-Pending, D4-Metriken; `attach_body`/`ensure_fields`-Erweiterung |
| `simulation.py` (Hot, ändern) | `inherit_strength` im v2-Kind-Pfad, Terminal-Finalize in `remove_dead`, `CHECKPOINT_FORMAT_VERSION = 2` + Loader-Guard |
| Tests | `tests/environment/test_phys_objects_cull.py`, `tests/agents/test_genetics_strength.py`, `tests/agents/test_perception_v2.py`, `tests/agents/test_brain_v2_forms.py`, `tests/agents/test_brain_v2_policy.py`, `tests/agents/test_brain_v2_training.py`, `tests/agents/test_curiosity_v2.py`, `tests/systems/test_causal_model_v2.py`, `tests/test_physics_v2_reward.py`, `tests/test_physics_v2_brain_integration.py`; Erweiterungen an `tests/test_physics_v2_system.py`, `tests/test_physics_v2_checkpoint.py`, `tests/environment/test_conservation_fuzz.py`, `tests/environment/physics/test_actions.py` |

**Dim-Layout v2 (29 kontinuierliche Dims, SSOT):**

```
0..6   v1-Köpfe: move_x, move_y, forage, cooperate, attack, build, research_drive
7..11  Verben:   grasp(7), release(8), strike(9), cut(10), eat(11)
12     Effort    (tanh-Output ∈ [−1,1] → effort01 = (a+1)/2)
13..20 Target-Query (8, gelernter Embedding-Raum)
21..28 Tool-Query   (8)
```

**Slot-Layout (10 Slots, SSOT):** Slots 0..7 = Boden (nächste 8 nach Chebyshev, stabile Sortierung), Slots 8..9 = Hand (`hands.held[0]`, `hands.held[1]`).

---

### Task 1: F4 — ε-Cull in `tick_decay` (Husks bilanziert entfernen)

Verwesende Objekte nähern sich asymptotisch 0 kg, verschwinden aber nie — sie fluten später die 8 Wahrnehmungs-Slots. Fix: Objekte mit `mass < 1e-6` werden entfernt, ihr Rest fließt bilanziert in `ledger["decayed"]` (kein Leck; die D1-Invariante hält).

**Files:**
- Modify: `artificial_society/environment/phys_objects.py` (Funktion `tick_decay`, Zeilen 302–317)
- Test: `tests/environment/test_phys_objects_cull.py` (NEU)

**Interfaces:**
- Consumes: `ObjectLayer` (bestehend: `all_objects`, `remove`, `ledger`, `conservation_terms`).
- Produces: `EPSILON_CULL_MASS_KG = 1e-6` (Modul-Konstante in `phys_objects.py`); `tick_decay(layer)` entfernt danach ALLE Objekte < 1e-6 kg (auch nicht-verwesende Winz-Fragmente). Task 2 baut auf dieser Struktur auf.

- [ ] **Step 1: Failing Test schreiben**

Neue Datei `tests/environment/test_phys_objects_cull.py`:

```python
"""F4/F5 (3a-Final-Review): ε-Cull + Verwesung gehaltener Objekte — ledger-bilanziert."""

from __future__ import annotations

import math
import random

import pytest

from artificial_society.environment.phys_objects import (
    EPSILON_CULL_MASS_KG,
    ObjectLayer,
    tick_decay,
)
from artificial_society.environment.physics.objects import make_object


def _layer():
    return ObjectLayer(8, 8, rng=random.Random(1))


def test_epsilon_cull_entfernt_husk_bilanziert():
    layer = _layer()
    layer.add(make_object("carcass", 5e-7), (2, 2), source="from_carcass")
    tick_decay(layer)
    assert layer.objects_at((2, 2)) == [], "Husk muss entfernt sein"
    assert layer.ledger["decayed"] == pytest.approx(5e-7), "Rest fließt nach ledger['decayed']"
    lhs, rhs = layer.conservation_terms()
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)


def test_epsilon_cull_wirkt_auch_auf_nicht_verwesende_winzlinge():
    """Winz-Fragmente aus strike/cut (Stein) verwesen nie — dürfen die Slots trotzdem
    nicht fluten: der Cull greift unabhängig vom Verwesungs-Gate."""
    layer = _layer()
    layer.add(make_object("granite", 5e-7), (3, 3), source="spawned")
    tick_decay(layer)
    assert layer.objects_at((3, 3)) == []
    assert layer.ledger["decayed"] == pytest.approx(5e-7)


def test_normale_objekte_bleiben_unberuehrt_vom_cull():
    layer = _layer()
    stein = make_object("granite", 2.0)
    layer.add(stein, (1, 1), source="spawned")
    tick_decay(layer)
    assert layer.objects_at((1, 1)) == [stein]
    assert stein.mass == 2.0  # Granit verwest nicht, Cull greift nicht
    assert EPSILON_CULL_MASS_KG == 1e-6
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `cd /Users/moritzbecker/projekt/as-lern-kopplung-3b && ../venv/bin/python -m pytest tests/environment/test_phys_objects_cull.py -q`
Expected: `ImportError: cannot import name 'EPSILON_CULL_MASS_KG'`

- [ ] **Step 3: Implementierung**

In `artificial_society/environment/phys_objects.py` die Funktion `tick_decay` (aktuell Zeilen 302–317) durch folgenden Block ersetzen (Konstante direkt darüber einfügen):

```python
EPSILON_CULL_MASS_KG = 1e-6  # F4 (3a-Final-Review): Husks unterhalb dieser Masse
# werden bilanziert entfernt (Rest → ledger['decayed']) — sonst fluten asymptotisch
# nie verschwindende Winz-Objekte die 8 Wahrnehmungs-Slots der Agenten (Plan 3b, C1).


def _decay_obj(obj, layer: ObjectLayer) -> None:
    """Ein Verwesungs-Schritt für EIN Objekt (eigenschaftsbasiertes Gate, B3.4)."""
    moisture = float(obj.props[IDX2["moisture"]])
    nutrition = float(obj.props[IDX2["nutrition"]])
    if moisture < DECAY_MOISTURE_MIN or nutrition <= 0.0:
        return
    verlust = obj.mass * DECAY_RATE
    obj.mass -= verlust
    layer.ledger["decayed"] += verlust
    obj.props[IDX2["nutrition"]] = nutrition * (1.0 - DECAY_RATE)
    tox = float(obj.props[IDX2["toxicity"]])
    if tox < TOX_SPOILAGE_CAP:
        obj.props[IDX2["toxicity"]] = min(TOX_SPOILAGE_CAP, tox + TOX_SPOILAGE_PER_TICK)


def tick_decay(layer: ObjectLayer) -> None:
    """Ein Verwesungs-Tick über alle Boden-Objekte: Masse und nutrition sinken
    exponentiell, toxicity steigt bis zur Kappe. Verweste Masse fließt
    bilanziert in ledger['decayed'] (kein Leck). ε-Cull (F4): Objekte unter
    EPSILON_CULL_MASS_KG werden bilanziert entfernt — unabhängig vom
    Verwesungs-Gate (auch Winz-Fragmente aus strike/cut)."""
    culls = []
    for obj, _pos in layer.all_objects():
        _decay_obj(obj, layer)
        if obj.mass < EPSILON_CULL_MASS_KG:
            culls.append(obj)
    for obj in culls:
        layer.ledger["decayed"] += obj.mass
        layer.remove(obj)
```

(Die bisherige Schleifen-Logik wandert 1:1 in `_decay_obj`; Verhalten für normale Objekte identisch. `culls` wird NACH der Iteration entfernt — `all_objects` iteriert `_by_pos`, Mutation während der Iteration wäre ein `RuntimeError`.)

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/environment/test_phys_objects_cull.py tests/environment/test_phys_objects.py tests/environment/test_conservation_fuzz.py -q`
Expected: alle grün (`3 passed` für die neue Datei; Fuzzer und Bestands-Tests unverändert grün).

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 237 = 234 + 3)
../venv/bin/python -m ruff check --fix artificial_society/environment/phys_objects.py tests/environment/test_phys_objects_cull.py
../venv/bin/python -m ruff format artificial_society/environment/phys_objects.py tests/environment/test_phys_objects_cull.py
git add artificial_society/environment/phys_objects.py tests/environment/test_phys_objects_cull.py
git commit -m "fix(3a-review F4): epsilon-Cull in tick_decay — Husks bilanziert nach ledger[decayed]"
```

---

### Task 2: F5 — Gehaltene Objekte verwesen mit (Frischhalte-Loophole schließen)

Heute verwest nur Bodenmasse — ein Agent könnte Fleisch ewig frisch halten, indem er es trägt. `tick_decay` bekommt die Hände der lebenden Agenten; ε-Cull greift auch in der Hand.

**Files:**
- Modify: `artificial_society/environment/phys_objects.py` (`tick_decay`-Signatur)
- Modify: `artificial_society/systems/physics_v2.py` (Hand-Liste durchreichen)
- Modify: `tests/environment/test_conservation_fuzz.py` (decay-Aktion mit Händen)
- Test: `tests/environment/test_phys_objects_cull.py` (erweitern)

**Interfaces:**
- Consumes: `_decay_obj`, `EPSILON_CULL_MASS_KG` aus Task 1; `Hands.held`/`Hands.release`.
- Produces: `tick_decay(layer, hands_list=())` — `hands_list` ist ein Iterable von `Hands`-Objekten; Default `()` hält alle bestehenden Aufrufer (3a-Tests) grün.

- [ ] **Step 1: Failing Tests schreiben**

An `tests/environment/test_phys_objects_cull.py` anhängen:

```python
from artificial_society.environment.physics.actions import DECAY_RATE
from artificial_society.environment.physics.body import Hands


def test_gehaltenes_fleisch_verwest_mit():
    """F5: Frischhalte-Loophole zu — Tragen konserviert nicht."""
    layer = _layer()
    hands = Hands()
    fleisch = make_object("raw_meat", 1.0)
    hands.held.append(fleisch)
    layer.ledger["spawned"] += 1.0  # Handbestückung bilanzieren (Testaufbau)
    tick_decay(layer, hands_list=(hands,))
    assert fleisch.mass == pytest.approx(1.0 * (1.0 - DECAY_RATE))
    assert layer.ledger["decayed"] == pytest.approx(1.0 * DECAY_RATE)
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)


def test_gehaltener_stein_verwest_nicht():
    layer = _layer()
    hands = Hands()
    stein = make_object("granite", 1.0)
    hands.held.append(stein)
    layer.ledger["spawned"] += 1.0
    tick_decay(layer, hands_list=(hands,))
    assert stein.mass == 1.0


def test_husk_in_der_hand_wird_gecullt():
    layer = _layer()
    hands = Hands()
    kruemel = make_object("carcass", 5e-7)
    hands.held.append(kruemel)
    layer.ledger["spawned"] += 5e-7
    tick_decay(layer, hands_list=(hands,))
    assert hands.held == []
    assert layer.ledger["decayed"] == pytest.approx(5e-7)
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)


def test_verwesung_der_haende_laeuft_ueber_den_sim_tick():
    """Integrations-Nachweis über das registrierte System (kein direkter Mechanik-Aufruf)."""
    from artificial_society.simulation import Simulation

    sim = Simulation(
        seed=42, physics_v2=True, headless=True, load_checkpoint=False,
        grid_w=20, grid_h=15, initial_population=8,
    )
    traeger = sim.agents[0]
    fleisch = make_object("raw_meat", 1.0)
    traeger.hands.held.append(fleisch)
    sim.world.objects.ledger["spawned"] += 1.0
    masse_vorher = fleisch.mass
    sim.step()
    assert fleisch.mass < masse_vorher, "Hand-Verwesung muss im Sim-Tick laufen"
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/test_phys_objects_cull.py -q`
Expected: 4 neue Tests FAIL (`TypeError: tick_decay() got an unexpected keyword argument 'hands_list'`), 3 alte PASS.

- [ ] **Step 3: Implementierung**

In `artificial_society/environment/phys_objects.py` die `tick_decay`-Funktion aus Task 1 ersetzen durch:

```python
def tick_decay(layer: ObjectLayer, hands_list=()) -> None:
    """Ein Verwesungs-Tick über alle Boden-Objekte UND die Hände lebender
    Agenten (F5: Frischhalte-Loophole zu — Tragen konserviert nicht). Masse und
    nutrition sinken exponentiell, toxicity steigt bis zur Kappe; verweste
    Masse fließt bilanziert in ledger['decayed'] (kein Leck). ε-Cull (F4):
    Objekte unter EPSILON_CULL_MASS_KG werden bilanziert entfernt — am Boden
    UND aus der Hand, unabhängig vom Verwesungs-Gate."""
    culls = []
    for obj, _pos in layer.all_objects():
        _decay_obj(obj, layer)
        if obj.mass < EPSILON_CULL_MASS_KG:
            culls.append(obj)
    for obj in culls:
        layer.ledger["decayed"] += obj.mass
        layer.remove(obj)
    for hands in hands_list:
        for obj in list(hands.held):
            _decay_obj(obj, layer)
            if obj.mass < EPSILON_CULL_MASS_KG:
                layer.ledger["decayed"] += obj.mass
                hands.release(obj)
```

In `artificial_society/systems/physics_v2.py` die `tick`-Methode ersetzen:

```python
    def tick(self, sim, tick: int) -> None:
        if not getattr(sim, "physics_v2", False):
            return
        layer = sim.world.objects
        # F5: gehaltene Objekte verwesen mit (Frischhalte-Loophole zu) — die
        # Hände aller lebenden v2-Agenten laufen durch denselben Verwesungs-Tick.
        hands_list = [
            a.hands
            for a in sim.agents
            if a.alive and getattr(a, "hands", None) is not None
        ]
        tick_decay(layer, hands_list=hands_list)
        tick_spawn(layer, sim.world.biomes)
```

In `tests/environment/test_conservation_fuzz.py` die decay-Aktion (Zeile 99–100) ersetzen:

```python
            elif aktion == "decay":
                tick_decay(layer, hands_list=(hands,))
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/environment/test_phys_objects_cull.py tests/environment/test_conservation_fuzz.py tests/test_physics_v2_system.py -q`
Expected: alle grün (`7 passed` in test_phys_objects_cull.py).

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 241)
../venv/bin/python -m ruff check --fix artificial_society/environment/phys_objects.py artificial_society/systems/physics_v2.py tests/environment/test_phys_objects_cull.py tests/environment/test_conservation_fuzz.py
../venv/bin/python -m ruff format artificial_society/environment/phys_objects.py artificial_society/systems/physics_v2.py tests/environment/test_phys_objects_cull.py tests/environment/test_conservation_fuzz.py
git add -A artificial_society/environment/phys_objects.py artificial_society/systems/physics_v2.py tests/environment/test_phys_objects_cull.py tests/environment/test_conservation_fuzz.py
git commit -m "fix(3a-review F5): gehaltene Objekte verwesen mit — tick_decay ueber Haende lebender Agenten"
```

---

### Task 3: F6 — Ledger-Asserts im Flag-aus-Systemtest nachziehen

Der Flag-aus-Systemtest prüft bisher nur Masse/Objektliste, nicht den Ledger — ein v1-Pfad, der still Ledger-Einträge schriebe, bliebe unbemerkt.

**Files:**
- Modify: `tests/test_physics_v2_system.py` (Funktion `test_gegenprobe_flag_aus_system_wirkt_nicht`)

**Interfaces:**
- Consumes: `ObjectLayer.ledger`, `conservation_terms` (bestehend).
- Produces: nichts Neues — reiner Test-Nachzieher.

- [ ] **Step 1: Test erweitern**

In `tests/test_physics_v2_system.py` ans Ende von `test_gegenprobe_flag_aus_system_wirkt_nicht` (nach `assert boden == [kadaver]`) anhängen:

```python
    # F6 (3a-Final-Review): Ledger-Asserts — der v1-Pfad darf den Ledger nur um
    # die Testaufbau-Buchung (from_carcass 50) bewegen, sonst gar nicht.
    ledger = sim.world.objects.ledger
    assert ledger["spawned"] == 0.0
    assert ledger["eaten"] == 0.0
    assert ledger["decayed"] == 0.0
    assert ledger["from_carcass"] == 50.0
    lhs, rhs = sim.world.objects.conservation_terms()
    assert lhs == rhs == 50.0
```

- [ ] **Step 2: Test laufen lassen — grün (Nachzieher, kein Produktions-Change)**

Run: `../venv/bin/python -m pytest tests/test_physics_v2_system.py -q`
Expected: `3 passed`

- [ ] **Step 3: Volle Suite + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 241)
../venv/bin/python -m ruff check --fix tests/test_physics_v2_system.py && ../venv/bin/python -m ruff format tests/test_physics_v2_system.py
git add tests/test_physics_v2_system.py
git commit -m "test(3a-review F6): Ledger-Asserts im Flag-aus-Systemtest"
```

### Task 4: `strength`-Gen (16. Gen, v2-only RNG-Draws)

Das Gen speist `Body(strength=...)` statt des 3a-Defaults 0.5. **Golden-Falle:** `random_genes`/`inherit_genes` laufen auch im v1-Pfad — ein zusätzlicher RNG-Draw dort verschiebt den globalen `random`-Strom und bricht die Golden-Trajectory. Deshalb: `GENE_RANGES`-Eintrag ja (Spec C5), aber Draw/Vererbung laufen AUSSCHLIESSLICH über neue v2-only-Funktionen; `inherit_genes` überspringt `strength` explizit; `random_genes` (explizites Dict-Literal, iteriert `GENE_RANGES` nicht) bleibt unverändert.

**Files:**
- Modify: `artificial_society/agents/genetics.py`
- Modify: `artificial_society/agents/agent.py` (`attach_body`, `ensure_fields`-Body-Zeile 187)
- Modify: `artificial_society/simulation.py` (`spawn_child_from_parent`, Zeile 201–202)
- Test: `tests/agents/test_genetics_strength.py` (NEU)

**Interfaces:**
- Consumes: `Body` (validiert `strength ∈ [0,1]`), `attach_body`, `clamp` (genetics).
- Produces: `GENE_RANGES["strength"] = (0.1, 0.9)`; `STRENGTH_MUTATION_SIGMA = 0.012`; `ensure_strength_gene(genes) -> None` (zieht uniform, idempotent); `inherit_strength(child_genes, parent_a, parent_b=None) -> None` (fitness-gewichtetes Mittel + `gauss(0, 0.012)`, geklemmt). **In simulation.py MUSS der Import aliased sein** (`inherit_strength as inherit_strength_gene` — Namenskollision mit der Lokalvariable in `spawn_child_from_parent`, Review F1, s. Step 4 (d)). Spätere Tasks (attach_body-Umbau in Task 14) behalten diese Signaturen.

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/agents/test_genetics_strength.py`:

```python
"""C5: strength-Gen (16.) — v2-only-Draws, v1-RNG-Strom bleibt unberührt."""

from __future__ import annotations

import random
from types import SimpleNamespace

from artificial_society.agents.agent import Agent, attach_body
from artificial_society.agents.genetics import (
    GENE_RANGES,
    STRENGTH_MUTATION_SIGMA,
    ensure_strength_gene,
    inherit_genes,
    inherit_strength,
    random_genes,
)


def test_gene_ranges_hat_strength_als_16_gen():
    assert GENE_RANGES["strength"] == (0.1, 0.9)
    assert len(GENE_RANGES) == 16
    assert STRENGTH_MUTATION_SIGMA == 0.012  # 0.012-Klasse (Spec C5)


def test_random_genes_zieht_kein_strength():
    """v1-RNG-Strom byte-identisch: random_genes darf den neuen Key NICHT ziehen."""
    random.seed(5)
    genes = random_genes()
    assert "strength" not in genes
    assert len(genes) == 15


def test_inherit_genes_ueberspringt_strength_auch_bei_v2_eltern():
    """Der GENE_RANGES-Loop in inherit_genes darf für strength keinen gauss-Draw
    machen (v1-Strom!) und keinen Key erzeugen — Vererbung läuft über inherit_strength."""
    random.seed(6)
    parent = SimpleNamespace(genes={**random_genes(), "strength": 0.7}, learning_score=1.0)
    child = inherit_genes(parent, parent)
    assert "strength" not in child


def test_ensure_strength_gene_zieht_einmal_und_ist_idempotent():
    random.seed(7)
    genes = {}
    ensure_strength_gene(genes)
    erster = genes["strength"]
    assert 0.1 <= erster <= 0.9
    ensure_strength_gene(genes)  # idempotent: kein zweiter Draw
    assert genes["strength"] == erster


def test_inherit_strength_fitness_gewichtet_mit_sigma_0012():
    random.seed(8)
    a = SimpleNamespace(genes={"strength": 0.3}, learning_score=1.0)
    b = SimpleNamespace(genes={"strength": 0.7}, learning_score=1.0)
    kinder = []
    for _ in range(200):
        child = {}
        inherit_strength(child, a, b)
        kinder.append(child["strength"])
    mittel = sum(kinder) / len(kinder)
    streuung = (sum((k - mittel) ** 2 for k in kinder) / len(kinder)) ** 0.5
    assert abs(mittel - 0.5) < 0.01  # gleiche Fitness ⇒ Mittelwert der Eltern
    assert 0.006 < streuung < 0.020  # σ ≈ 0.012, NICHT 0.25-Klasse


def test_attach_body_speist_body_aus_dem_gen():
    random.seed(9)
    agent = Agent.spawn_random(0, 0)
    assert "strength" not in agent.genes  # v1-Spawn zieht es nicht
    attach_body(agent)
    assert "strength" in agent.genes
    assert agent.body.strength == agent.genes["strength"]
    assert 0.1 <= agent.body.strength <= 0.9


def test_spawn_child_from_parent_v2_laeuft_ohne_namenskollision():
    """Regression (Review F1): spawn_child_from_parent hat eine LOKALE Variable
    `inherit_strength` (Gewichts-Vererbungsstärke für inherit_weights_from) —
    der genetics-Import MUSS aliased sein (inherit_strength_gene), sonst
    UnboundLocalError beim ersten v2-Kind-Spawn (latent bis Task 14)."""
    from artificial_society.simulation import Simulation

    sim = Simulation(
        seed=11, physics_v2=True, headless=True, load_checkpoint=False,
        grid_w=20, grid_h=15, initial_population=8,
    )
    eltern = sim.agents[0]
    kind = sim.spawn_child_from_parent(eltern, dict(eltern.genes))
    assert "strength" in kind.genes
    assert kind.body is not None
    assert kind.body.strength == kind.genes["strength"]
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_genetics_strength.py -q`
Expected: `ImportError: cannot import name 'STRENGTH_MUTATION_SIGMA'`

- [ ] **Step 3: Implementierung genetics.py**

In `artificial_society/agents/genetics.py`:

(a) `GENE_RANGES` um den 16. Eintrag erweitern (nach `'social_bandwidth'`):

```python
    'social_bandwidth':     (0.0, 1.0),
    # 16. Gen (Plan 3b, Spec C5): relative Körperkraft, speist Body(strength=...).
    # ACHTUNG Golden: Draw/Vererbung laufen NUR über ensure_strength_gene /
    # inherit_strength (v2-Pfad) — random_genes und der inherit_genes-Loop
    # lassen strength aus, sonst verschöbe ein zusätzlicher RNG-Draw den
    # v1-Strom und damit die Golden-Trajectory.
    'strength':             (0.1, 0.9),
```

(b) Unter `MUTATION_FITNESS_BIAS` ergänzen:

```python
STRENGTH_MUTATION_SIGMA = 0.012  # 0.012-Klasse (Spec C5) — NICHT 0.25 (~25 % der Range/Generation)
```

(c) In `inherit_genes` als ERSTE Zeile des `for k, (lo, hi) in GENE_RANGES.items():`-Loops:

```python
    for k, (lo, hi) in GENE_RANGES.items():
        if k == 'strength':
            continue  # v2-only (inherit_strength); ein gauss-Draw hier würde den v1-RNG-Strom verschieben
        val_a = parent_a.genes[k]
```

(d) Am Datei-Ende zwei neue Funktionen:

```python
def ensure_strength_gene(genes: dict) -> None:
    """Zieht das strength-Gen (NUR im v2-Pfad aufrufen: attach_body).

    random_genes lässt strength bewusst aus — ein zusätzlicher Draw dort würde
    den v1-RNG-Strom und damit die Golden-Trajectory verschieben. Idempotent.
    """
    if 'strength' not in genes:
        genes['strength'] = random.uniform(*GENE_RANGES['strength'])


def inherit_strength(child_genes: dict, parent_a, parent_b=None) -> None:
    """Vererbung des strength-Gens (NUR im v2-Kind-Pfad aufrufen).

    Fitness-gewichtetes Mittel wie inherit_genes, Mutations-σ 0.012-Klasse
    (Spec C5). Eltern ohne Gen (Alt-Checkpoints) zählen als 0.5 (3a-Default).
    """
    parent_b = parent_b or parent_a
    lo, hi = GENE_RANGES['strength']
    val_a = parent_a.genes.get('strength', 0.5)
    val_b = parent_b.genes.get('strength', 0.5)
    score_a = max(0.01, getattr(parent_a, 'learning_score', 1.0))
    score_b = max(0.01, getattr(parent_b, 'learning_score', 1.0))
    w_a = score_a / (score_a + score_b)
    base = w_a * val_a + (1.0 - w_a) * val_b
    child_genes['strength'] = clamp(base + random.gauss(0, STRENGTH_MUTATION_SIGMA), lo, hi)
```

- [ ] **Step 4: Implementierung agent.py + simulation.py**

In `artificial_society/agents/agent.py`:

(a) Import ergänzen (Zeile 13, bestehende genetics-Import-Zeile ersetzen):

```python
from artificial_society.agents.genetics import ensure_strength_gene, inherit_genes, random_genes
```

(b) `attach_body` (Zeilen 192–200) ersetzen durch:

```python
def attach_body(agent) -> None:
    """Physik-v2-Embodiment: Body + Hände; Kraft aus dem strength-Gen (Plan 3b).

    ensure_strength_gene zieht RNG NUR hier (v2-Pfad) — nie in random_genes
    (Golden). Die Körpermasse ist real geankert (BODY_MASS_DEFAULT_KG).
    """
    agent.physics_v2 = True
    ensure_strength_gene(agent.genes)
    agent.body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=agent.genes["strength"])
    agent.hands = Hands()
```

(c) In `ensure_fields` die Body-Rekonstruktions-Zeile (Zeile 186–187) ersetzen:

```python
    if agent.physics_v2 and agent.body is None:
        agent.body = Body(
            body_mass=BODY_MASS_DEFAULT_KG, strength=agent.genes.get("strength", 0.5)
        )
```

In `artificial_society/simulation.py`:

(d) Import ergänzen (nach dem bestehenden `agents.agent`-Import-Block) — **zwingend ALIASED (Review F1):** `spawn_child_from_parent` hat bereits eine LOKALE Variable `inherit_strength` (Zeilen 203–205: die Gewichts-Vererbungsstärke für `inherit_weights_from`). Ein unaliasierter Import + Aufruf in derselben Funktion VOR dieser Zuweisung macht den Namen funktionsweit lokal ⇒ `UnboundLocalError` beim ersten v2-Kind-Spawn (latent bis Task 14):

```python
from artificial_society.agents.genetics import inherit_strength as inherit_strength_gene
```

(e) In `spawn_child_from_parent` den Block `if self.physics_v2: attach_body(child)` (Zeilen 201–202) ersetzen:

```python
        if self.physics_v2:
            # strength wird über den eigenen v2-Pfad vererbt (inherit_genes
            # überspringt es — Golden), DANN baut attach_body den Body daraus.
            # ALIAS beachten: `inherit_strength` ist in dieser Funktion bereits
            # die lokale Gewichts-Vererbungsstärke — daher inherit_strength_gene.
            inherit_strength_gene(child.genes, parent, other_parent)
            attach_body(child)
```

- [ ] **Step 5: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_genetics_strength.py tests/test_physics_v2_embodiment.py tests/test_regression_golden.py tests/test_headless.py -q`
Expected: alle grün (`7 passed` in der neuen Datei; Golden/Digest UNVERÄNDERT grün — das ist der eigentliche Beweis).

- [ ] **Step 6: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 248)
../venv/bin/python -m ruff check --fix artificial_society/agents/genetics.py artificial_society/agents/agent.py artificial_society/simulation.py tests/agents/test_genetics_strength.py
../venv/bin/python -m ruff format artificial_society/agents/genetics.py artificial_society/agents/agent.py artificial_society/simulation.py tests/agents/test_genetics_strength.py
git add artificial_society/agents/genetics.py artificial_society/agents/agent.py artificial_society/simulation.py tests/agents/test_genetics_strength.py
git commit -m "feat(3b): strength-Gen (16.) — v2-only Draw/Vererbung, speist Body statt Default 0.5"
```

---

### Task 5: `ActionResult.remainder` (additive 3a-API-Ergänzung)

Das Causal Model (C4) prädiziert „dasselbe Objekt im nächsten Tick". Bei `do_cut` wird das Ziel physisch durch den `remainder` ersetzt (neue `PhysObject`-Instanz) — ohne Referenz darauf wäre jedes geschnittene Objekt fürs Causal-Target verloren. `ActionResult` bekommt ein additives Feld (Default `None` ⇒ alle 3a-Aufrufer unverändert).

**Files:**
- Modify: `artificial_society/environment/physics/actions.py` (`ActionResult`, `do_cut`)
- Test: `tests/environment/physics/test_actions.py` (erweitern, am Dateiende)

**Interfaces:**
- Consumes: bestehende `do_cut`/`CutResult`-Mechanik.
- Produces: `ActionResult.remainder: PhysObject | None` — bei erfolgreichem cut die physische Fortsetzung des Ziels; Task 14 konsumiert es für die Causal-Target-Fortschreibung.

- [ ] **Step 1: Failing Test schreiben**

Ans Ende von `tests/environment/physics/test_actions.py` anhängen (Importe der Datei enthalten bereits `make_object`, `Body`, `Hands`, `do_cut` — falls nicht, lokal importieren):

```python
def test_cut_result_traegt_remainder_referenz():
    """3b (C4): der remainder ist die physische Fortsetzung des Ziels — das
    Causal-Target folgt ihm. Masse bleibt exakt erhalten."""
    import random as _random

    from artificial_society.environment.phys_objects import ObjectLayer
    from artificial_society.environment.physics.actions import do_cut, do_grasp, do_strike
    from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG, Body, Hands
    from artificial_society.environment.physics.objects import make_object
    from artificial_society.environment.physics.props import IDX2

    rng = _random.Random(99)
    layer = ObjectLayer(8, 8, rng=rng)
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
    klinge = max(schlag.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))
    hands.release(hammer)
    layer.add(hammer, (4, 4))
    do_grasp(body, hands, layer, (4, 4), klinge)

    schnitt = do_cut(body, hands, layer, (4, 4), klinge, kadaver, 0.8)

    assert schnitt.extracted is not None
    assert schnitt.remainder is not None
    assert schnitt.remainder is not kadaver  # neue Instanz — deshalb braucht 3b die Referenz
    assert schnitt.extracted.mass + schnitt.remainder.mass == kadaver.mass
    assert layer.position_of(schnitt.remainder) == (4, 4)
    # Fehlschlag-/No-yield-Pfade tragen remainder=None (Default)
    granit = make_object("granite", 2.0)
    layer.add(granit, (4, 4), source="spawned")
    kein_schnitt = do_cut(body, hands, layer, (4, 4), klinge, granit, 0.8)
    assert kein_schnitt.remainder is None
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_actions.py -q -k remainder`
Expected: FAIL mit `AttributeError: 'ActionResult' object has no attribute 'remainder'` (oder assert `None is not None`).

- [ ] **Step 3: Implementierung**

In `artificial_society/environment/physics/actions.py`:

(a) `ActionResult` (Zeilen 144–155) um ein Feld erweitern — nach `extracted`:

```python
    extracted: PhysObject | None = None
    remainder: PhysObject | None = None  # cut: physische Fortsetzung des Ziels (3b, Causal-Target C4)
```

(b) Im Erfolgs-Return von `do_cut` (letzte Zeile der Funktion, Zeile 335–337) `remainder=remainder` ergänzen:

```python
    return ActionResult(
        ok=True,
        verb="cut",
        energy_delta_sim=energy_delta,
        extracted=extracted,
        remainder=remainder,
        work_j=work_j,
    )
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/environment/physics/test_actions.py tests/environment/test_conservation_fuzz.py -q`
Expected: alle grün.

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 249)
../venv/bin/python -m ruff check --fix artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
../venv/bin/python -m ruff format artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
git add artificial_society/environment/physics/actions.py tests/environment/physics/test_actions.py
git commit -m "feat(3b): ActionResult.remainder — Causal-Target folgt der physischen Fortsetzung beim cut"
```

---

### Task 6: `perception_v2.py` — Slots, Maskierung, zulässige Mengen

Die Objekt-Wahrnehmung (C1): 10 Slots × 17 Features, Maske für leere Slots, plus die verb-spezifischen Zulässigkeits-Masken (C2). Reines NumPy, kein torch, kein Gehirn-Wissen.

**Files:**
- Create: `artificial_society/agents/perception_v2.py`
- Test: `tests/agents/test_perception_v2.py` (NEU)

**Interfaces:**
- Consumes: `ObjectLayer.objects_near(pos, radius)` (liefert `(obj, (x,y))`-Paare, insertion-ordered), `agent.pos`, `agent.hands.held`, `PhysObject.props/mass`, `N_PROPS_V2 = 13`.
- Produces (von Tasks 7–15 konsumiert, Signaturen FIX):
  - Konstanten: `K_GROUND_SLOTS = 8`, `N_HAND_SLOTS = 2`, `N_SLOTS = 10`, `SLOT_FEATS = 17`, `PERCEPTION_RADIUS = 4`, `SLOT_MASS_NORM_KG = 25.0`
  - `SlotView` (dataclass): `feats (np.float32 (10,17))`, `mask (np.bool_ (10,))`, `objs (list[PhysObject|None] len 10)`, `dists (np.int32 (10,))`, `at_own_pos (np.bool_ (10,))`, `held_flags (np.bool_ (10,))`
  - `build_slots(agent, layer) -> SlotView`
  - `admissible_masks(view) -> dict[str, np.ndarray]` mit Keys `"grasp"`, `"held"`, `"target"`

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/agents/test_perception_v2.py`:

```python
"""C1: Objekt-Slots — Features, Maskierung, nächste-8-Auswahl, zulässige Mengen (C2)."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from artificial_society.agents.perception_v2 import (
    K_GROUND_SLOTS,
    N_SLOTS,
    PERCEPTION_RADIUS,
    SLOT_FEATS,
    SLOT_MASS_NORM_KG,
    admissible_masks,
    build_slots,
)
from artificial_society.environment.phys_objects import ObjectLayer
from artificial_society.environment.physics.body import Hands
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.props import IDX2


def _agent(pos=(5, 5), held=()):
    hands = Hands()
    hands.held.extend(held)
    return SimpleNamespace(pos=pos, hands=hands)


def _layer():
    return ObjectLayer(12, 12, rng=random.Random(3))


def test_konstanten_aus_der_spec():
    assert (K_GROUND_SLOTS, N_SLOTS, SLOT_FEATS, PERCEPTION_RADIUS) == (8, 10, 17, 4)
    assert SLOT_MASS_NORM_KG == 25.0


def test_slot_features_layout_boden():
    layer = _layer()
    stein = make_object("granite", 50.0)  # > 25 kg → mass-Feature gekappt bei 1
    layer.add(stein, (7, 4), source="spawned")  # dx=+2, dy=−1
    view = build_slots(_agent(), layer)
    assert view.mask[0] and view.objs[0] is stein
    f = view.feats[0]
    np.testing.assert_allclose(f[:13], stein.props)
    assert f[13] == 1.0  # min(mass/25, 1)
    assert f[14] == 2 / 4 and f[15] == -1 / 4  # dx/4, dy/4
    assert f[16] == 0.0  # held-Flag
    assert view.dists[0] == 2 and not view.at_own_pos[0] and not view.held_flags[0]


def test_hand_slots_8_und_9():
    fleisch = make_object("raw_meat", 1.0)
    klinge = make_object("flint", 0.3)
    view = build_slots(_agent(held=(fleisch, klinge)), _layer())
    assert view.mask[8] and view.mask[9]
    assert view.objs[8] is fleisch and view.objs[9] is klinge
    assert view.feats[8][16] == 1.0 and view.feats[9][16] == 1.0  # held-Flag
    assert view.feats[8][14] == 0.0 and view.feats[8][15] == 0.0  # dx=dy=0
    assert view.held_flags[8] and view.held_flags[9]
    assert not view.mask[:8].any()


def test_leere_slots_sind_nullvektor_plus_maske():
    view = build_slots(_agent(), _layer())
    assert not view.mask.any()
    np.testing.assert_array_equal(view.feats, np.zeros((N_SLOTS, SLOT_FEATS), dtype=np.float32))
    assert view.objs == [None] * N_SLOTS


def test_naechste_8_nach_chebyshev_r4():
    layer = _layer()
    fern = make_object("granite", 1.0)
    layer.add(fern, (10, 10), source="spawned")  # Chebyshev 5 > 4 → unsichtbar
    for i in range(9):  # 9 Objekte in Reichweite → nur die nächsten 8
        layer.add(make_object("granite", 1.0), (5 + min(i, 4), 5), source="spawned")
    view = build_slots(_agent(), layer)
    assert view.mask[:8].all()
    assert fern not in view.objs
    assert sorted(view.dists[:8].tolist()) == view.dists[:8].tolist()  # nach Distanz sortiert


def test_admissible_masks_c2():
    layer = _layer()
    hier = make_object("carcass", 25.0)
    nah = make_object("granite", 1.0)
    fern = make_object("granite", 1.0)
    layer.add(hier, (5, 5), source="from_carcass")  # eigene Position
    layer.add(nah, (6, 5), source="spawned")  # r=1
    layer.add(fern, (5, 8), source="spawned")  # r=3
    klinge = make_object("flint", 0.3)
    view = build_slots(_agent(held=(klinge,)), layer)
    masks = admissible_masks(view)
    i_hier, i_nah, i_fern = view.objs.index(hier), view.objs.index(nah), view.objs.index(fern)
    # grasp: Boden r ≤ 1
    assert masks["grasp"][i_hier] and masks["grasp"][i_nah] and not masks["grasp"][i_fern]
    assert not masks["grasp"][8]  # Gehaltenes ist kein grasp-Ziel
    # held: nur Hand-Slots
    assert masks["held"][8] and masks["held"][:8].sum() == 0
    # target (strike/cut/eat): Boden an eigener Position ∪ gehalten
    assert masks["target"][i_hier] and masks["target"][8]
    assert not masks["target"][i_nah] and not masks["target"][i_fern]
    # Nutzlast-Check: 13 Props der Slot-Features stimmen mit IDX2-Layout überein
    assert view.feats[i_hier][IDX2["nutrition"]] == hier.props[IDX2["nutrition"]]
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_perception_v2.py -q`
Expected: `ModuleNotFoundError: No module named 'artificial_society.agents.perception_v2'`

- [ ] **Step 3: Implementierung**

Neue Datei `artificial_society/agents/perception_v2.py`:

```python
"""Objekt-Wahrnehmung der Physik v2 (Plan 3b, Spec C1/C2).

10 Slots: die K=8 nächsten Boden-Objekte im Chebyshev-Radius 4 + 2 Hand-Slots.
17 Features je Slot: 13 Props + mass_kg/25 (gekappt bei 1; Anker 25-kg-Kadaver)
+ dx/4 + dy/4 (relativ, normiert) + held-Flag. Leere Slots = Null-Vektor + Maske.

Wahrnehmung ist reine Eigenschafts-Wahrnehmung: kein Objekt hat eine ID im
Gehirn — das id-basierte Slot-Tracking (Task 7) dient NUR dem Causal-Target
und dem Logging. Dieses Modul kennt weder torch noch das Gehirn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from artificial_society.environment.physics.props import N_PROPS_V2

K_GROUND_SLOTS = 8
N_HAND_SLOTS = 2
N_SLOTS = K_GROUND_SLOTS + N_HAND_SLOTS  # 10
SLOT_FEATS = 17  # 13 Props + mass + dx + dy + held
PERCEPTION_RADIUS = 4  # Chebyshev (Spec C1)
SLOT_MASS_NORM_KG = 25.0  # Anker: 25-kg-Kadaver (Spec C1)

_MASS_IDX = N_PROPS_V2  # 13
_DX_IDX = N_PROPS_V2 + 1  # 14
_DY_IDX = N_PROPS_V2 + 2  # 15
_HELD_IDX = N_PROPS_V2 + 3  # 16


@dataclass
class SlotView:
    """Eine Wahrnehmungs-Momentaufnahme; objs/dists/at_own_pos/held_flags sind
    Metadaten für Zulässigkeit + Tracking — das Gehirn sieht nur feats+mask."""

    feats: np.ndarray  # (10, 17) float32
    mask: np.ndarray  # (10,) bool — True = Slot belegt
    objs: list  # len 10; PhysObject | None
    dists: np.ndarray  # (10,) int32 — Chebyshev; Hand-Slots = 0
    at_own_pos: np.ndarray  # (10,) bool — Boden-Objekt an eigener Position
    held_flags: np.ndarray  # (10,) bool


def _slot_features(obj, dx: int, dy: int, held: bool) -> np.ndarray:
    f = np.zeros(SLOT_FEATS, dtype=np.float32)
    f[:N_PROPS_V2] = obj.props
    f[_MASS_IDX] = min(obj.mass / SLOT_MASS_NORM_KG, 1.0)
    f[_DX_IDX] = dx / PERCEPTION_RADIUS
    f[_DY_IDX] = dy / PERCEPTION_RADIUS
    f[_HELD_IDX] = 1.0 if held else 0.0
    return f


def build_slots(agent, layer) -> SlotView:
    """Slots 0..7: nächste Boden-Objekte (Chebyshev ≤ 4, stabil nach Distanz
    sortiert — objects_near ist insertion-ordered, sorted() ist stabil ⇒
    deterministisch). Slots 8..9: gehaltene Objekte."""
    feats = np.zeros((N_SLOTS, SLOT_FEATS), dtype=np.float32)
    mask = np.zeros(N_SLOTS, dtype=bool)
    objs: list = [None] * N_SLOTS
    dists = np.zeros(N_SLOTS, dtype=np.int32)
    at_own = np.zeros(N_SLOTS, dtype=bool)
    held_flags = np.zeros(N_SLOTS, dtype=bool)

    x, y = agent.pos
    near = layer.objects_near((x, y), PERCEPTION_RADIUS)
    near.sort(key=lambda op: max(abs(op[1][0] - x), abs(op[1][1] - y)))
    for i, (obj, (ox, oy)) in enumerate(near[:K_GROUND_SLOTS]):
        d = max(abs(ox - x), abs(oy - y))
        feats[i] = _slot_features(obj, ox - x, oy - y, held=False)
        mask[i] = True
        objs[i] = obj
        dists[i] = d
        at_own[i] = d == 0

    held = list(agent.hands.held) if getattr(agent, "hands", None) is not None else []
    for j, obj in enumerate(held[:N_HAND_SLOTS]):
        s = K_GROUND_SLOTS + j
        feats[s] = _slot_features(obj, 0, 0, held=True)
        mask[s] = True
        objs[s] = obj
        held_flags[s] = True

    return SlotView(
        feats=feats, mask=mask, objs=objs, dists=dists, at_own_pos=at_own, held_flags=held_flags
    )


def admissible_masks(view: SlotView) -> dict:
    """Zulässige Slots je Verb-Rolle (Spec C2):
    grasp  — Boden-Objekt im Chebyshev-Radius 1;
    held   — gehaltene Objekte (release-Ziel; Werkzeug für strike/cut);
    target — Boden an eigener Position ∪ gehalten (strike/cut/eat-Ziel)."""
    ground = view.mask & ~view.held_flags
    held = view.mask & view.held_flags
    return {
        "grasp": ground & (view.dists <= 1),
        "held": held,
        "target": (ground & view.at_own_pos) | held,
    }
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_perception_v2.py -q`
Expected: `6 passed`

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 255)
../venv/bin/python -m ruff check --fix artificial_society/agents/perception_v2.py tests/agents/test_perception_v2.py
../venv/bin/python -m ruff format artificial_society/agents/perception_v2.py tests/agents/test_perception_v2.py
git add artificial_society/agents/perception_v2.py tests/agents/test_perception_v2.py
git commit -m "feat(3b C1): perception_v2 — 10 Objekt-Slots, 17 Features, Maskierung, zulaessige Mengen"
```

---

### Task 7: Slot-Identitäts-Tracking + Count-Novelty (`NoveltyBuckets`)

Das id-basierte Tracking (`resolve_slot_of`) liefert dem Causal Model sein Target („dasselbe Objekt im nächsten Tick"); `NoveltyBuckets` ist die dritte Neugier-Quelle (C4.3): `1/√n` über Eigenschafts-Buckets `floor(p/0.25)`, pro Agent, LRU 4096 — direkt aus dem Archive-Prototyp (`_bucket_counts`/`info_gain`) übernommen (Ideen, nicht Code).

**Files:**
- Modify: `artificial_society/agents/perception_v2.py` (anhängen)
- Test: `tests/agents/test_perception_v2.py` (anhängen)

**Interfaces:**
- Consumes: `SlotView` aus Task 6.
- Produces (FIX):
  - `resolve_slot_of(view: SlotView, obj) -> int` — Slot-Index per `is`-Identität, `-1` wenn nicht sichtbar
  - `NOVELTY_BUCKET_WIDTH = 0.25`, `NOVELTY_LRU_CAP = 4096`
  - `NoveltyBuckets` mit `observe_view(view: SlotView) -> float` (max über die Slot-Novelties dieses Ticks; pro Tick zählt jeder distinkte Bucket-Key genau einmal) und `counts: OrderedDict` (picklebar)

- [ ] **Step 1: Failing Tests schreiben**

An `tests/agents/test_perception_v2.py` anhängen:

```python
import math
import pickle

from artificial_society.agents.perception_v2 import (
    NOVELTY_LRU_CAP,
    NoveltyBuckets,
    resolve_slot_of,
)


def test_resolve_slot_of_ist_id_basiert():
    layer = _layer()
    a = make_object("granite", 1.0)
    b = make_object("granite", 1.0)  # wertgleich, andere Identität
    layer.add(a, (5, 5), source="spawned")
    layer.add(b, (5, 6), source="spawned")
    view = build_slots(_agent(), layer)
    assert view.objs[resolve_slot_of(view, a)] is a
    assert view.objs[resolve_slot_of(view, b)] is b
    assert resolve_slot_of(view, make_object("granite", 1.0)) == -1  # nie gesehen


def test_novelty_faellt_mit_1_durch_wurzel_n():
    """D3 (c): Bucket-Novelty fällt mit 1/√n über wiederholte Wahrnehmung."""
    layer = _layer()
    layer.add(make_object("granite", 1.0), (5, 5), source="spawned")
    view = build_slots(_agent(), layer)
    buckets = NoveltyBuckets()
    werte = [buckets.observe_view(view) for _ in range(4)]
    assert werte == [1.0, 1.0 / math.sqrt(2), 1.0 / math.sqrt(3), 1.0 / math.sqrt(4)]


def test_novelty_neuer_eigenschaftspunkt_zahlt_wieder_voll():
    layer = _layer()
    layer.add(make_object("granite", 1.0), (5, 5), source="spawned")
    agent = _agent()
    buckets = NoveltyBuckets()
    buckets.observe_view(build_slots(agent, layer))
    layer.add(make_object("carcass", 25.0), (5, 5), source="from_carcass")
    assert buckets.observe_view(build_slots(agent, layer)) == 1.0  # neuer Bucket → 1/√1


def test_novelty_dedupe_innerhalb_eines_ticks():
    """Zwei wertgleiche Steine im selben Tick zählen den Bucket nur EINMAL."""
    layer = _layer()
    layer.add(make_object("granite", 1.0), (5, 5), source="spawned")
    layer.add(make_object("granite", 1.0), (5, 6), source="spawned")
    buckets = NoveltyBuckets()
    buckets.observe_view(build_slots(_agent(), layer))
    assert list(buckets.counts.values()) == [1]


def test_novelty_lru_kappt_bei_4096_und_ist_picklebar():
    buckets = NoveltyBuckets(cap=3)
    for i in range(5):
        buckets.counts[("k", i)] = 1
        buckets._enforce_cap()
    assert len(buckets.counts) == 3
    assert ("k", 0) not in buckets.counts  # ältester Key flog raus
    assert NOVELTY_LRU_CAP == 4096
    wieder = pickle.loads(pickle.dumps(buckets))
    assert dict(wieder.counts) == dict(buckets.counts)
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_perception_v2.py -q`
Expected: `ImportError: cannot import name 'NoveltyBuckets'` (6 alte PASS, Collection-Error für die Datei).

- [ ] **Step 3: Implementierung**

An `artificial_society/agents/perception_v2.py` anhängen (Import `math` + `OrderedDict` oben ergänzen: `import math` und `from collections import OrderedDict`):

```python
def resolve_slot_of(view: SlotView, obj) -> int:
    """Slot-Index eines Objekts per Identität (id-basiertes Tracking, Spec C1).
    NUR fürs Causal-Target/Logging — das Gehirn sieht nie Identitäten.
    -1 = aus der Wahrnehmung gefallen (Causal Model: maskiert, kein Loss)."""
    for i, o in enumerate(view.objs):
        if o is obj:
            return i
    return -1


NOVELTY_BUCKET_WIDTH = 0.25  # key = floor(p/0.25) über 13 Props (Spec C4.3)
NOVELTY_LRU_CAP = 4096


class NoveltyBuckets:
    """Count-based Novelty über Eigenschafts-Buckets, pro Agent (Spec C4.3).

    novelty = 1/√n(key) beim Wahrnehmen/Halten/Erzeugen; LRU-gekappt (4096
    Keys). Ideen (Bucket-Counts, 1/√n) aus archive/.../causal_model.py
    (_bucket_counts/info_gain); zustandsbasiert, keine designer-gewählten
    Events. Ein OrderedDict ist picklebar (Checkpoint-Verträglichkeit).
    """

    def __init__(self, cap: int = NOVELTY_LRU_CAP):
        self.cap = cap
        self.counts: OrderedDict = OrderedDict()

    def _key(self, props) -> tuple:
        return tuple(int(min(float(p), 1.0) // NOVELTY_BUCKET_WIDTH) for p in props)

    def _enforce_cap(self) -> None:
        while len(self.counts) > self.cap:
            self.counts.popitem(last=False)

    def observe_view(self, view: SlotView) -> float:
        """Novelty dieses Ticks: max über alle belegten Slots; jeder distinkte
        Bucket-Key zählt pro Tick genau einmal (zwei wertgleiche Steine sind
        EIN Punkt im Eigenschaftsraum)."""
        best = 0.0
        seen: set = set()
        for i in range(N_SLOTS):
            if not view.mask[i]:
                continue
            key = self._key(view.objs[i].props)
            if key in seen:
                continue
            seen.add(key)
            n = self.counts.get(key, 0) + 1
            self.counts[key] = n
            self.counts.move_to_end(key)
            self._enforce_cap()
            best = max(best, 1.0 / math.sqrt(n))
        return best
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_perception_v2.py -q`
Expected: `11 passed`

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 260)
../venv/bin/python -m ruff check --fix artificial_society/agents/perception_v2.py tests/agents/test_perception_v2.py
../venv/bin/python -m ruff format artificial_society/agents/perception_v2.py tests/agents/test_perception_v2.py
git add artificial_society/agents/perception_v2.py tests/agents/test_perception_v2.py
git commit -m "feat(3b C1/C4): Slot-Identitaets-Tracking + Count-Novelty (1/sqrt n, LRU 4096)"
```

### Task 8: Brain-v2-Skelett — Konstruktor-Flag, Attention-Set-Encoder, `forward_v2`

Kern der Architektur-Vorentscheidung: `Brain(physics_v2=True)` baut die v2-Module ZUSÄTZLICH und NACH den v1-Modulen (Init-RNG-Reihenfolge des v1-Pfads bleibt identisch). `forward_v2` implementiert C1: Slot-Embedding, 1-Head-Attention mit Query aus `h_prev`, attn-Pool ⊕ masked-Max-Pool, GRU-Input 192. All-Masked ⇒ `obj_ctx = 0`, kein NaN.

**Files:**
- Modify: `artificial_society/agents/brain.py`
- Test: `tests/agents/test_brain_v2_forms.py` (NEU)

**Interfaces:**
- Consumes: bestehende Brain-Module; `SLOT_FEATS`-Konvention aus Task 6 (Zahl 17 wird in brain.py als eigene Konstante geführt — brain.py importiert NICHT aus perception_v2).
- Produces (FIX, von Tasks 9–15 konsumiert):
  - Konstanten (Modul-Ebene, exakt diese Namen): `OBJ_SLOTS = 10`, `SLOT_FEATS_V2 = 17`, `SLOT_EMBED_DIM = 32`, `QUERY_DIM = 8`, `OBJ_CTX_DIM = 64`, `ACTION_SIZE_V2 = 29`, `V1_HEAD_DIMS = 7`, `RESEARCH_DRIVE_DIM = 6`, `VERB_SLICE = slice(7, 12)`, `VERBS_V2 = ("grasp", "release", "strike", "cut", "eat")`, `EFFORT_DIM = 12`, `TARGET_QUERY_SLICE = slice(13, 21)`, `TOOL_QUERY_SLICE = slice(21, 29)`, `VERB_THRESHOLD = 0.5`, `VERB_INIT_BIAS = 0.2`, `GAMMA_V2 = 0.99`, `PPO_EPOCHS_V2 = 4`, `N_MINIBATCHES_V2 = 4`, `MINIBATCH_SIZE_V2 = 32`, `KL_EARLY_STOP_V2 = 0.02`, `ENTROPY_COEF_NEW = 0.01`, `LOGSTD_FLOOR_NEW = -0.8`, `DEATH_REWARD_V2 = -3.0`, `OBS_TARGET_EXCLUDED = frozenset({26} | set(range(34, 57)))`, `OBS_TARGET_INCLUDED_IDX` (Tuple der 33 übrigen Indizes), `CURIO_TARGET_DIM = 203`, `CURIO_STAT_MOMENTUM = 0.01`
  - `Brain(input_size=57, hidden_size=96, action_size=7, plasticity=1.0, physics_v2=False)` — Attribut `self.physics_v2`; v2-Module: `slot_embed`, `attn_query`, `w_sel`, `next_slots_head`; Buffer `curio_err_mean (203,)`, `curio_err_var (203,)`
  - `Brain.forward_v2(obs (B,57), hidden (B,96), slot_feats (B,10,17), slot_mask (B,10) bool) -> (mean (B,29), std (B,29), value (B,), next_hidden (B,96), embeds (B,10,32), attn (B,10))`
  - intern: `Brain._clamped_logstd()`, `Brain._encode_slots(slot_feats, slot_mask, hidden) -> (obj_ctx, embeds, attn)`

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/agents/test_brain_v2_forms.py`:

```python
"""D3-Form-Tests: 29 kont. Dims, GRU-Input 192, v1-Formen unverändert, All-Masked ⇒ obj_ctx=0."""

from __future__ import annotations

import torch

from artificial_society.agents.brain import (
    ACTION_SIZE_V2,
    CURIO_TARGET_DIM,
    LOGSTD_FLOOR_NEW,
    OBS_TARGET_EXCLUDED,
    OBS_TARGET_INCLUDED_IDX,
    VERB_INIT_BIAS,
    Brain,
)


def _v2():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def test_v1_formen_unveraendert():
    """Golden-/Checkpoint-Garantie: das v1-Brain hat exakt die heutigen Module."""
    torch.manual_seed(0)
    brain = Brain()
    assert brain.physics_v2 is False
    assert brain.gru.input_size == 128 and brain.gru.hidden_size == 96
    assert brain.policy_mean.out_features == 7
    assert brain.policy_logstd.shape == (7,)
    assert brain.world_fc[0].in_features == 96 + 7
    for verboten in ("slot_embed", "attn_query", "w_sel", "next_slots_head"):
        assert not hasattr(brain, verboten), f"v1-Brain darf kein {verboten} haben"
    assert "curio_err_mean" not in dict(brain.named_buffers())


def test_v2_formen_nach_spec():
    brain = _v2()
    assert brain.physics_v2 is True
    assert brain.action_size == ACTION_SIZE_V2 == 29
    assert brain.gru.input_size == 192 and brain.gru.hidden_size == 96  # 128 + 64
    assert brain.policy_mean.out_features == 29
    assert brain.policy_logstd.shape == (29,)
    assert brain.slot_embed[0].in_features == 17 and brain.slot_embed[0].out_features == 32
    assert isinstance(brain.slot_embed[1], torch.nn.LayerNorm)
    assert brain.attn_query.in_features == 96 and brain.attn_query.out_features == 32
    assert brain.w_sel.in_features == 32 and brain.w_sel.out_features == 8
    assert brain.w_sel.bias is None
    assert brain.next_slots_head.out_features == 170  # 10 × 17
    assert brain.world_fc[0].in_features == 96 + 29
    assert brain.curio_err_mean.shape == (CURIO_TARGET_DIM,) == (203,)
    assert LOGSTD_FLOOR_NEW == -0.8


def test_verb_init_bias_und_logstd_floor():
    brain = _v2()
    assert torch.all(brain.policy_mean.bias[7:12] == VERB_INIT_BIAS)  # +0.2 Babbling-Prior
    with torch.no_grad():
        brain.policy_logstd.fill_(-5.0)
    ls = brain._clamped_logstd()
    assert torch.all(ls[:7] == -2.0)  # alte Gruppe: Floor −2.0 (wie heute)
    assert torch.all(ls[7:] == -0.8)  # neue Gruppe: Floor −0.8 (Spec C2)


def test_neugier_target_ausschluss_dims():
    """D3 Neugier (a): last_reward (26), Causal (34–36), Episodic (37–48), Hormone (49–56)."""
    assert OBS_TARGET_EXCLUDED == frozenset({26} | set(range(34, 57)))
    assert len(OBS_TARGET_INCLUDED_IDX) == 33
    assert 26 not in OBS_TARGET_INCLUDED_IDX
    assert all(d not in OBS_TARGET_INCLUDED_IDX for d in range(34, 57))


def test_forward_v2_formen_und_all_masked():
    brain = _v2()
    obs = torch.zeros(1, 57)
    hidden = torch.zeros(1, 96)
    feats = torch.zeros(1, 10, 17)
    mask = torch.zeros(1, 10, dtype=torch.bool)  # kein Objekt in Sicht, leere Hände
    mean, std, value, next_hidden, embeds, attn = brain.forward_v2(obs, hidden, feats, mask)
    assert mean.shape == (1, 29) and std.shape == (1, 29)
    assert next_hidden.shape == (1, 96) and embeds.shape == (1, 10, 32)
    for t in (mean, std, value, next_hidden, embeds, attn):
        assert not torch.isnan(t).any(), "All-Masked darf kein NaN erzeugen (Spec C1)"
    obj_ctx, _, attn0 = brain._encode_slots(feats, mask, hidden)
    assert torch.all(obj_ctx == 0.0), "All-Masked ⇒ obj_ctx = 0"
    assert torch.all(attn0 == 0.0)


def test_forward_v2_teilmaske_nutzt_nur_belegte_slots():
    brain = _v2()
    hidden = torch.zeros(1, 96)
    feats = torch.rand(1, 10, 17)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, 3] = True
    obj_ctx, embeds, attn = brain._encode_slots(feats, mask, hidden)
    assert attn[0, 3] == 1.0 and attn[0].sum() == 1.0  # einziger Slot trägt alles
    assert not torch.isnan(obj_ctx).any()
    # Max-Pool = Embedding des einzigen Slots
    assert torch.allclose(obj_ctx[0, 32:], embeds[0, 3])


def test_inherit_weights_ueberspringt_shape_mismatch_v1_v2():
    """C5: inherit_weights_from überspringt Mismatches still (gewollt beim
    Architektur-Wechsel) — v2-Kind von v1-Eltern crasht nicht."""
    torch.manual_seed(1)
    v1, v2 = Brain(), Brain(physics_v2=True)
    v2.inherit_weights_from(v1)  # darf nicht werfen
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_forms.py -q`
Expected: `ImportError: cannot import name 'ACTION_SIZE_V2'`

- [ ] **Step 3: Implementierung brain.py**

(a) Import `math` oben ergänzen (nach `import os`):

```python
import math
```

(b) Nach dem Block `RESEARCH_DRIVE_THRESHOLD = 0.4` (Zeile 89) die v2-Konstanten einfügen:

```python
# ---------------------------------------------------------------------------
# Physik v2 (Plan 3b, Spec C1–C4): Objekt-Slots, neue Köpfe, PPO-Anpassungen.
# Die v2-Netz-Teile existieren NUR im v2-Modus (Brain(physics_v2=True)) —
# v1-Brains bleiben form-identisch zu heute (Golden + alte Checkpoints).
# ---------------------------------------------------------------------------
OBJ_SLOTS = 10  # K=8 Boden (Chebyshev r≤4) + 2 Hand (Spec C1)
SLOT_FEATS_V2 = 17  # 13 Props + mass/25 + dx/4 + dy/4 + held
SLOT_EMBED_DIM = 32
QUERY_DIM = 8
OBJ_CTX_DIM = 2 * SLOT_EMBED_DIM  # attn-Pool ⊕ masked-Max-Pool = 64

ACTION_SIZE_V2 = 29  # 7 v1-Köpfe + 5 Verben + 1 Effort + 8 Target-Query + 8 Tool-Query
V1_HEAD_DIMS = 7
RESEARCH_DRIVE_DIM = 6  # v2: aus der log-prob-Summe ausgenommen (tote Kopplung, Spec C2)
VERB_SLICE = slice(7, 12)
VERBS_V2 = ("grasp", "release", "strike", "cut", "eat")
EFFORT_DIM = 12  # tanh-Output ∈ [−1,1] → effort01 = (a+1)/2
TARGET_QUERY_SLICE = slice(13, 21)
TOOL_QUERY_SLICE = slice(21, 29)
VERB_THRESHOLD = 0.5  # Aktivierung > 0.5 = „will"; höchstes Verb gewinnt
VERB_INIT_BIAS = 0.2  # optimistischer Init-Bias (Manipulations-Babbling-Prior, Spec C2)

GAMMA_V2 = 0.99  # Kredit-Horizont ~100 Ticks (Knapping→Kadaver→Schneiden→Essen)
PPO_EPOCHS_V2 = 4
N_MINIBATCHES_V2 = 4
MINIBATCH_SIZE_V2 = 32  # 4×32 aus dem 128er-Buffer
KL_EARLY_STOP_V2 = 0.02
ENTROPY_COEF_NEW = 0.01  # neue Kopf-Gruppe inkl. Kategorial (alte Gruppe: ENTROPY_COEF 0.004)
LOGSTD_FLOOR_NEW = -0.8  # log_std-Floor der neuen Köpfe
DEATH_REWARD_V2 = -3.0  # Terminal-Malus (C3), gemünzt in finalize_terminal

# Neugier-Target (C4.1): selbstbezügliche/nichtstationäre Obs-Dims raus —
# last_reward (26), Causal-Memory (34–36), Episodic-Retrieval (37–48), Hormone (49–56).
OBS_TARGET_EXCLUDED = frozenset({26} | set(range(34, 57)))
OBS_TARGET_INCLUDED_IDX = tuple(i for i in range(INPUT_SIZE) if i not in OBS_TARGET_EXCLUDED)
CURIO_TARGET_DIM = len(OBS_TARGET_INCLUDED_IDX) + OBJ_SLOTS * SLOT_FEATS_V2  # 33 + 170 = 203
CURIO_STAT_MOMENTUM = 0.01  # laufendes per-Dim Mean+Std (EMA)
```

(c) `Brain.__init__` ersetzen (Zeilen 112–165). WICHTIG: die v1-Modul-Konstruktion bleibt in EXAKT heutiger Reihenfolge; alles Neue kommt danach und nur unter dem Flag (Init-RNG-Reihenfolge des v1-Pfads identisch ⇒ Golden):

```python
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        action_size=ACTION_SIZE,
        plasticity: float = 1.0,
        physics_v2: bool = False,
    ):
        """
        plasticity-Gen (0.3..1.8) skaliert die Lernrate.
        Biologisches Vorbild: Neuronale Plastizitaet variiert zwischen Individuen.

        action_size = 7 (Emergenz v3):
          0: move_x
          1: move_y
          2: forage
          3: cooperate
          4: attack
          5: build
          6: research_drive  <-- NEU: Netz lernt selbst wann es forscht

        physics_v2 (Plan 3b): baut ZUSÄTZLICH (und NACH den v1-Modulen — die
        Init-RNG-Reihenfolge des v1-Pfads bleibt byte-identisch) den
        Attention-Set-Encoder und die neuen Köpfe; action_size wird 29,
        GRU-Input 128+64=192. v1-Brains bleiben form-identisch zu heute.
        """
        super().__init__()
        self.physics_v2 = bool(physics_v2)
        if self.physics_v2:
            action_size = ACTION_SIZE_V2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 160),
            nn.Tanh(),
            nn.Linear(160, 128),
            nn.Tanh(),
        )
        gru_input = 128 + (OBJ_CTX_DIM if self.physics_v2 else 0)
        self.gru = nn.GRUCell(gru_input, hidden_size)
        self.policy_mean = nn.Linear(hidden_size, action_size)
        self.policy_logstd = nn.Parameter(torch.full((action_size,), -0.45, dtype=torch.float32))
        self.value_head = nn.Linear(hidden_size, 1)
        self.world_fc = nn.Sequential(
            nn.Linear(hidden_size + action_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.next_obs_head = nn.Linear(128, input_size)
        self.reward_head = nn.Linear(128, 1)
        if self.physics_v2:
            # Set-Encoder (C1): geteilte Slot-Embeddings + 1-Head-Attention,
            # Query aus dem GRU-Zustand des VORTICKS.
            self.slot_embed = nn.Sequential(
                nn.Linear(SLOT_FEATS_V2, SLOT_EMBED_DIM), nn.LayerNorm(SLOT_EMBED_DIM)
            )
            self.attn_query = nn.Linear(hidden_size, SLOT_EMBED_DIM)
            # Slot-Auswahl (C2): Logits = query · W_sel(embed_i) / sqrt(8)
            self.w_sel = nn.Linear(SLOT_EMBED_DIM, QUERY_DIM, bias=False)
            # World-Model-Head über die rohen Slot-Features des nächsten Ticks (C4.1)
            self.next_slots_head = nn.Linear(128, OBJ_SLOTS * SLOT_FEATS_V2)
            with torch.no_grad():
                # Optimistischer Prior: +0.2 auf die Verb-Means (Spec C2) —
                # struktureller Prior, keine Belohnung.
                self.policy_mean.bias[VERB_SLICE] = VERB_INIT_BIAS
            # Laufendes per-Dim Mean+Std des Neugier-Vorhersagefehlers (C4.1)
            self.register_buffer("curio_err_mean", torch.zeros(CURIO_TARGET_DIM))
            self.register_buffer("curio_err_var", torch.ones(CURIO_TARGET_DIM))
        effective_lr = LEARNING_RATE * max(0.5, min(2.5, plasticity))
        self.optimizer = optim.Adam(self.parameters(), lr=effective_lr)
        self.rollout = RolloutBuffer()

        # --- NGU-style episodic novelty memory ---
        self.episodic_memory = EpisodicMemory(
            capacity=EPISODIC_MEMORY_CAPACITY,
            k=EPISODIC_K,
        )

        # Netz auf GPU verschieben
        self.to(device)
```

(d) In `inherit_weights_from` den `continue`-Zweig kommentieren (Zeile 180):

```python
                if child_param.shape != parent_param.shape:
                    # Spec C5: still überspringen ist beim Architektur-Wechsel
                    # v1→v2 GEWOLLT — nur form-gleiche Teile (encoder, value,
                    # v1-Anteile) werden vererbt, neue Module starten frisch.
                    continue
```

(e) Nach `forward` (hinter Zeile 217) die neuen v2-Methoden einfügen — `forward` selbst bleibt UNVERÄNDERT:

```python
    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): Attention-Set-Encoder + v2-Forward (Spec C1)
    # ------------------------------------------------------------------
    def _clamped_logstd(self):
        """v1: Clamp (−2.0, 0.7) wie heute. v2: neue Köpfe mit Floor −0.8 (C2)."""
        if not self.physics_v2:
            return self.policy_logstd.clamp(-2.0, 0.7)
        alt = self.policy_logstd[:V1_HEAD_DIMS].clamp(-2.0, 0.7)
        neu = self.policy_logstd[V1_HEAD_DIMS:].clamp(LOGSTD_FLOOR_NEW, 0.7)
        return torch.cat([alt, neu])

    def _encode_slots(self, slot_feats, slot_mask, hidden_tensor):
        """embed_i = LayerNorm(Linear(17→32)); q = Linear(96→32)(h_prev);
        attn = masked_softmax(q·K^T/√32); obj_ctx = attn-Pool ⊕ masked-Max-Pool (64).
        Alle Slots maskiert ⇒ obj_ctx = 0 und attn = 0 — kein NaN (Spec C1/D3)."""
        embeds = self.slot_embed(slot_feats)  # (B, 10, 32)
        any_slot = slot_mask.any(dim=-1, keepdim=True)  # (B, 1)
        q = self.attn_query(hidden_tensor)  # (B, 32)
        scores = torch.einsum("bd,bsd->bs", q, embeds) / math.sqrt(SLOT_EMBED_DIM)
        scores = scores.masked_fill(~slot_mask, -1e9)
        attn = torch.softmax(scores, dim=-1) * slot_mask.float()  # all-masked ⇒ exakt 0
        pooled = torch.einsum("bs,bsd->bd", attn, embeds)  # (B, 32)
        maxed = embeds.masked_fill(~slot_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        maxed = torch.where(any_slot, maxed, torch.zeros_like(maxed))
        pooled = torch.where(any_slot, pooled, torch.zeros_like(pooled))
        return torch.cat([pooled, maxed], dim=-1), embeds, attn

    def forward_v2(self, obs_tensor, hidden_tensor, slot_feats, slot_mask):
        """v2-Forward: GRU-Input = concat(encoder(obs_57), obj_ctx) (128+64=192).
        Die 57 v1-Obs-Dims bleiben unverändert (Spec C1)."""
        z = self.encoder(obs_tensor)
        obj_ctx, embeds, attn = self._encode_slots(slot_feats, slot_mask, hidden_tensor)
        next_hidden = self.gru(torch.cat([z, obj_ctx], dim=-1), hidden_tensor)
        mean = torch.tanh(self.policy_mean(next_hidden))
        log_std = self._clamped_logstd().unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        value = self.value_head(next_hidden).squeeze(-1)
        return mean, std, value, next_hidden, embeds, attn
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_forms.py tests/test_regression_golden.py tests/test_headless.py -q`
Expected: `test_brain_v2_forms.py: 8 passed`; Golden/Digest grün (v1-Konstruktion unverändert).

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 268)
../venv/bin/python -m ruff check --fix artificial_society/agents/brain.py tests/agents/test_brain_v2_forms.py
../venv/bin/python -m ruff format artificial_society/agents/brain.py tests/agents/test_brain_v2_forms.py
git add artificial_society/agents/brain.py tests/agents/test_brain_v2_forms.py
git commit -m "feat(3b C1): Brain-v2-Skelett — Konstruktor-Flag, Attention-Set-Encoder, forward_v2 (192er GRU)"
```

---

### Task 9: `act_v2` — Sampling, Verb-Auflösung, kategoriale Slot-Auswahl

Der v2-Aktionspfad (C2): kontinuierliches Sampling (rsample+tanh) mit `research_drive`-Ausschluss aus der log-prob, Verb-Auflösung (höchstes > 0.5), dann kategoriale Ziehung von Tool- und Target-Slot über `query·W_sel(embed)/√8`. Machbarkeits-Vorprüfung je Verb, damit entweder BEIDE Kategorial-Terme gezogen werden oder keiner (saubere log-prob-Struktur). Planner wird hier NIE aufgerufen (C5).

**Files:**
- Modify: `artificial_society/agents/brain.py`
- Test: `tests/agents/test_brain_v2_policy.py` (NEU)

**Interfaces:**
- Consumes: `forward_v2`, `_clamped_logstd` (Task 8); `admissible_masks`-Konvention aus Task 6 (dict mit `"grasp"`, `"held"`, `"target"` als np-bool (10,)).
- Produces (FIX):
  - `Brain._continuous_log_prob(mean, std, action_tensor) -> (log_prob (B,), dist)` — atanh-Transform exakt wie `evaluate_actions` (ohne +1e-8); v2: Dim 6 aus der Summe ausgenommen
  - `Brain._slot_logits(query (B,8), embeds (B,10,32), admissible (B,10) bool) -> (B,10)`
  - `Brain._categorical_terms(logits, idx (B,)) -> (chosen_log_prob (B,), entropy (B,))`
  - `Brain.act_v2(features (list 57), hidden_state (96,), slot_feats (np (10,17)), slot_mask (np (10,)), masks (dict)) -> dict` mit Keys: `obs_tensor (1,57)`, `hidden_in (1,96)`, `value`, `next_hidden (96,)`, `action_tensor (1,29)`, `action_list (29)`, `log_prob (1,)`, `entropy`, `slot_feats (1,10,17)`, `slot_mask (1,10)`, `verb (str|None)`, `effort (float ∈ [0,1])`, `target_idx (int, −1 = keiner)`, `tool_idx (int)`, `target_mask (1,10) bool`, `tool_mask (1,10) bool`, `slot_embeds (10,32)`, `attn (10,)`, `research_drive (float)`

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/agents/test_brain_v2_policy.py`:

```python
"""D3: Permutations-Invarianz, All-Masked-No-op, kategoriale Auswahl, log-prob-Struktur."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.brain import (
    RESEARCH_DRIVE_DIM,
    VERB_SLICE,
    VERBS_V2,
    Brain,
)


def _brain():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def _leere_masken():
    return {
        "grasp": np.zeros(10, dtype=bool),
        "held": np.zeros(10, dtype=bool),
        "target": np.zeros(10, dtype=bool),
    }


def _volle_szene():
    """2 Boden-Objekte an eigener Position (Slots 0, 1), 2 gehalten (8, 9)."""
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    for s in (0, 1, 8, 9):
        feats[s] = np.random.RandomState(s).rand(17).astype(np.float32)
        mask[s] = True
    feats[8][16] = feats[9][16] = 1.0
    masks = {
        "grasp": np.array([True, True] + [False] * 8),
        "held": np.array([False] * 8 + [True, True]),
        "target": np.array([True, True] + [False] * 6 + [True, True]),
    }
    return feats, mask, masks


def test_permutations_invarianz_vor_dem_sampling():
    """D3: Slot-Shuffle ⇒ identische Verteilungsparameter (μ, log_std,
    zurücksortierte attn) — Vergleich VOR dem Sampling (Sample-Vergleiche sind
    durch RNG-Konsum flaky)."""
    brain = _brain()
    hidden = torch.randn(1, 96)
    obs = torch.rand(1, 57)
    feats = torch.rand(1, 10, 17)
    mask = torch.ones(1, 10, dtype=torch.bool)
    perm = torch.tensor([3, 1, 4, 0, 7, 5, 2, 6, 8, 9])  # Boden-Slots permutiert
    m1, s1, v1, h1, _, a1 = brain.forward_v2(obs, hidden, feats, mask)
    m2, s2, v2, h2, _, a2 = brain.forward_v2(obs, hidden, feats[:, perm], mask[:, perm])
    assert torch.allclose(m1, m2, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)
    assert torch.allclose(h1, h2, atol=1e-5)
    assert torch.allclose(a1[:, perm], a2, atol=1e-5)  # attn zurücksortiert identisch


def test_all_masked_noop_ohne_nan_und_ohne_slot_logprob():
    brain = _brain()
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    for _ in range(20):  # mehrere Samples: Verben feuern oft (Init-Bias +0.2)
        step = brain.act_v2([0.0] * 57, brain.initial_hidden(), feats, mask, _leere_masken())
        assert step["target_idx"] == -1 and step["tool_idx"] == -1
        assert not step["target_mask"].any() and not step["tool_mask"].any()
        assert torch.isfinite(step["log_prob"]).all()
        assert not torch.isnan(step["action_tensor"]).any()
        # log_prob == reiner kontinuierlicher Anteil (kein Kategorial-Term)
        mean, std, _, _, _, _ = brain.forward_v2(
            step["obs_tensor"], step["hidden_in"], step["slot_feats"], step["slot_mask"]
        )
        lp_cont, _ = brain._continuous_log_prob(mean, std, step["action_tensor"])
        assert torch.equal(step["log_prob"], lp_cont)


def test_research_drive_fehlt_in_der_logprob_summe():
    """D3: Dim 6 ist im v2 totes Rauschen — nicht im PPO-Ratio."""
    brain = _brain()
    mean = torch.zeros(1, 29)
    std = torch.ones(1, 29)
    a = torch.zeros(1, 29)
    lp0, _ = brain._continuous_log_prob(mean, std, a)
    a2 = a.clone()
    a2[0, RESEARCH_DRIVE_DIM] = 0.9  # nur research_drive ändern
    lp1, _ = brain._continuous_log_prob(mean, std, a2)
    assert torch.equal(lp0, lp1)
    a3 = a.clone()
    a3[0, 0] = 0.9  # Kontrolle: andere Dim ändert die log-prob sehr wohl
    lp2, _ = brain._continuous_log_prob(mean, std, a3)
    assert not torch.equal(lp0, lp2)


def test_kategoriale_auswahl_zieht_nur_zulaessige_slots():
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(3)
    verbs_gesehen = set()
    for _ in range(300):
        step = brain.act_v2([0.1] * 57, brain.initial_hidden(), feats, mask, masks)
        if step["verb"] is None:
            assert step["target_idx"] == -1
            continue
        verbs_gesehen.add(step["verb"])
        assert step["target_idx"] >= 0
        assert bool(step["target_mask"][0, step["target_idx"]])  # nur zulässige Ziele
        if step["verb"] in ("strike", "cut"):
            assert step["tool_idx"] in (8, 9)  # Werkzeug nur aus der Hand
            assert step["target_idx"] != step["tool_idx"]  # Ziel ≠ Werkzeug
        else:
            assert step["tool_idx"] == -1
        if step["verb"] == "grasp":
            assert step["target_idx"] in (0, 1)
        if step["verb"] == "release":
            assert step["target_idx"] in (8, 9)
    assert {"grasp", "release", "strike", "cut", "eat"} & verbs_gesehen, (
        "mit Init-Bias +0.2 müssen Verben in 300 Samples feuern"
    )


def test_strike_ohne_gehaltenes_objekt_ist_noop_ohne_slot_terme():
    brain = _brain()
    feats, mask, _ = _volle_szene()
    mask[8] = mask[9] = False  # Hände leer
    feats[8] = feats[9] = 0.0
    masks = {
        "grasp": np.array([True, True] + [False] * 8),
        "held": np.zeros(10, dtype=bool),
        "target": np.array([True, True] + [False] * 8),
    }
    torch.manual_seed(4)
    for _ in range(200):
        step = brain.act_v2([0.1] * 57, brain.initial_hidden(), feats, mask, masks)
        if step["verb"] == "strike":
            raise AssertionError("strike ohne Schläger muss als No-op aufgelöst werden")
        if step["verb"] == "cut":
            assert step["tool_idx"] == -1  # bloße Hand: kein Tool-Term
            assert step["target_idx"] in (0, 1)


def test_verbs_v2_reihenfolge():
    assert VERBS_V2 == ("grasp", "release", "strike", "cut", "eat")
    assert VERB_SLICE == slice(7, 12)
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_policy.py -q`
Expected: FAIL — `AttributeError: 'Brain' object has no attribute 'act_v2'` (Permutations-Test kann schon grün sein, er nutzt nur forward_v2).

- [ ] **Step 3: Implementierung**

In `artificial_society/agents/brain.py` nach `forward_v2` einfügen:

```python
    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): Aktions-Sampling mit kategorialer Slot-Auswahl (C2)
    # ------------------------------------------------------------------
    def _continuous_log_prob(self, mean, std, action_tensor):
        """Gemeinsamer log-prob-Pfad für Sampling UND Retraining (bitgleiche
        Reproduktion, D3): identischer atanh-Transform wie evaluate_actions.
        v2: research_drive (Dim 6) ist aus der Summe ausgenommen (Spec C2)."""
        dist = torch.distributions.Normal(mean, std)
        clipped = torch.clamp(action_tensor, -0.999, 0.999)
        raw = 0.5 * torch.log((1 + clipped) / (1 - clipped))
        per_dim = dist.log_prob(raw) - torch.log(1 - clipped.pow(2) + 1e-6)
        if self.physics_v2:
            keep = torch.ones(per_dim.shape[-1], dtype=torch.bool, device=per_dim.device)
            keep[RESEARCH_DRIVE_DIM] = False
            per_dim = per_dim[..., keep]
        return per_dim.sum(dim=-1), dist

    def _slot_logits(self, query, embeds, admissible):
        """Auswahl-Logits = query · W_sel(embed_i) / √8 über zulässige Slots (C2)."""
        keys = self.w_sel(embeds)  # (B, 10, 8)
        logits = torch.einsum("bq,bsq->bs", query, keys) / math.sqrt(QUERY_DIM)
        return logits.masked_fill(~admissible, -1e9)

    @staticmethod
    def _categorical_terms(logits, idx):
        """(log-prob des gezogenen Slots, Entropie der Kategorial-Verteilung).
        Unzulässige Slots tragen p=0 exakt (softmax von −1e9 unterläuft zu 0)."""
        logp = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * logp).sum(dim=-1)
        chosen = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return chosen, entropy

    @staticmethod
    def _verb_feasible(verb, grasp_t, held_t, target_t) -> bool:
        """Machbarkeits-Vorprüfung je Verb: entweder werden BEIDE Kategorial-
        Terme gezogen oder keiner (leere zulässige Menge ⇒ No-op ohne
        log-prob-Anteil, Spec C2)."""
        n_held = int(held_t.sum())
        n_ground_atpos = int((target_t & ~held_t).sum())
        if verb == "grasp":
            return bool(grasp_t.any())
        if verb == "release":
            return n_held >= 1
        if verb == "eat":
            return bool(target_t.any())
        if verb == "strike":
            # Schläger MUSS aus der Hand kommen; Ziel = das andere Gehaltene
            # oder Boden an eigener Position.
            return n_held >= 1 and (n_ground_atpos >= 1 or n_held >= 2)
        if verb == "cut":
            if n_held >= 1:
                return n_ground_atpos >= 1 or n_held >= 2
            return n_ground_atpos >= 1  # bloße Hand
        return False

    def act_v2(self, features, hidden_state, slot_feats, slot_mask, masks):
        """v2-Aktionswahl (C2): Sampling, nie Argmax; Planner deaktiviert (C5).
        masks: dict aus perception_v2.admissible_masks ('grasp'/'held'/'target',
        np-bool (10,)). Rückgabe-Transitionen tragen Maske + Slot-Index, damit
        evaluate_actions_v2 exakt dieselbe Konditionierung reproduziert."""
        with torch.no_grad():
            obs = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            hidden = hidden_state.unsqueeze(0)
            feats = torch.as_tensor(slot_feats, dtype=torch.float32, device=device).unsqueeze(0)
            smask = torch.as_tensor(
                np.asarray(slot_mask), dtype=torch.bool, device=device
            ).unsqueeze(0)
            mean, std, value, next_hidden, embeds, attn = self.forward_v2(
                obs, hidden, feats, smask
            )

            dist = torch.distributions.Normal(mean, std)
            action = torch.tanh(dist.rsample())
            log_prob, _ = self._continuous_log_prob(mean, std, action)
            entropy = dist.entropy().sum(dim=-1)
            a = action.squeeze(0)

            grasp_t = torch.as_tensor(
                np.asarray(masks["grasp"]), dtype=torch.bool, device=device
            )
            held_t = torch.as_tensor(np.asarray(masks["held"]), dtype=torch.bool, device=device)
            target_t = torch.as_tensor(
                np.asarray(masks["target"]), dtype=torch.bool, device=device
            )

            verbs = a[VERB_SLICE]
            verb = None
            if float(verbs.max()) > VERB_THRESHOLD:
                kandidat = VERBS_V2[int(torch.argmax(verbs))]
                if self._verb_feasible(kandidat, grasp_t, held_t, target_t):
                    verb = kandidat
            effort = float((a[EFFORT_DIM] + 1.0) * 0.5)

            tool_idx = -1
            target_idx = -1
            tool_mask_used = torch.zeros_like(smask)
            target_mask_used = torch.zeros_like(smask)
            if verb in ("strike", "cut") and bool(held_t.any()):
                tool_mask_used = held_t.unsqueeze(0)
                logits = self._slot_logits(
                    a[TOOL_QUERY_SLICE].unsqueeze(0), embeds, tool_mask_used
                )
                tool_idx = int(torch.distributions.Categorical(logits=logits).sample())
                lp, _ent = self._categorical_terms(
                    logits, torch.tensor([tool_idx], device=device)
                )
                log_prob = log_prob + lp
            if verb is not None:
                base = {
                    "grasp": grasp_t,
                    "release": held_t,
                    "strike": target_t,
                    "cut": target_t,
                    "eat": target_t,
                }[verb].clone()
                if tool_idx >= 0:
                    base[tool_idx] = False  # Ziel ≠ Werkzeug
                target_mask_used = base.unsqueeze(0)
                logits = self._slot_logits(
                    a[TARGET_QUERY_SLICE].unsqueeze(0), embeds, target_mask_used
                )
                target_idx = int(torch.distributions.Categorical(logits=logits).sample())
                lp, _ent = self._categorical_terms(
                    logits, torch.tensor([target_idx], device=device)
                )
                log_prob = log_prob + lp

            action_list = a.detach().tolist()
            return {
                "obs_tensor": obs,
                "hidden_in": hidden.detach(),
                "value": value.detach(),
                "next_hidden": next_hidden.squeeze(0).detach(),
                "action_tensor": action.detach(),
                "action_list": action_list,
                "log_prob": log_prob.detach(),
                "entropy": entropy.detach(),
                "slot_feats": feats.detach(),
                "slot_mask": smask.detach(),
                "verb": verb,
                "effort": effort,
                "target_idx": target_idx,
                "tool_idx": tool_idx,
                "target_mask": target_mask_used.detach(),
                "tool_mask": tool_mask_used.detach(),
                "slot_embeds": embeds.squeeze(0).detach(),
                "attn": attn.squeeze(0).detach(),
                "research_drive": float(action_list[RESEARCH_DRIVE_DIM]),
            }
```

Zusätzlich oben in brain.py `import numpy as np` ergänzen (nach `import math`).

Hinweis zur Machbarkeits-Prüfung: `_verb_feasible` stellt sicher, dass nach Abzug des Tool-Slots die Ziel-Menge nie leer ist — die Struktur „Tool-Term gezogen, Target-Term unmöglich" kann nicht auftreten; jede Transition hat 0 oder (1 bzw. 2) Kategorial-Terme, exakt reproduzierbar über die gespeicherten Masken.

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_policy.py tests/agents/test_brain_v2_forms.py -q`
Expected: `14 passed` (6 + 8)

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 274)
../venv/bin/python -m ruff check --fix artificial_society/agents/brain.py tests/agents/test_brain_v2_policy.py
../venv/bin/python -m ruff format artificial_society/agents/brain.py tests/agents/test_brain_v2_policy.py
git add artificial_society/agents/brain.py tests/agents/test_brain_v2_policy.py
git commit -m "feat(3b C2): act_v2 — Verben, Effort, Queries, kategoriale Slot-Auswahl; research_drive raus aus log-prob"
```

### Task 10: `store_transition_v2` + `evaluate_actions_v2` + Retraining-Reproduktion

Die log-prob hat variable Struktur (Slot-Terme nur wenn ein Verb feuert und die Menge nicht leer ist) — jede Transition speichert die zulässigen Masken und die gezogenen Indizes, damit `evaluate_actions_v2` beim PPO-Retraining **bitgleich** dieselbe Konditionierung reproduziert (D3).

**Files:**
- Modify: `artificial_society/agents/brain.py`
- Test: `tests/agents/test_brain_v2_policy.py` (anhängen)

**Interfaces:**
- Consumes: `act_v2`-Rückgabe (Task 9), `_continuous_log_prob`/`_slot_logits`/`_categorical_terms`.
- Produces (FIX):
  - `Brain.store_transition_v2(brain_step, reward, done, next_obs, next_slot_feats (np (10,17)), next_slot_mask (np (10,)))` — Transition-Dict-Keys: `obs (57,)`, `hidden (96,)`, `action (29,)`, `log_prob ()`, `value ()`, `reward (float, geclampt ±6)`, `done (bool)`, `next_obs (57,)`, `slot_feats (10,17)`, `slot_mask (10,)`, `target_mask (10,)`, `tool_mask (10,)`, `target_idx (int)`, `tool_idx (int)`, `next_slot_feats (10,17)`, `next_slot_mask (10,)`
  - `Brain.evaluate_actions_v2(obs (B,57), hid (B,96), slot_feats (B,10,17), slot_mask (B,10), actions (B,29), target_mask (B,10), target_idx (B,) long, tool_mask (B,10), tool_idx (B,) long) -> (log_prob (B,), ent_old (B,), ent_new (B,), value (B,), next_hidden (B,96))` — `idx = −1` ⇒ kein Term für diese Zeile; `ent_new` enthält die Kategorial-Entropien (0.01-Gruppe)

- [ ] **Step 1: Failing Tests schreiben**

An `tests/agents/test_brain_v2_policy.py` anhängen:

```python
def _act_und_speichere(brain, feats, mask, masks, n=40):
    """n Ticks act_v2 + store_transition_v2 mit synthetischen next-Werten."""
    hidden = brain.initial_hidden()
    steps = []
    for _ in range(n):
        step = brain.act_v2([0.1] * 57, hidden, feats, mask, masks)
        hidden = step["next_hidden"]
        brain.store_transition_v2(step, 0.5, False, [0.2] * 57, feats, mask)
        steps.append(step)
    return steps


def test_transition_speichert_maske_und_slot_index():
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(5)
    _act_und_speichere(brain, feats, mask, masks, n=40)
    assert len(brain.rollout) == 40
    t = brain.rollout.storage[0]
    for key, shape in (
        ("obs", (57,)), ("hidden", (96,)), ("action", (29,)),
        ("slot_feats", (10, 17)), ("slot_mask", (10,)),
        ("target_mask", (10,)), ("tool_mask", (10,)),
        ("next_slot_feats", (10, 17)), ("next_slot_mask", (10,)),
    ):
        assert tuple(t[key].shape) == shape, key
    assert isinstance(t["target_idx"], int) and isinstance(t["tool_idx"], int)
    mit_slot = [t for t in brain.rollout.storage if t["target_idx"] >= 0]
    assert mit_slot, "in 40 Ticks muss mindestens ein Verb gefeuert haben (Init-Bias)"
    for t in mit_slot:
        assert bool(t["target_mask"][t["target_idx"]])


def test_retraining_reproduktion_bitgleiche_logprobs():
    """D3: evaluate_actions_v2 liefert mit Maske+Slot-Index BITGLEICHE log-probs
    wie zur Sampling-Zeit (identischer Code-Pfad, unveränderte Gewichte).

    Review F6: scheitert torch.equal NUR an Batch-Numerik (B=1 vs. B=40,
    max. Diff < 1e-7), ist der Degrade auf allclose(atol=1e-7, rtol=0.0) der
    NORMALE Ausgang — s. Robustheits-Hinweis im Plan, keine Debug-Schleife."""
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(6)
    _act_und_speichere(brain, feats, mask, masks, n=40)
    batch = brain.rollout.storage
    obs = torch.stack([t["obs"] for t in batch])
    hid = torch.stack([t["hidden"] for t in batch])
    sf = torch.stack([t["slot_feats"] for t in batch])
    sm = torch.stack([t["slot_mask"] for t in batch])
    act = torch.stack([t["action"] for t in batch])
    tm = torch.stack([t["target_mask"] for t in batch])
    om = torch.stack([t["tool_mask"] for t in batch])
    ti = torch.tensor([t["target_idx"] for t in batch], dtype=torch.long)
    oi = torch.tensor([t["tool_idx"] for t in batch], dtype=torch.long)
    alt = torch.stack([t["log_prob"] for t in batch])

    neu, ent_old, ent_new, value, next_hidden = brain.evaluate_actions_v2(
        obs, hid, sf, sm, act, tm, ti, om, oi
    )
    assert torch.equal(neu, alt), "Retraining-Reproduktion muss bitgleich sein (D3)"
    assert next_hidden.shape == (40, 96)
    # Kategorial-Entropie zählt zur NEUEN Gruppe: Zeilen mit Slot-Ziehung haben mehr ent_new
    mit = torch.tensor([t["target_idx"] >= 0 for t in batch])
    if mit.any() and (~mit).any():
        assert ent_new[mit].mean() > ent_new[~mit].mean()
    assert ent_old.shape == (40,) and torch.isfinite(ent_old).all()


def test_evaluate_v2_gradient_erreicht_w_sel_und_slot_embed():
    """C2 (a): echter Gradientenpfad in Query UND Slot-Embeddings."""
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(7)
    _act_und_speichere(brain, feats, mask, masks, n=40)
    batch = [t for t in brain.rollout.storage if t["target_idx"] >= 0]
    assert batch
    obs = torch.stack([t["obs"] for t in batch])
    hid = torch.stack([t["hidden"] for t in batch])
    sf = torch.stack([t["slot_feats"] for t in batch])
    sm = torch.stack([t["slot_mask"] for t in batch])
    act = torch.stack([t["action"] for t in batch])
    tm = torch.stack([t["target_mask"] for t in batch])
    om = torch.stack([t["tool_mask"] for t in batch])
    ti = torch.tensor([t["target_idx"] for t in batch], dtype=torch.long)
    oi = torch.tensor([t["tool_idx"] for t in batch], dtype=torch.long)
    lp, _, _, _, _ = brain.evaluate_actions_v2(obs, hid, sf, sm, act, tm, ti, om, oi)
    lp.sum().backward()
    assert brain.w_sel.weight.grad is not None and brain.w_sel.weight.grad.abs().sum() > 0
    assert brain.slot_embed[0].weight.grad is not None
    assert brain.slot_embed[0].weight.grad.abs().sum() > 0
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_policy.py -q`
Expected: neue Tests FAIL mit `AttributeError: 'Brain' object has no attribute 'store_transition_v2'`.

- [ ] **Step 3: Implementierung**

In `artificial_society/agents/brain.py` nach `act_v2` einfügen:

```python
    def store_transition_v2(
        self, brain_step, reward, done, next_obs, next_slot_feats, next_slot_mask
    ):
        """v2-Transition (C2): speichert zusätzlich Slot-Features, Slot-Maske,
        zulässige Masken und gezogene Indizes — evaluate_actions_v2 reproduziert
        damit exakt dieselbe log-prob-Konditionierung (variable Struktur)."""
        self.rollout.add(
            {
                "obs": brain_step["obs_tensor"].detach().squeeze(0),
                "hidden": brain_step["hidden_in"].detach().squeeze(0),
                "action": brain_step["action_tensor"].detach().squeeze(0),
                "log_prob": brain_step["log_prob"].detach().squeeze(0),
                "value": brain_step["value"].detach().squeeze(0),
                "reward": max(-REWARD_CLAMP, min(REWARD_CLAMP, reward)),
                "done": done,
                "next_obs": torch.tensor(next_obs, dtype=torch.float32, device=device),
                "slot_feats": brain_step["slot_feats"].detach().squeeze(0),
                "slot_mask": brain_step["slot_mask"].detach().squeeze(0),
                "target_mask": brain_step["target_mask"].detach().squeeze(0),
                "tool_mask": brain_step["tool_mask"].detach().squeeze(0),
                "target_idx": int(brain_step["target_idx"]),
                "tool_idx": int(brain_step["tool_idx"]),
                "next_slot_feats": torch.as_tensor(
                    np.asarray(next_slot_feats), dtype=torch.float32, device=device
                ),
                "next_slot_mask": torch.as_tensor(
                    np.asarray(next_slot_mask), dtype=torch.bool, device=device
                ),
            }
        )

    def evaluate_actions_v2(
        self, obs, hid, slot_feats, slot_mask, actions, target_mask, target_idx,
        tool_mask, tool_idx,
    ):
        """PPO-Retraining-Forward (C2): reproduziert die Sampling-log-prob
        bitgleich über denselben Code-Pfad (_continuous_log_prob, _slot_logits,
        _categorical_terms). idx = −1 ⇒ die Zeile hat keinen Slot-Term.
        Rückgabe: (log_prob, ent_alt (Dims 0..6), ent_neu (Dims 7..28 +
        Kategorial), value, next_hidden)."""
        mean, std, value, next_hidden, embeds, _ = self.forward_v2(
            obs, hid, slot_feats, slot_mask
        )
        log_prob, dist = self._continuous_log_prob(mean, std, actions)
        ent_per_dim = dist.entropy()
        ent_old = ent_per_dim[:, :V1_HEAD_DIMS].sum(dim=-1)
        ent_new = ent_per_dim[:, V1_HEAD_DIMS:].sum(dim=-1)

        for idx, mask_b, query_slice in (
            (tool_idx, tool_mask, TOOL_QUERY_SLICE),
            (target_idx, target_mask, TARGET_QUERY_SLICE),
        ):
            rows = idx >= 0
            if not bool(rows.any()):
                continue
            logits = self._slot_logits(
                actions[rows][:, query_slice], embeds[rows], mask_b[rows]
            )
            lp, ent = self._categorical_terms(logits, idx[rows])
            lp_full = torch.zeros_like(log_prob)
            lp_full[rows] = lp
            ent_full = torch.zeros_like(ent_new)
            ent_full[rows] = ent
            log_prob = log_prob + lp_full
            ent_new = ent_new + ent_full  # Kategorial-Entropie → 0.01-Gruppe (C2)
        return log_prob, ent_old, ent_new, value, next_hidden
```

Reihenfolge-Hinweis (bitgleiche Reproduktion): `act_v2` addiert erst den Tool-, dann den Target-Term — `evaluate_actions_v2` iteriert in derselben Reihenfolge `(tool, target)`. Float-Addition ist nicht assoziativ; die Reihenfolge ist Teil des Kontrakts.

Robustheits-Hinweis zum `torch.equal`-Assert (Review F6 — bitte VOR dem ersten Testlauf lesen): **Erwarte, dass exakte Bitgleichheit scheitert.** Sampling läuft mit B=1, Retraining mit B=32/40 — BLAS-Blocking der gebatchten Linear-Layer erzeugt routinemäßig 1-ulp-Differenzen. Scheitert `torch.equal` mit maximaler Differenz < 1e-7 bei korrekten Masken/Indizes, ist der definierte Degrade-Pfad der **NORMALE Ausgang**: den Assert auf `torch.allclose(neu, alt, atol=1e-7, rtol=0.0)` umstellen, Kommentar im Test (Verweis auf diesen Hinweis), Vermerk im Task-Report — **keine Debug-Schleife starten.** Der eigentliche Kontrakt ist die exakt reproduzierte KONDITIONIERUNG (Masken, Indizes, Dim-6-Ausschluss, Term-Reihenfolge): jede Abweichung > 1e-7 ist ein echter Konditionierungs-Bug und wird NICHT toleriert. (Der `torch.equal`-Assert im All-Masked-Test von Task 9 ist davon NICHT betroffen — dort vergleichen beide Seiten B=1 gegen B=1 im selben Pfad, das ist echt bitgleich.)

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_policy.py -q`
Expected: `9 passed`

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 277)
../venv/bin/python -m ruff check --fix artificial_society/agents/brain.py tests/agents/test_brain_v2_policy.py
../venv/bin/python -m ruff format artificial_society/agents/brain.py tests/agents/test_brain_v2_policy.py
git add artificial_society/agents/brain.py tests/agents/test_brain_v2_policy.py
git commit -m "feat(3b C2): store_transition_v2 + evaluate_actions_v2 — bitgleiche Retraining-Reproduktion"
```

---

### Task 11: `predict_world_v2` + `nextslot_err` (Neugier-Quelle 1)

Das v2-World-Model prädiziert next-obs (57, Loss/Fehler nur auf den 33 eingeschlossenen Dims) UND die rohen, maskierten Slot-Features (17×10) des nächsten Ticks — NICHT `obj_ctx` (das mit den eigenen Gewichten wandert und nie konvergiert). `nextslot_error` normalisiert per Dim mit laufendem Mean+Std und ersetzt den v1-Intrinsic-Pfad (dessen `rew_err = pred_reward.abs()` gar kein Fehlerterm war — ersatzlos gestrichen).

**Files:**
- Modify: `artificial_society/agents/brain.py`
- Test: `tests/agents/test_curiosity_v2.py` (NEU)

**Interfaces:**
- Consumes: `forward_v2`, Buffer `curio_err_mean/curio_err_var`, Konstanten aus Task 8; Transition-Keys aus Task 10.
- Produces (FIX):
  - `Brain.predict_world_v2(hidden (B,96), action (B,29)) -> (next_obs (B,57), next_slots (B,170), reward (B,))` — beide Prädiktions-Köpfe tanh-aktiviert
  - `Brain.nextslot_error(brain_step, next_obs (list 57), next_slot_feats (np (10,17)), next_slot_mask (np (10,))) -> float` — aktualisiert die Running-Stats (EMA, Momentum 0.01) und gibt `mean(z²)` über die gültigen Dims zurück; `z = (err − mean_d)/(std_d + 1e-6)`
  - `Brain._world_loss_v2(next_hidden, actions, next_obs, next_slot_feats, next_slot_mask, rewards) -> Tensor` (von Task 13 konsumiert): MSE auf den 33 eingeschlossenen Obs-Dims + maskierte MSE auf den Slot-Dims + Reward-Head-MSE

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/agents/test_curiosity_v2.py`:

```python
"""D3-Neugier: (a) selbstbezügliche Dims nicht im Target, per-Dim-Normalisierung, Maskierung."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.brain import (
    CURIO_TARGET_DIM,
    OBS_TARGET_INCLUDED_IDX,
    Brain,
)


def _brain():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def _step(brain):
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    masks = {k: np.zeros(10, dtype=bool) for k in ("grasp", "held", "target")}
    return brain.act_v2([0.1] * 57, brain.initial_hidden(), feats, mask, masks)


def test_predict_world_v2_formen():
    brain = _brain()
    h = torch.zeros(1, 96)
    a = torch.zeros(1, 29)
    next_obs, next_slots, reward = brain.predict_world_v2(h, a)
    assert next_obs.shape == (1, 57)
    assert next_slots.shape == (1, 170)  # 10 × 17 rohe Slot-Features
    assert reward.shape == (1,)


def test_selbstbezuegliche_dims_nicht_im_nextslot_target():
    """D3 (a): last_reward (26), Causal (34–36), Episodic (37–48), Hormone (49–56)
    ändern den Fehler NICHT — sie sind nachweislich nicht im Target."""
    brain = _brain()
    step = _step(brain)
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    basis = [0.3] * 57
    gestoert = list(basis)
    for dim in [26, *range(34, 57)]:
        gestoert[dim] = 0.9  # massive Störung NUR auf ausgeschlossenen Dims
    b1 = Brain(physics_v2=True)
    b1.load_state_dict(brain.state_dict())
    e1 = brain.nextslot_error(step, basis, feats, mask)
    e2 = b1.nextslot_error(step, gestoert, feats, mask)
    assert e1 == e2
    # Gegenprobe: eine EINGESCHLOSSENE Dim ändert den Fehler
    b2 = Brain(physics_v2=True)
    b2.load_state_dict(brain.state_dict())
    anders = list(basis)
    anders[OBS_TARGET_INCLUDED_IDX[0]] = 0.9
    assert b2.nextslot_error(step, anders, feats, mask) != e1


def test_maskierte_slots_zaehlen_nicht():
    """Nur belegte Slots tragen zum Slot-Anteil des Fehlers bei."""
    brain = _brain()
    step = _step(brain)
    leer = np.zeros((10, 17), dtype=np.float32)
    voll_aber_maskiert = np.full((10, 17), 0.9, dtype=np.float32)
    maske_aus = np.zeros(10, dtype=bool)
    b1 = Brain(physics_v2=True)
    b1.load_state_dict(brain.state_dict())
    e1 = brain.nextslot_error(step, [0.3] * 57, leer, maske_aus)
    e2 = b1.nextslot_error(step, [0.3] * 57, voll_aber_maskiert, maske_aus)
    assert e1 == e2  # maskierte Features sind unsichtbar


def test_running_stats_normalisieren_wiederholten_fehler():
    """Per-Dim Mean+Std: ein konstanter Fehler wird über die Zeit wegnormalisiert
    (z → 0), statt ewige Rausch-Rente zu zahlen."""
    brain = _brain()
    step = _step(brain)
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    werte = [brain.nextslot_error(step, [0.5] * 57, feats, mask) for _ in range(600)]
    assert werte[-1] < werte[0] * 0.2, "konstanter Fehler muss wegnormalisiert werden"
    assert brain.curio_err_mean.shape == (CURIO_TARGET_DIM,)
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_curiosity_v2.py -q`
Expected: FAIL — `AttributeError: 'Brain' object has no attribute 'predict_world_v2'`.

- [ ] **Step 3: Implementierung**

In `artificial_society/agents/brain.py` nach `evaluate_actions_v2` einfügen:

```python
    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): World-Model + nextslot_err (Neugier-Quelle 1, C4.1)
    # ------------------------------------------------------------------
    def predict_world_v2(self, hidden_tensor, action_tensor):
        """v2-World-Model: prädiziert next-obs (57) UND die rohen Slot-Features
        (10×17 = 170) des nächsten Ticks — NICHT obj_ctx (das mit den eigenen
        Gewichten wandert und nie konvergiert, Spec C4.1)."""
        hidden_tensor = _ensure_2d(hidden_tensor)
        action_tensor = _ensure_2d(action_tensor)
        z = self.world_fc(torch.cat([hidden_tensor, action_tensor], dim=-1))
        next_obs = torch.tanh(self.next_obs_head(z))
        next_slots = torch.tanh(self.next_slots_head(z))  # Features ∈ [−1,1] (dx/4, dy/4)
        reward = self.reward_head(z).squeeze(-1)
        return next_obs, next_slots, reward

    def _curio_target(self, next_obs_t, next_slot_feats_t, next_slot_mask_t):
        """(target (203,), valid (203,)): 33 eingeschlossene Obs-Dims ⊕ 170
        Slot-Dims; Slot-Dims sind nur gültig, wo der Slot belegt ist."""
        incl = torch.tensor(OBS_TARGET_INCLUDED_IDX, dtype=torch.long, device=device)
        target = torch.cat([next_obs_t[incl], next_slot_feats_t.reshape(-1)])
        valid = torch.cat(
            [
                torch.ones(len(OBS_TARGET_INCLUDED_IDX), dtype=torch.bool, device=device),
                next_slot_mask_t.unsqueeze(-1)
                .expand(OBJ_SLOTS, SLOT_FEATS_V2)
                .reshape(-1),
            ]
        )
        return target, valid

    def nextslot_error(self, brain_step, next_obs, next_slot_feats, next_slot_mask):
        """Neugier-Quelle 1 (C4.1): Vorhersagefehler auf den rohen, maskierten
        Slot-Features + den bereinigten Obs-Dims des nächsten Ticks.
        Normalisierung PER DIM mit laufendem Mean+Std (EMA, Momentum 0.01) —
        nicht global, nicht nur Std. Der bisherige rew_err-Term (v1,
        pred_reward.abs() — gar kein Fehlerterm) ist ersatzlos gestrichen."""
        with torch.no_grad():
            pred_obs, pred_slots, _ = self.predict_world_v2(
                brain_step["next_hidden"], brain_step["action_tensor"]
            )
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            feats_t = torch.as_tensor(
                np.asarray(next_slot_feats), dtype=torch.float32, device=device
            )
            mask_t = torch.as_tensor(
                np.asarray(next_slot_mask), dtype=torch.bool, device=device
            )
            target, valid = self._curio_target(next_obs_t, feats_t, mask_t)
            incl = torch.tensor(OBS_TARGET_INCLUDED_IDX, dtype=torch.long, device=device)
            pred = torch.cat([pred_obs.squeeze(0)[incl], pred_slots.squeeze(0)])
            err = pred - target
            m = CURIO_STAT_MOMENTUM
            new_mean = torch.where(
                valid, (1.0 - m) * self.curio_err_mean + m * err, self.curio_err_mean
            )
            new_var = torch.where(
                valid,
                (1.0 - m) * self.curio_err_var + m * (err - new_mean).pow(2),
                self.curio_err_var,
            )
            self.curio_err_mean.copy_(new_mean)
            self.curio_err_var.copy_(new_var)
            z = (err - new_mean) / (new_var.sqrt() + 1e-6)
            return float(z[valid].pow(2).mean())

    def _world_loss_v2(
        self, next_hidden, actions, next_obs, next_slot_feats, next_slot_mask, rewards
    ):
        """Trainings-Loss des v2-World-Models: MSE auf den 33 eingeschlossenen
        Obs-Dims + maskierte MSE auf den Slot-Dims + Reward-Head-MSE."""
        pred_obs, pred_slots, pred_rew = self.predict_world_v2(next_hidden, actions)
        incl = torch.tensor(OBS_TARGET_INCLUDED_IDX, dtype=torch.long, device=device)
        obs_loss = F.mse_loss(pred_obs[:, incl], next_obs[:, incl])
        b = next_slot_feats.shape[0]
        flach = next_slot_feats.reshape(b, -1)
        maske = (
            next_slot_mask.unsqueeze(-1)
            .expand(b, OBJ_SLOTS, SLOT_FEATS_V2)
            .reshape(b, -1)
            .float()
        )
        slot_loss = ((pred_slots - flach).pow(2) * maske).sum() / maske.sum().clamp(min=1.0)
        rew_loss = F.mse_loss(pred_rew.view(-1), rewards)
        return obs_loss + slot_loss + rew_loss
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_curiosity_v2.py -q`
Expected: `4 passed`

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 281)
../venv/bin/python -m ruff check --fix artificial_society/agents/brain.py tests/agents/test_curiosity_v2.py
../venv/bin/python -m ruff format artificial_society/agents/brain.py tests/agents/test_curiosity_v2.py
git add artificial_society/agents/brain.py tests/agents/test_curiosity_v2.py
git commit -m "feat(3b C4.1): predict_world_v2 + nextslot_err — Slot-Target, per-Dim-Normalisierung, rew_err gestrichen"
```

---

### Task 12: `systems/causal_model.py` — `CausalModelV2` (torch, NLL, Hinge-Epistemik)

Ehrliche **Neuimplementierung** in torch (der Archive-Prototyp ist numpy/12-Prop/8-Aktionen — strukturell nicht übernehmbar; übernommen werden seine IDEEN: `err_ema`, Bucket-Counts → letztere leben schon in Task 7). Eingabe `detach(gru_h 96) ⊕ Aktions-Vektor (22) ⊕ detach(Slot-Embed 32)`, Ausgabe (μ, log σ) über die 17 Slot-Features desselben Objekts im nächsten Tick. Der `detach` ist entschieden: das Causal Model liest die Repräsentation, es formt sie nicht (Repräsentations-Dynamik bleibt bei PPO). Epistemik = `max(0, mean(z²) − 1)` mit **Hinge AUSSEN** (ein Hinge innen würde dauerhaft ≈ 0.48 zahlen — skaleninvariante Rausch-Rente). Das ist KEIN registriertes Sim-System — ein Pro-Agent-Modul.

**Files:**
- Create: `artificial_society/systems/causal_model.py`
- Test: `tests/systems/test_causal_model_v2.py` (NEU)

**Interfaces:**
- Consumes: torch; nichts aus brain.py/perception_v2.py (nur Dimensions-Konventionen als lokale Konstanten — kein Import-Zyklus).
- Produces (FIX, von Tasks 14/15 konsumiert):
  - Konstanten: `CAUSAL_HIDDEN_IN = 96`, `CAUSAL_ACTION_DIMS = 22`, `CAUSAL_EMBED_DIMS = 32`, `CAUSAL_INPUT = 150`, `CAUSAL_TARGET = 17`, `CAUSAL_LR = 3e-4`, `LOG_SIGMA_FLOOR = -2.0`, `LOG_SIGMA_CEIL = 2.0`, `ERR_EMA_ALPHA = 0.1`
  - `CausalModelV2(plasticity: float = 1.0)` (nn.Module; eigener Adam mit `CAUSAL_LR × clamp(plasticity, 0.5, 2.5)`; Attribut `err_ema: float`)
  - `CausalModelV2.forward(x (B,150)) -> (mu (B,17), log_sigma (B,17))` — log σ geklemmt auf [−2, 2]
  - `CausalModelV2.observe(gru_h (96,), action22 (22,), slot_embed (32,), target17 (Tensor (17,))) -> dict` mit Keys `nll (float)`, `epistemic (float ≥ 0)`, `mean_abs_err (float)` — Epistemik VOR dem Update berechnet, dann ein NLL-SGD-Schritt, dann err_ema-Update

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/systems/test_causal_model_v2.py`:

```python
"""D3 Causal-Model: NLL sinkt deterministisch, σ wächst verrauscht, Hinge-Epistemik → 0."""

from __future__ import annotations

import torch

from artificial_society.systems.causal_model import (
    CAUSAL_INPUT,
    CAUSAL_LR,
    LOG_SIGMA_FLOOR,
    CausalModelV2,
)


def _fixe_eingabe():
    torch.manual_seed(0)
    return torch.randn(96), torch.rand(22) * 2 - 1, torch.randn(32)


def test_formen_und_sigma_floor():
    torch.manual_seed(0)
    model = CausalModelV2()
    assert CAUSAL_INPUT == 96 + 22 + 32 == 150
    mu, log_sigma = model.forward(torch.zeros(1, 150))
    assert mu.shape == (1, 17) and log_sigma.shape == (1, 17)
    assert torch.all(log_sigma >= LOG_SIGMA_FLOOR)  # log σ ≥ −2 (Spec C4)


def test_nll_sinkt_auf_deterministischer_sequenz():
    torch.manual_seed(1)
    model = CausalModelV2()
    h, a, e = _fixe_eingabe()
    ziel = torch.rand(17)
    erste = model.observe(h, a, e, ziel)["nll"]
    for _ in range(300):
        letzte = model.observe(h, a, e, ziel)["nll"]
    assert letzte < erste, "NLL muss auf deterministischer Sequenz sinken"


def test_sigma_waechst_auf_verrauschter_sequenz():
    torch.manual_seed(2)
    det = CausalModelV2()
    noisy = CausalModelV2()
    noisy.load_state_dict(det.state_dict())
    h, a, e = _fixe_eingabe()
    ziel = torch.full((17,), 0.5)
    gen = torch.Generator().manual_seed(7)
    for _ in range(400):
        det.observe(h, a, e, ziel)
        rausch = ziel + torch.randn(17, generator=gen) * 0.3
        noisy.observe(h, a, e, rausch)
    x = torch.cat([h, a, e]).unsqueeze(0)
    _, ls_det = det.forward(x)
    _, ls_noisy = noisy.forward(x)
    assert ls_noisy.mean() > ls_det.mean(), "σ muss auf verrauschtem Prozess wachsen"


def test_epistemik_geht_auf_aleatorischem_prozess_gegen_null():
    """D3 (b): auskonvergierter, rein aleatorischer Prozess zahlt gegen 0 —
    ewiges Steineschlagen als Neugier-Farm ist zu (Hinge AUSSEN)."""
    torch.manual_seed(3)
    model = CausalModelV2()
    h, a, e = _fixe_eingabe()
    gen = torch.Generator().manual_seed(11)
    epistemik = []
    for _ in range(2000):
        rausch = torch.full((17,), 0.5) + torch.randn(17, generator=gen) * 0.2
        epistemik.append(model.observe(h, a, e, rausch.clamp(0.0, 1.0))["epistemic"])
    frueh = sum(epistemik[:100]) / 100
    spaet = sum(epistemik[-200:]) / 200
    assert spaet < 0.25, f"konvergierte Epistemik muss ≈ 0 sein, ist {spaet}"
    assert spaet < frueh
    assert all(v >= 0.0 for v in epistemik)  # Hinge: nie negativ


def test_lr_skaliert_mit_plastizitaets_gen():
    torch.manual_seed(4)
    lr = CausalModelV2(plasticity=1.8).optimizer.param_groups[0]["lr"]
    assert lr == CAUSAL_LR * 1.8
    lr_geklemmt = CausalModelV2(plasticity=9.0).optimizer.param_groups[0]["lr"]
    assert lr_geklemmt == CAUSAL_LR * 2.5


def test_err_ema_wird_gepflegt():
    torch.manual_seed(5)
    model = CausalModelV2()
    h, a, e = _fixe_eingabe()
    assert model.err_ema == 1.0
    model.observe(h, a, e, torch.rand(17))
    assert model.err_ema != 1.0  # Logging-Signal (Archive-Idee)
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/systems/test_causal_model_v2.py -q`
Expected: `ModuleNotFoundError` beim Import — `systems/causal_model.py` existiert nicht (nur im `archive/`).

- [ ] **Step 3: Implementierung**

Neue Datei `artificial_society/systems/causal_model.py`:

```python
"""Causal Model der Physik v2 (Plan 3b, Spec C4) — torch-Neuimplementierung.

Verteilungs-Vorwärtsprädiktor über Objekt-Eigenschaften: aus
detach(GRU-Zustand 96) ⊕ verkörpertem Aktions-Vektor (22: Verben+Effort+Queries)
⊕ detach(Slot-Embedding 32 des gewählten Ziels) werden (μ, log σ) über die 17
Slot-Features DESSELBEN Objekts im nächsten Tick prädiziert (Identitäts-Tracking
aus perception_v2; bei strike-Zerstörung: massereichstes Fragment; aus der
Wahrnehmung gefallen: maskiert, kein Loss — beides regelt agent.py).

Der detach ist entschieden (Spec C4): der eigene Optimizer (LR × Plastizitäts-
Gen) trainiert NICHT durch GRU/Encoder hindurch — das Causal Model liest die
Repräsentation, es formt sie nicht (Repräsentations-Dynamik bleibt bei PPO).

Epistemische Neugier = max(0, mean(z²) − 1) mit z = (x − μ)/σ — Hinge AUSSEN:
für ein perfekt kalibriertes Modell ist E[mean(z²)] = 1, der Hinge schickt den
Term gegen 0; ein Hinge innen würde dauerhaft ≈ 0.48 zahlen (Rausch-Rente).

Ideen (err_ema, Bucket-Counts) aus archive/artificial_society/systems/
causal_model.py — der numpy/12-Prop-Code selbst ist strukturell nicht
übernehmbar. Dies ist KEIN registriertes Sim-System, sondern ein
Pro-Agent-Modul (Besitz: agent.causal_model, nur physics_v2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

CAUSAL_HIDDEN_IN = 96  # detach(gru_h)
CAUSAL_ACTION_DIMS = 22  # 5 Verben + 1 Effort + 8 Target-Query + 8 Tool-Query
CAUSAL_EMBED_DIMS = 32  # detach(Slot-Embed des gewählten Ziels)
CAUSAL_INPUT = CAUSAL_HIDDEN_IN + CAUSAL_ACTION_DIMS + CAUSAL_EMBED_DIMS  # 150
CAUSAL_TARGET = 17  # Slot-Features des getrackten Objekts im nächsten Tick
CAUSAL_LR = 3e-4  # × Plastizitäts-Gen (geklemmt wie Brain: 0.5..2.5)
LOG_SIGMA_FLOOR = -2.0  # Spec C4: früh ist σ winzig → z² kann 100+ erreichen
LOG_SIGMA_CEIL = 2.0
ERR_EMA_ALPHA = 0.1  # Logging-Signal (Archive-Idee)


class CausalModelV2(nn.Module):
    def __init__(self, plasticity: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CAUSAL_INPUT, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, CAUSAL_TARGET)
        self.logsigma_head = nn.Linear(64, CAUSAL_TARGET)
        lr = CAUSAL_LR * max(0.5, min(2.5, float(plasticity)))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.err_ema = 1.0

    def forward(self, x):
        z = self.net(x)
        mu = self.mu_head(z)
        log_sigma = self.logsigma_head(z).clamp(LOG_SIGMA_FLOOR, LOG_SIGMA_CEIL)
        return mu, log_sigma

    def observe(self, gru_h, action22, slot_embed, target17) -> dict:
        """Ein Beobachtungs-Schritt: Epistemik VOR dem Update (Überraschung des
        aktuellen Modells), dann ein NLL-Gradientenschritt, dann err_ema."""
        x = torch.cat(
            [
                gru_h.detach().reshape(-1).float(),
                action22.detach().reshape(-1).float(),
                slot_embed.detach().reshape(-1).float(),
            ]
        ).unsqueeze(0)
        target = target17.detach().reshape(1, CAUSAL_TARGET).float()

        with torch.no_grad():
            mu, log_sigma = self.forward(x)
            z2 = ((target - mu) / log_sigma.exp()).pow(2)
            epistemic = max(0.0, float(z2.mean()) - 1.0)  # Hinge AUSSEN (Spec C4)
            mean_abs = float((target - mu).abs().mean())

        mu, log_sigma = self.forward(x)
        nll = (0.5 * ((target - mu) / log_sigma.exp()).pow(2) + log_sigma).mean()
        self.optimizer.zero_grad()
        nll.backward()
        self.optimizer.step()

        self.err_ema = (1.0 - ERR_EMA_ALPHA) * self.err_ema + ERR_EMA_ALPHA * mean_abs
        return {"nll": float(nll.detach()), "epistemic": epistemic, "mean_abs_err": mean_abs}
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/systems/test_causal_model_v2.py -q`
Expected: `6 passed`

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 287)
../venv/bin/python -m ruff check --fix artificial_society/systems/causal_model.py tests/systems/test_causal_model_v2.py
../venv/bin/python -m ruff format artificial_society/systems/causal_model.py tests/systems/test_causal_model_v2.py
git add artificial_society/systems/causal_model.py tests/systems/test_causal_model_v2.py
git commit -m "feat(3b C4): CausalModelV2 — torch-NLL-Praediktor, Hinge-Epistemik aussen, sigma-Floor, err_ema"
```

### Task 13: PPO v2 — `_train_v2` (γ 0.99, 4 Epochen, 4×32 Minibatches, KL-Stop) + `finalize_terminal`

Die PPO-Anpassungen (C2): 20 Epochen auf einem einzigen Batch kollabieren σ und töten die Verben, bevor Kadaver und Klinge je koinzidieren — deshalb 4 Epochen über 4×32-Minibatches mit KL-Early-Stop 0.02 und Entropy pro Kopf-Gruppe. `finalize_terminal` liefert die echte Terminal-Transition beim Tod: `done=True` erreicht den Buffer, `r_death = −3.0` wird genau einmal gemünzt, der Restbuffer wird geflusht und trainiert.

**Files:**
- Modify: `artificial_society/agents/brain.py`
- Test: `tests/agents/test_brain_v2_training.py` (NEU)

**Interfaces:**
- Consumes: `evaluate_actions_v2` (Task 10), `_world_loss_v2`/`predict_world_v2` (Task 11), Transition-Format aus Task 10, Konstanten aus Task 8.
- Produces (FIX):
  - `Brain.maybe_train()` — dispatcht bei `self.physics_v2` auf `_train_v2` (v1-Körper UNVERÄNDERT)
  - `Brain._train_v2(batch: list[dict]) -> float | None` — GAE mit `GAMMA_V2`; Minibatch-Permutation via `torch.randperm` (global geseedeter torch-Strom); KL-Early-Stop VOR dem Optimizer-Schritt des betroffenen Minibatches; bricht das GESAMTE Training ab
  - `Brain.finalize_terminal(death_reward=DEATH_REWARD_V2) -> float | None` — mutiert die letzte Transition (`done=True`, `reward += −3.0`, re-geclampt), trainiert bei `len ≥ 2`, leert den Buffer immer; No-op bei leerem Buffer

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/agents/test_brain_v2_training.py`:

```python
"""D3: PPO-v2-Parameter, KL-Early-Stop, Terminal-Transition + Buffer-Flush."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.brain import (
    GAMMA_V2,
    KL_EARLY_STOP_V2,
    MINIBATCH_SIZE_V2,
    N_MINIBATCHES_V2,
    PPO_EPOCHS_V2,
    REWARD_CLAMP,
    ROLLOUT_HORIZON,
    Brain,
)


def _brain():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def _fuelle_buffer(brain, n, reward=0.5):
    feats = np.zeros((10, 17), dtype=np.float32)
    feats[0] = 0.3
    mask = np.zeros(10, dtype=bool)
    mask[0] = True
    masks = {
        "grasp": mask.copy(),
        "held": np.zeros(10, dtype=bool),
        "target": np.zeros(10, dtype=bool),
    }
    hidden = brain.initial_hidden()
    for _ in range(n):
        step = brain.act_v2([0.1] * 57, hidden, feats, mask, masks)
        hidden = step["next_hidden"]
        brain.store_transition_v2(step, reward, False, [0.2] * 57, feats, mask)


def test_ppo_v2_konstanten():
    assert GAMMA_V2 == 0.99
    assert PPO_EPOCHS_V2 == 4
    assert (N_MINIBATCHES_V2, MINIBATCH_SIZE_V2) == (4, 32)
    assert KL_EARLY_STOP_V2 == 0.02


def test_maybe_train_v2_trainiert_bei_128_und_leert_buffer():
    brain = _brain()
    _fuelle_buffer(brain, ROLLOUT_HORIZON - 1)
    assert brain.maybe_train() is None  # unterhalb des Horizonts: nichts
    _fuelle_buffer(brain, 1)
    vorher = [p.detach().clone() for p in brain.parameters()]
    loss = brain.maybe_train()
    assert loss is not None and np.isfinite(loss)
    assert len(brain.rollout) == 0
    geaendert = any(
        not torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert geaendert, "Training muss Gewichte bewegen"


def test_kl_early_stop_bricht_training_ab():
    """KL > 0.02 ⇒ Abbruch VOR dem Optimizer-Schritt des Minibatches."""
    brain = _brain()
    _fuelle_buffer(brain, ROLLOUT_HORIZON)
    # Alte log-probs künstlich massiv verschieben ⇒ approx-KL riesig
    for t in brain.rollout.storage:
        t["log_prob"] = t["log_prob"] + 10.0
    vorher = [p.detach().clone() for p in brain.parameters()]
    brain.maybe_train()
    unveraendert = all(
        torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert unveraendert, "KL-Stop muss greifen, bevor der erste Schritt appliziert wird"


def test_finalize_terminal_setzt_done_und_r_death_und_flusht():
    """D3: sterbender Agent schreibt done=True-Transition, Buffer wird trainiert."""
    brain = _brain()
    _fuelle_buffer(brain, 10, reward=0.5)
    letzte = brain.rollout.storage[-1]
    assert letzte["done"] is False
    erwartet = max(-REWARD_CLAMP, min(REWARD_CLAMP, letzte["reward"] - 3.0))
    vorher = [p.detach().clone() for p in brain.parameters()]

    loss = brain.finalize_terminal()

    assert letzte["done"] is True
    assert letzte["reward"] == erwartet  # r_death = −3.0 genau einmal
    assert loss is not None and np.isfinite(loss)
    assert len(brain.rollout) == 0  # Restbuffer geflusht
    geaendert = any(
        not torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert geaendert, "Terminal-Flush muss trainieren"


def test_finalize_terminal_randfaelle():
    brain = _brain()
    assert brain.finalize_terminal() is None  # leerer Buffer: No-op
    _fuelle_buffer(brain, 1)
    assert brain.finalize_terminal() is None  # 1 Transition: kein Training (GAE/Norm.)
    assert len(brain.rollout) == 0  # aber geleert
    # F2 (Review): kleine Buffer MÜSSEN trainieren — n=4 macht Gradientenschritte
    # (mb_size = max(2, n // 4); mit max(1, …) würde der 1er-Skip alles überspringen).
    _fuelle_buffer(brain, 4)
    vorher = [p.detach().clone() for p in brain.parameters()]
    loss = brain.finalize_terminal()
    assert loss is not None and np.isfinite(loss)
    geaendert = any(
        not torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert geaendert, "Terminal-Flush bei n=4 muss trainieren (Spec C2: geflusht UND trainiert)"
    assert len(brain.rollout) == 0
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_training.py -q`
Expected: FAIL — `maybe_train` läuft in den v1-Pfad und crasht am v2-Transition-Format (`KeyError`) bzw. `AttributeError: finalize_terminal`.

- [ ] **Step 3: Implementierung**

(a) In `artificial_society/agents/brain.py`, Methode `maybe_train`: füge NUR den folgenden Block direkt nach dem Horizon-Check (`if len(self.rollout) < ROLLOUT_HORIZON: return None`) ein — die Methode NICHT ersetzen, der restliche v1-Körper (ab `batch = self.rollout.storage`) bleibt Zeichen für Zeichen bestehen:

```python
        if self.physics_v2:
            # v2 (C2): eigener Trainings-Pfad (γ 0.99, Minibatches, KL-Stop).
            loss = self._train_v2(self.rollout.storage)
            self.rollout.clear()
            return loss
```

(b) Nach `maybe_train` die neuen Methoden einfügen:

```python
    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): PPO-Anpassungen (Spec C2)
    # ------------------------------------------------------------------
    def _train_v2(self, batch):
        """PPO über den v2-Buffer: GAMMA 0.99, 4 Epochen, 4×32-Minibatches,
        KL-Early-Stop 0.02, Entropy pro Kopf-Gruppe (alt 0.004 / neu 0.01 inkl.
        Kategorial). 20 Epochen auf einem Batch würden σ kollabieren und die
        Verben töten, bevor Kadaver und Klinge je koinzidieren (RL-Review)."""
        n = len(batch)
        if n < 2:
            return None
        obs = torch.stack([t["obs"] for t in batch]).to(device)
        hid = torch.stack([t["hidden"] for t in batch]).to(device)
        sf = torch.stack([t["slot_feats"] for t in batch]).to(device)
        sm = torch.stack([t["slot_mask"] for t in batch]).to(device)
        actions = torch.stack([t["action"] for t in batch]).to(device)
        tm = torch.stack([t["target_mask"] for t in batch]).to(device)
        om = torch.stack([t["tool_mask"] for t in batch]).to(device)
        ti = torch.tensor([t["target_idx"] for t in batch], dtype=torch.long, device=device)
        oi = torch.tensor([t["tool_idx"] for t in batch], dtype=torch.long, device=device)
        old_log_probs = torch.stack([t["log_prob"] for t in batch]).view(-1).to(device)
        values = torch.stack([t["value"] for t in batch]).view(-1).to(device)
        rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32, device=device)
        dones = torch.tensor(
            [1.0 if t["done"] else 0.0 for t in batch], dtype=torch.float32, device=device
        )
        next_obs = torch.stack([t["next_obs"] for t in batch]).to(device)
        nsf = torch.stack([t["next_slot_feats"] for t in batch]).to(device)
        nsm = torch.stack([t["next_slot_mask"] for t in batch]).to(device)

        with torch.no_grad():
            _, _, next_values, _, _, _ = self.forward_v2(next_obs, hid, nsf, nsm)
            next_values = next_values.view(-1)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + GAMMA_V2 * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + GAMMA_V2 * GAE_LAMBDA * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # F2 (Review): max(2, …) — mit max(1, …) ergäbe n∈{2..7} mb_size 1 und
        # der 1er-Skip unten würde JEDEN Minibatch überspringen: finalize_terminal
        # machte bei 2–7 Transitionen null Gradientenschritte, die −3.0-Terminal-
        # Transition verfiele (Spec C2 verlangt „geflusht UND trainiert").
        mb_size = MINIBATCH_SIZE_V2 if n >= ROLLOUT_HORIZON else max(2, n // N_MINIBATCHES_V2)
        last_loss = None
        for _ in range(PPO_EPOCHS_V2):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                if len(idx) < 2:
                    continue  # 1er-Minibatch: MSE/Std entartet
                new_log_probs, ent_old, ent_new, new_values, next_hidden = (
                    self.evaluate_actions_v2(
                        obs[idx], hid[idx], sf[idx], sm[idx], actions[idx],
                        tm[idx], ti[idx], om[idx], oi[idx],
                    )
                )
                approx_kl = (old_log_probs[idx] - new_log_probs).mean()
                if float(approx_kl) > KL_EARLY_STOP_V2:
                    return last_loss  # KL-Early-Stop: GESAMTES Training abbrechen
                ratios = torch.exp(new_log_probs - old_log_probs[idx])
                unclipped = ratios * advantages[idx]
                clipped_r = torch.clamp(ratios, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages[idx]
                actor_loss = -torch.min(unclipped, clipped_r).mean()
                critic_loss = F.mse_loss(new_values.view(-1), returns[idx])
                world_loss = self._world_loss_v2(
                    next_hidden, actions[idx], next_obs[idx], nsf[idx], nsm[idx], rewards[idx]
                )
                loss = (
                    ACTOR_COEF * actor_loss
                    + CRITIC_COEF * critic_loss
                    + WORLD_COEF * world_loss
                    - ENTROPY_COEF * ent_old.mean()
                    - ENTROPY_COEF_NEW * ent_new.mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())
        return last_loss

    def finalize_terminal(self, death_reward=DEATH_REWARD_V2):
        """Echte Terminal-Transition beim Tod (C2/C3): done=True erreicht den
        Buffer, r_death (−3.0) wird GENAU EINMAL auf die letzte Transition
        gemünzt (Aufrufort: der ursachen-agnostische Todes-Aggregationspunkt
        Simulation.remove_dead), der Restbuffer wird geflusht und trainiert.
        Heute stirbt ein Agent, bevor store_transition den Todes-Tick je sieht
        — der Buffer verfiel untrainiert."""
        if not self.rollout.storage:
            return None
        last = self.rollout.storage[-1]
        last["reward"] = max(-REWARD_CLAMP, min(REWARD_CLAMP, last["reward"] + death_reward))
        last["done"] = True
        loss = self._train_v2(self.rollout.storage) if len(self.rollout) >= 2 else None
        self.rollout.clear()
        return loss
```

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `../venv/bin/python -m pytest tests/agents/test_brain_v2_training.py tests/agents/test_brain_v2_policy.py -q`
Expected: `14 passed` (5 + 9)

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 292)
../venv/bin/python -m ruff check --fix artificial_society/agents/brain.py tests/agents/test_brain_v2_training.py
../venv/bin/python -m ruff format artificial_society/agents/brain.py tests/agents/test_brain_v2_training.py
git add artificial_society/agents/brain.py tests/agents/test_brain_v2_training.py
git commit -m "feat(3b C2): PPO v2 — gamma 0.99, 4 Epochen, 4x32 Minibatches, KL-Stop 0.02, finalize_terminal"
```

---

### Task 14: agent.py v2 — Wahrnehmung, `act_v2`, Verb→`do_*`-Mapping, Embodiment-Verdrahtung

Der Agent nimmt im v2-Modus Slots wahr, sampelt über `act_v2` (Planner-frei) und führt EINE verkörperte Aktion pro Tick aus (Mapping auf `do_grasp/do_release/do_strike/do_cut/do_eat`; `ActionResult`-Deltas auf energy/health). `attach_body`/`ensure_fields` rüsten v2-Brain, Causal Model, Novelty-Buckets. Die verkörperte Aktion läuft VOR `primitive_move` (Manipulation an der Position, an der wahrgenommen wurde — die gesampelten Masken bleiben konsistent zur Ausführung). Reward bleibt in diesem Task noch v1-verdrahtet (Task 15 baut ihn um) — der v2-Pfad ist nach diesem Task lauffähig, aber noch nicht C3-rein.

**Files:**
- Modify: `artificial_society/agents/agent.py`
- Test: `tests/test_physics_v2_brain_integration.py` (NEU)

**Interfaces:**
- Consumes: `build_slots`/`admissible_masks`/`resolve_slot_of`/`NoveltyBuckets` (Tasks 6/7), `act_v2` (Task 9), `do_*`/`ActionResult.remainder` (Task 5), `CausalModelV2` (Task 12), `ensure_strength_gene` (Task 4).
- Produces (FIX, von Task 15 konsumiert):
  - `Agent._execute_embodied(world, brain_step, view) -> ActionResult | None` — setzt `self._causal_next_target` (PhysObject | None; strike-Zerstörung → massereichstes Fragment, cut → remainder), wendet Deltas an, pflegt `metrics["verbs_fired"]` (= Ausführungs-VERSUCHE, auch ok=False), `metrics["verbs_failed"]` (ok=False-Teilmenge), `metrics["verbs_noop"]`, `metrics["discovery_events_by_agent"]`
  - `Agent._resolve_causal_pending(view) -> None` — setzt `self._causal_epistemic_pending` (float) aus dem Pending des Vortricks; leert `self._causal_pending`
  - `ensure_fields`/`attach_body` erzeugen: `agent.brain (physics_v2=True)`, `agent.causal_model`, `agent._novelty_buckets`, `agent._causal_pending = None`, `agent._causal_epistemic_pending = 0.0`, `agent.curiosity_last`
  - update()-Lokale des v2-Pfads: `view` (SlotView der Entscheidungs-Wahrnehmung), `brain_step` (act_v2-Dict)

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/test_physics_v2_brain_integration.py`:

```python
"""3b-Integration: v2-Agent nimmt Slots wahr, act_v2 läuft im Sim-Tick, Verben mappen auf do_*."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.agent import Agent, attach_body
from artificial_society.agents.perception_v2 import build_slots
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _sim_v2(seed=42):
    return Simulation(seed=seed, physics_v2=True, **_PARAMS)


def _erzwinge_verb(agent, verb_idx, target_slot, tool_slot=-1):
    """Monkeypatch-freier Zwang: wir rufen _execute_embodied direkt mit einem
    handgebauten brain_step — die Policy selbst bleibt Sampling (Prinzip 1);
    getestet wird hier NUR das Mapping Verb→do_* (Spec C2)."""
    view = build_slots(agent, agent._test_world.objects)
    action = torch.zeros(1, 29)
    step = {
        "verb": ("grasp", "release", "strike", "cut", "eat")[verb_idx],
        "effort": 0.9,
        "target_idx": target_slot,
        "tool_idx": tool_slot,
        "action_tensor": action,
        "slot_embeds": torch.zeros(10, 32),
    }
    return agent._execute_embodied(agent._test_world, step, view), view


def test_attach_body_ruestet_v2_brain_und_causal_model():
    torch.manual_seed(0)
    agent = Agent.spawn_random(3, 3)
    attach_body(agent)
    assert agent.brain.physics_v2 is True
    assert agent.brain.action_size == 29
    assert agent.causal_model is not None
    assert agent._novelty_buckets is not None
    assert agent._causal_pending is None


def test_v2_kind_erbt_gewichte_vom_v2_eltern_brain():
    sim = _sim_v2()
    eltern = sim.agents[0]
    kind = sim.spawn_child_from_parent(eltern, dict(eltern.genes))
    assert kind.brain.physics_v2 is True
    assert kind.brain.gru.input_size == 192
    assert "strength" in kind.genes


def test_verb_mapping_grasp_und_release():
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    stein = make_object("granite", 1.0)
    sim.world.objects.add(stein, agent.pos, source="spawned")
    view = build_slots(agent, sim.world.objects)
    slot = view.objs.index(stein)

    result, _ = _erzwinge_verb(agent, 0, slot)  # grasp
    assert result.ok and stein in agent.hands.held
    assert sim.world.objects.metrics["verbs_fired"]["grasp"] == 1

    view2 = build_slots(agent, sim.world.objects)
    slot2 = view2.objs.index(stein)  # jetzt Hand-Slot 8
    assert slot2 == 8
    step = {
        "verb": "release", "effort": 0.5, "target_idx": slot2, "tool_idx": -1,
        "action_tensor": torch.zeros(1, 29), "slot_embeds": torch.zeros(10, 32),
    }
    result2 = agent._execute_embodied(sim.world, step, view2)
    assert result2.ok and agent.hands.held == []
    assert sim.world.objects.position_of(stein) == agent.pos


def test_verb_mapping_eat_wendet_deltas_an():
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    fleisch = make_object("raw_meat", 0.5)
    fleisch.props[10] = 0.5  # toxicity hoch → Health-Delta sichtbar
    sim.world.objects.add(fleisch, agent.pos, source="spawned")
    agent.energy, agent.health = 50.0, 90.0
    view = build_slots(agent, sim.world.objects)
    slot = view.objs.index(fleisch)
    result, _ = _erzwinge_verb(agent, 4, slot)  # eat
    assert result.ok
    assert agent.energy > 50.0  # energy_delta_sim angewendet
    assert agent.health < 90.0  # health_delta (Toxin) angewendet


def test_verb_mapping_strike_setzt_causal_target_auf_fragment():
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    agent.body.strength = 1.0
    hammer = make_object("granite", 1.5)
    flint = make_object("flint", 0.8)
    agent.hands.held.append(hammer)
    sim.world.objects.ledger["spawned"] += 1.5
    sim.world.objects.add(flint, agent.pos, source="spawned")
    view = build_slots(agent, sim.world.objects)
    ziel, werkzeug = view.objs.index(flint), view.objs.index(hammer)
    result, _ = _erzwinge_verb(agent, 2, ziel, tool_slot=werkzeug)  # strike
    assert result.ok and result.fragments
    schwerstes = max(result.fragments, key=lambda f: f.mass)
    assert agent._causal_next_target is schwerstes  # C4: massereichstes Fragment


def test_v2_sim_laueft_und_planner_bleibt_stumm():
    """8 Agenten × 6 Ticks über sim.step(): act_v2 im Einsatz, kein plan_action."""
    aufrufe = []
    from artificial_society.agents.brain import Brain

    original = Brain.plan_action

    def spion(self, *a, **kw):
        aufrufe.append(1)
        return original(self, *a, **kw)

    Brain.plan_action = spion
    try:
        sim = _sim_v2()
        for _ in range(6):
            sim.step()
    finally:
        Brain.plan_action = original
    assert aufrufe == [], "C5: Planner ist im v2-Modus deaktiviert"
    assert all(a.brain.physics_v2 for a in sim.agents)
    assert sum(len(a.brain.rollout) for a in sim.agents) > 0, "Transitionen müssen fließen"
    assert not any(np.isnan(a.last_reward) for a in sim.agents)
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `../venv/bin/python -m pytest tests/test_physics_v2_brain_integration.py -q`
Expected: FAIL — `attach_body` baut kein v2-Brain (`assert agent.brain.physics_v2 is True` schlägt fehl), `_execute_embodied` existiert nicht.

- [ ] **Step 3: Implementierung agent.py — Imports, attach_body, ensure_fields**

(a) Imports ergänzen (nach dem `life_stage`-Import):

```python
from artificial_society.agents.perception_v2 import (
    NoveltyBuckets,
    admissible_masks,
    build_slots,
    resolve_slot_of,
)
```

und nach dem `culture`-Import:

```python
from artificial_society.systems.causal_model import CausalModelV2
```

sowie in der bestehenden actions-Import-Zeile (Zeile 20) die `do_*` ergänzen:

```python
from artificial_society.environment.physics.actions import (
    do_cut,
    do_eat,
    do_grasp,
    do_release,
    do_strike,
    enforce_carry_budget,
)
```

(b) `attach_body` (Stand nach Task 4) erweitern:

```python
def attach_body(agent) -> None:
    """Physik-v2-Embodiment: Body + Hände aus dem strength-Gen, v2-Brain,
    Causal Model und Novelty-Buckets (Plan 3b). Zieht RNG nur im v2-Pfad.

    Das v2-Brain wird VOR inherit_weights_from gebaut (spawn_child_from_parent
    ruft attach_body zuerst) — Eltern- und Kind-Brain sind dann form-gleich.
    """
    agent.physics_v2 = True
    ensure_strength_gene(agent.genes)
    agent.body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=agent.genes["strength"])
    agent.hands = Hands()
    plast = agent.genes.get("plasticity", 1.0)
    agent.brain = Brain(plasticity=plast, physics_v2=True)
    agent.hidden_state = agent.brain.initial_hidden()
    agent.causal_model = CausalModelV2(plasticity=plast)
    agent._novelty_buckets = NoveltyBuckets()
    agent._causal_pending = None
    agent._causal_epistemic_pending = 0.0
    agent.curiosity_last = {"nextslot": 0.0, "causal": 0.0, "novelty": 0.0}
```

(c) In `ensure_fields` NACH dem bestehenden Physik-v2-Block (hinter der `Hands()`-Zeile 189) ergänzen — alles idempotent, RNG-Draws nur wenn v2-Objekte fehlen (Checkpoint-Nachrüstung):

```python
    if agent.physics_v2:
        # Plan 3b: v2-Brain + Neugier-Apparat idempotent nachrüsten
        # (Checkpoint-geladene Agenten; frisch gebaute sind schon komplett).
        if not getattr(agent.brain, "physics_v2", False):
            print(f"[compat] Agent {agent.id}: v1-Brain im v2-Modus, rebuilding.")
            agent.brain = Brain(
                plasticity=agent.genes.get("plasticity", 1.0), physics_v2=True
            )
            agent.hidden_state = agent.brain.initial_hidden()
        if getattr(agent, "causal_model", None) is None:
            agent.causal_model = CausalModelV2(
                plasticity=agent.genes.get("plasticity", 1.0)
            )
        if getattr(agent, "_novelty_buckets", None) is None:
            agent._novelty_buckets = NoveltyBuckets()
        if not hasattr(agent, "_causal_pending"):
            agent._causal_pending = None
        if not hasattr(agent, "_causal_epistemic_pending"):
            agent._causal_epistemic_pending = 0.0
        if not hasattr(agent, "curiosity_last"):
            agent.curiosity_last = {"nextslot": 0.0, "causal": 0.0, "novelty": 0.0}
```

- [ ] **Step 4: Implementierung agent.py — `_resolve_causal_pending` + `_execute_embodied`**

Als neue Methoden der `Agent`-Klasse (nach `_sleep_tick` einfügen):

```python
    def _resolve_causal_pending(self, view) -> None:
        """C4: löst das Causal-Pending des Vortricks gegen die aktuelle
        Wahrnehmung auf — dasselbe Objekt (id-basiert) einen Tick später.
        Aus der Wahrnehmung gefallen ⇒ maskiert, kein Loss, Epistemik 0."""
        self._causal_epistemic_pending = 0.0
        pending = getattr(self, "_causal_pending", None)
        if pending is None:
            return
        self._causal_pending = None
        slot = resolve_slot_of(view, pending["target"])
        if slot < 0:
            return
        out = self.causal_model.observe(
            pending["h"],
            pending["act"],
            pending["embed"],
            torch.as_tensor(view.feats[slot]),
        )
        self._causal_epistemic_pending = out["epistemic"]

    def _execute_embodied(self, world, brain_step, view):
        """C2/B4: Mapping des aktiven Verbs auf do_grasp/do_release/do_strike/
        do_cut/do_eat — EINE verkörperte Aktion pro Tick, an der Position der
        Wahrnehmung (vor primitive_move). ActionResult-Deltas gehen auf
        energy/health; D4-Metriken (Verb-Raten, DiscoveryV2 je Agent) werden
        an der ObjectLayer gepflegt. Kein Reward hier (C3 zahlt Physiologie)."""
        self._causal_next_target = None
        verb = brain_step["verb"]
        if verb is None:
            return None
        layer = world.objects
        metrics = layer.metrics
        target_idx = brain_step["target_idx"]
        tool_idx = brain_step["tool_idx"]
        target = view.objs[target_idx] if target_idx >= 0 else None
        tool = view.objs[tool_idx] if tool_idx >= 0 else None
        if target is None:
            noop = metrics.setdefault("verbs_noop", {})
            noop[verb] = noop.get(verb, 0) + 1
            return None
        effort = brain_step["effort"]

        if verb == "grasp":
            result = do_grasp(self.body, self.hands, layer, self.pos, target)
        elif verb == "release":
            result = do_release(self.body, self.hands, layer, self.pos, target)
        elif verb == "strike":
            if tool is None:
                return None  # act_v2 verhindert das; defensiv trotzdem No-op
            n_discovery = len(layer.discovery.entries)
            result = do_strike(
                self.body, self.hands, layer, self.pos, tool, target, effort, random
            )
        elif verb == "cut":
            n_discovery = len(layer.discovery.entries)
            result = do_cut(self.body, self.hands, layer, self.pos, tool, target, effort)
        elif verb == "eat":
            result = do_eat(self.body, self.hands, layer, self.pos, target)
        else:
            return None

        if result.energy_delta_sim:
            self.energy = max(0.0, min(MAX_ENERGY, self.energy + result.energy_delta_sim))
        if result.health_delta:
            self.health = max(0.0, self.health + result.health_delta)
            if self.health <= 0:
                self.alive = False  # Toxin-Tod: Terminal via remove_dead (Task 15)

        # D4 (Review F4): verbs_fired zählt Ausführungs-VERSUCHE — auch ok=False
        # (z. B. target_out_of_reach nach Überlast-Drop). verbs_failed zählt die
        # ok=False-Teilmenge separat: die Pilot-Diagnostik braucht Versuchsrate
        # UND Erfolgsrate der Policy (Erfolge = fired − failed).
        fired = metrics.setdefault("verbs_fired", {})
        fired[verb] = fired.get(verb, 0) + 1
        if not result.ok:
            failed = metrics.setdefault("verbs_failed", {})
            failed[verb] = failed.get(verb, 0) + 1
        if verb in ("strike", "cut"):
            gewachsen = len(layer.discovery.entries) - n_discovery
            if gewachsen:
                je_agent = metrics.setdefault("discovery_events_by_agent", {})
                je_agent[self.id] = je_agent.get(self.id, 0) + gewachsen

        # C4: das Causal-Target folgt der physischen Fortsetzung des Objekts —
        # strike-Zerstörung: massereichstes Fragment; cut: remainder.
        naechstes = target
        if verb == "strike" and result.fragments:
            naechstes = max(result.fragments, key=lambda f: f.mass)
        elif verb == "cut" and result.remainder is not None:
            naechstes = result.remainder
        self._causal_next_target = naechstes
        return result
```

(`do_strike` erwartet duck-typed RNG mit `.randint/.random` — das global via `seed_all` geseedete `random`-Modul erfüllt das; Determinismus-Kontrakt bleibt gewahrt, Draws passieren nur im v2-Pfad.)

- [ ] **Step 5: Implementierung agent.py — update()-Verdrahtung**

(a) Den Planning/act-Block in `update` (Zeilen 1143–1168, von `self._planning_stride = (` bis zum Ende des `action = {...}`-Dicts) ersetzen durch:

```python
        if self.physics_v2:
            # v2 (C1/C2/C5): Objekt-Slots wahrnehmen, Causal-Pending des
            # Vortricks auflösen, dann Policy-Sampling OHNE Planner —
            # plan_action/imagine_rollout sind mit der v2-Architektur
            # inkompatibel (encoder(pred_next_obs) ohne obj_ctx; argmax).
            view = build_slots(self, world.objects)
            self._resolve_causal_pending(view)
            brain_step = self.brain.act_v2(
                features, self.hidden_state, view.feats, view.mask, admissible_masks(view)
            )
        else:
            self._planning_stride = (
                2 if getattr(self, "goal_stack", None) and not self.goal_stack.is_empty() else 4
            )
            use_planning = tick >= getattr(self, "_next_planning_tick", 0)
            if use_planning:
                self._next_planning_tick = tick + self._planning_stride

            research_mode = self._need_inv_cooldown <= 0 or (
                getattr(self, "goal_stack", None) is not None
                and not self.goal_stack.is_empty()
            )
            brain_step = self.brain.act(
                features,
                self.hidden_state,
                use_planning=use_planning,
                research_mode=research_mode,
            )
        self.hidden_state = brain_step["next_hidden"]
        action_list = brain_step["action_list"]
        action = {
            "move_x": action_list[0],
            "move_y": action_list[1],
            "forage": action_list[2],
            "cooperate": action_list[3],
            "attack": action_list[4],
            "build": action_list[5],
        }
```

(b) Direkt NACH dem `if not self.physics_v2 and getattr(self, "goal_stack", ...)`-Block und VOR `self.primitive_move(world, action)` einfügen:

```python
        if self.physics_v2:
            self._causal_next_target = None  # pro Tick frisch (auch im Schlaf)
            if not self.is_sleeping:
                # EINE verkörperte Aktion pro Tick (B4), ausgeführt an der
                # Wahrnehmungs-Position (vor primitive_move — die gesampelten
                # Zulässigkeits-Masken bleiben konsistent zur Ausführung).
                self._execute_embodied(world, brain_step, view)
```

(c) Transition-Speicherung dispatchen (sonst liefe der v2-Buffer mit v1-Format-Dicts in `_train_v2` und crashte bei 128 Ticks): den `self.brain.store_transition(...)`-Aufruf (Zeilen 1299–1308) ersetzen durch:

```python
        if self.physics_v2:
            # v2-Transition (C2): Slot-Kontext + Masken + Indizes; next_view =
            # Wahrnehmung NACH der Aktion (Task 15 nutzt sie auch für die Neugier).
            next_view = build_slots(self, world.objects)
            self.brain.store_transition_v2(
                brain_step,
                effective_reward,
                not self.alive,
                next_features_raw,
                next_view.feats,
                next_view.mask,
            )
        else:
            self.brain.store_transition(
                brain_step["obs_tensor"],
                brain_step["hidden_in"],
                brain_step["action_tensor"],
                brain_step["log_prob"],
                brain_step["value"],
                effective_reward,
                not self.alive,
                next_features_raw,
            )
```

(Der Reward ist in diesem Task übergangsweise noch der v1-Wert — Task 15 ersetzt ihn durch die C3-Terme. Der v1-Intrinsic-Block läuft übergangsweise auch im v2 mit; er funktioniert formal auf dem v2-Brain (predict_world nimmt die 29er-Aktion) und wird in Task 15 v1-gegatet.)

(d) Test-Hilfszugriff: KEINE weiteren Änderungen — der Test hängt `agent._test_world` selbst an.

- [ ] **Step 6: Tests laufen lassen — grün**

Run: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest tests/test_physics_v2_brain_integration.py tests/test_physics_v2_death.py tests/test_physics_v2_embodiment.py tests/test_regression_golden.py -q`
Expected: `test_physics_v2_brain_integration.py: 6 passed`; Bestands-v2-Tests und Golden grün.

- [ ] **Step 7: Volle Suite + ruff + Commit**

```bash
rm -f checkpoint.pkl
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 298)
../venv/bin/python -m ruff check --fix artificial_society/agents/agent.py tests/test_physics_v2_brain_integration.py
../venv/bin/python -m ruff format artificial_society/agents/agent.py tests/test_physics_v2_brain_integration.py
git add artificial_society/agents/agent.py tests/test_physics_v2_brain_integration.py
git commit -m "feat(3b C2): Agent-v2-Verdrahtung — Slot-Wahrnehmung, act_v2, Verb->do_*-Mapping, Planner aus"
```

### Task 15: C3-Reward, Neugier-Assemblierung, Causal-Pending, Terminal-Verdrahtung

Der Belohnungs-Umbau (C3): der v2-Reward wird am Ende des Ticks aus den Physiologie-Termen NEU gebaut — alle v1-Reward-Akkumulationen (Forage-Event 0.05, Attack, Territorium, Koop, Sprache, Social-Learning, Trade) laufen mechanisch weiter, werden aber verworfen (B6: „System läuft" ≠ „System zahlt"). Keine `cognition_mult`-Skalierung. Die Neugier (C4) wird aus den drei Quellen assembliert; das Causal-Pending wird für den Folgetick angelegt; `remove_dead` finalisiert Terminal-Transitionen (`r_death = −3.0` genau einmal, ursachen-agnostisch).

**Files:**
- Modify: `artificial_society/agents/agent.py` (update()-Reward-Region, `_assemble_curiosity_v2`)
- Modify: `artificial_society/simulation.py` (`remove_dead`)
- Test: `tests/test_physics_v2_reward.py` (NEU)

**Interfaces:**
- Consumes: `nextslot_error` (Task 11), `finalize_terminal` (Task 13), `_causal_epistemic_pending`/`_causal_next_target`/`_novelty_buckets` (Task 14), `V1_HEAD_DIMS` aus brain.py.
- Produces (FIX):
  - `Agent._assemble_curiosity_v2(brain_step, next_features_raw, next_view, world) -> float` — geclampte curiosity; setzt `self.curiosity_last` (dict mit `nextslot`/`causal`/`novelty`), akkumuliert `metrics["curiosity_sums"]`, legt `self._causal_pending` an
  - v2-`last_reward` = exakt `r_energy + r_health + r_deficit + 0.3·curiosity` (Terminal-Malus kommt NICHT hier, sondern in `finalize_terminal` — genau einmal pro Tod)

- [ ] **Step 1: Failing Tests schreiben**

Neue Datei `tests/test_physics_v2_reward.py`:

```python
"""D3: Belohnungs-Reinheit (Test rechnet gegen), v1-Trigger tot (Spy + Positiv-Kontrolle),
Terminal-Transition über sim.step()."""

from __future__ import annotations

import numpy as np
import pytest

import artificial_society.agents.agent as agent_mod
from artificial_society.agents.brain import Brain
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _sim(physics_v2, seed=42):
    return Simulation(seed=seed, physics_v2=physics_v2, **_PARAMS)


def test_belohnungs_reinheit_gegen_gemockte_events(monkeypatch):
    """Gemockte Event-Quellen (Territorium, Social Learning, Sprache) liefern
    riesige Boni — der v2-Reward muss trotzdem EXAKT die C3-Terme sein."""
    monkeypatch.setattr(agent_mod, "territory_reward_for_agent", lambda a, w: 99.0)
    monkeypatch.setattr(agent_mod, "social_learning_step", lambda a, ags, t: 99.0)
    monkeypatch.setattr(
        agent_mod, "_maybe_mark_language", lambda a, c, t, v, r: r + 99.0
    )
    sim = _sim(physics_v2=True)
    agent = sim.agents[0]
    agent.is_sleeping = False
    e0, h0 = agent.energy, agent.health

    agent.update(sim.world, sim.agents, tick=0, tribes=sim.tribes,
                 economy=sim.economy, technology=sim.technology)

    cl = agent.curiosity_last
    curiosity = max(0.0, min(2.0, 0.25 * cl["nextslot"] + 0.5 * cl["causal"] + 0.25 * cl["novelty"]))
    erwartet = (
        0.6 * (agent.energy - e0) / 45.0
        + 0.6 * (agent.health - h0) / 50.0
        - 0.3 * max(0.0, (60.0 - agent.energy) / 60.0)
        + 0.3 * curiosity
    )
    assert agent.last_reward == pytest.approx(erwartet, abs=1e-9), (
        "v2-Reward muss exakt r_energy + r_health + r_deficit + 0.3*curiosity sein"
    )


def test_v2_reward_ohne_cognition_skalierung(monkeypatch):
    """C3 exakt: kein cognition_mult im v2 (der wäre ein multiplikativer Shaping-Kanal)."""
    sim = _sim(physics_v2=True)
    agent = sim.agents[0]
    original_modifiers = agent.endocrine.modifiers  # VOR dem Patch binden (sonst Rekursion)
    monkeypatch.setattr(
        agent.endocrine, "modifiers", lambda: {**original_modifiers(), "cognition": 3.0}
    )
    e0, h0 = agent.energy, agent.health
    agent.update(sim.world, sim.agents, tick=0, tribes=sim.tribes,
                 economy=sim.economy, technology=sim.technology)
    cl = agent.curiosity_last
    curiosity = max(0.0, min(2.0, 0.25 * cl["nextslot"] + 0.5 * cl["causal"] + 0.25 * cl["novelty"]))
    erwartet = (
        0.6 * (agent.energy - e0) / 45.0
        + 0.6 * (agent.health - h0) / 50.0
        - 0.3 * max(0.0, (60.0 - agent.energy) / 60.0)
        + 0.3 * curiosity
    )
    assert agent.last_reward == pytest.approx(erwartet, abs=1e-9)


def _mit_spy(monkeypatch, ziel_modul, name):
    zaehler = {"n": 0}
    original = getattr(ziel_modul, name)

    def spion(*a, **kw):
        zaehler["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(ziel_modul, name, spion)
    return zaehler


def test_v1_trigger_tot_mit_v1_positiv_kontrolle(monkeypatch):
    """D3/F7: Invention-Trigger + Planner per Spy überwacht — 0 Aufrufe im v2
    über 6 Ticks, UND die Positiv-Kontrolle beweist, dass die Spies greifen."""
    need = _mit_spy(monkeypatch, agent_mod, "agent_invent_from_need")
    inv = _mit_spy(monkeypatch, agent_mod, "agent_try_invention")
    cook = _mit_spy(monkeypatch, agent_mod, "agent_try_cook")
    plan = _mit_spy(monkeypatch, Brain, "plan_action")

    sim = _sim(physics_v2=True)
    for _ in range(6):
        sim.step()
    assert need["n"] == 0 and inv["n"] == 0 and cook["n"] == 0, "v1-Erfindung ist im v2 AUS (B6)"
    assert plan["n"] == 0, "Planner ist im v2 deaktiviert (C5)"

    # Positiv-Kontrolle: dieselben Spies feuern im v1-Modus (deterministisch:
    # invent_from_need läuft bei Cooldown 0 jeden Tick; plan_action bei tick 0).
    sim_v1 = _sim(physics_v2=False)
    for _ in range(6):
        sim_v1.step()
    assert need["n"] >= 1, "Positiv-Kontrolle: Spy muss im v1 feuern"
    assert plan["n"] >= 1, "Positiv-Kontrolle: Planner-Spy muss im v1 feuern"


def test_terminal_transition_ueber_sim_step(monkeypatch):
    """D3: Tod im Tick ⇒ finalize_terminal läuft genau einmal (remove_dead),
    der Buffer ist danach geleert."""
    aufrufe = []
    original = Brain.finalize_terminal

    def spion(self, *a, **kw):
        aufrufe.append(self)
        return original(self, *a, **kw)

    monkeypatch.setattr(Brain, "finalize_terminal", spion)
    sim = _sim(physics_v2=True)
    for _ in range(3):
        sim.step()  # Buffer der Agenten füllen
    opfer = sim.agents[0]
    opfer_brain = opfer.brain
    assert len(opfer_brain.rollout) > 0
    opfer.health = 0.01
    opfer.energy = 0.0

    sim.step()

    assert opfer not in sim.agents
    assert aufrufe.count(opfer_brain) == 1, "finalize_terminal genau EINMAL pro Tod"
    assert len(opfer_brain.rollout) == 0, "Restbuffer geflusht"


def test_v1_reward_pfad_unveraendert():
    """Verzweigungs-Gegenprobe: im v1 skaliert cognition_mult weiter und die
    Event-Rewards zählen (keine Regression durch den v2-Umbau)."""
    sim = _sim(physics_v2=False)
    agent = sim.agents[0]
    agent.update(sim.world, sim.agents, tick=0, tribes=sim.tribes,
                 economy=sim.economy, technology=sim.technology)
    assert not hasattr(agent, "curiosity_last") or agent.physics_v2 is False
    assert np.isfinite(agent.last_reward)
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest tests/test_physics_v2_reward.py -q`
Expected: Reinheits-Tests FAIL (v2-Reward enthält noch die 99er-Boni bzw. `curiosity_last` bleibt `{0,0,0}`-Init), Terminal-Test FAIL (finalize_terminal wird nie gerufen).

- [ ] **Step 3: Implementierung agent.py**

(a) Import aus brain.py erweitern (Zeile 9):

```python
from artificial_society.agents.brain import INPUT_SIZE, V1_HEAD_DIMS, Brain
```

(b) In `update` direkt nach `ensure_fields(self)` (Zeile 1085) einfügen:

```python
        if self.physics_v2:
            # C3: ΔE/ΔH über den GANZEN Tick (Metabolik, Arbeit, Essen, Toxin,
            # Schlaf-Regeneration, Koop-Transfers — alles Physiologie).
            e_start, h_start = self.energy, self.health
```

(c) Neue Methode `_assemble_curiosity_v2` (nach `_execute_embodied` einfügen):

```python
    def _assemble_curiosity_v2(self, brain_step, next_features_raw, next_view, world):
        """C4: curiosity = 0.25·nextslot_err + 0.50·causal_epistemic +
        0.25·novelty, geclampt [0, 2]. Loggt die Zerlegung (D4: drei Quellen
        separat) und legt das Causal-Pending für den nächsten Tick an."""
        nextslot_err = self.brain.nextslot_error(
            brain_step, next_features_raw, next_view.feats, next_view.mask
        )
        causal_epistemic = self._causal_epistemic_pending
        novelty = self._novelty_buckets.observe_view(next_view)
        curiosity = max(
            0.0, min(2.0, 0.25 * nextslot_err + 0.50 * causal_epistemic + 0.25 * novelty)
        )
        self.curiosity_last = {
            "nextslot": nextslot_err,
            "causal": causal_epistemic,
            "novelty": novelty,
        }
        sums = world.objects.metrics.setdefault(
            "curiosity_sums", {"nextslot": 0.0, "causal": 0.0, "novelty": 0.0}
        )
        sums["nextslot"] += nextslot_err
        sums["causal"] += causal_epistemic
        sums["novelty"] += novelty
        target = getattr(self, "_causal_next_target", None)
        if target is not None and brain_step["target_idx"] >= 0:
            # C4: Input = detach(gru_h) ⊕ verkörperter Aktions-Vektor (22)
            # ⊕ detach(Slot-Embed des gewählten Ziels); Target = dasselbe
            # Objekt im nächsten Tick (Auflösung: _resolve_causal_pending).
            self._causal_pending = {
                "h": brain_step["next_hidden"],
                "act": brain_step["action_tensor"].squeeze(0)[V1_HEAD_DIMS:],
                "embed": brain_step["slot_embeds"][brain_step["target_idx"]],
                "target": target,
            }
        return curiosity
```

(d) Die Region von `next_features_raw = self.local_features(world, agents)` bis einschließlich `effective_reward = reward * cognition_mult` (Stand nach Task 14) ersetzen durch:

```python
        next_features_raw = self.local_features(world, agents)
        if not self.physics_v2:
            # v1-Intrinsic (inkl. rew_err-Mechanik) und NGU-Episodic laufen NUR
            # im v1 — der v2 ersetzt beides durch die C4-Neugier (rew_err ist
            # ersatzlos gestrichen; Planner/NGU sind im v2 aus).
            intrinsic = self.brain.intrinsic_reward(
                brain_step["hidden_in"],
                brain_step["action_tensor"],
                next_features_raw,
            )
            reward += 0.3 * intrinsic

            next_obs_t = torch.tensor(
                next_features_raw,
                dtype=torch.float32,
                device=brain_step["hidden_in"].device,
            )
            self.brain.episodic_memory.novelty(next_obs_t)

        context_vec = np.asarray(next_features_raw, dtype=np.float32)
        reward = _maybe_mark_language(self, current_cell, tick, context_vec, reward)
        _observe_tokens(self, current_cell, context_vec, reward)
        _maybe_collect_language_convergence(self, agents, tick)
        _compact_material_inventory(self, getattr(self, "_inventory_cap", 24))

        cognition_mult = mods.get("cognition", 1.0)
        if self.physics_v2:
            # C3: nur Überleben + Neugier. Der bis hier akkumulierte v1-Reward
            # (Forage-Event, Koop, Attack, Territorium, Sprache, Social
            # Learning, Trade) wird bewusst VERWORFEN — die Mechanik lief, sie
            # zahlt nur nicht (B6). Keine cognition-Skalierung: C3 exakt.
            # r_death (−3.0) kommt NICHT hier, sondern genau einmal in
            # Brain.finalize_terminal am Todes-Aggregationspunkt remove_dead.
            next_view = build_slots(self, world.objects)
            curiosity = self._assemble_curiosity_v2(
                brain_step, next_features_raw, next_view, world
            )
            reward = (
                0.6 * (self.energy - e_start) / 45.0
                + 0.6 * (self.health - h_start) / 50.0
                - 0.3 * max(0.0, (60.0 - self.energy) / 60.0)
                + 0.3 * curiosity
            )
            effective_reward = reward
        else:
            effective_reward = reward * cognition_mult
```

(e) Der store-Dispatch aus Task 14 folgt direkt darauf — dort die inzwischen doppelte Zeile `next_view = build_slots(self, world.objects)` im v2-Zweig ENTFERNEN (das `next_view` aus (d) wird wiederverwendet):

```python
        if self.physics_v2:
            self.brain.store_transition_v2(
                brain_step,
                effective_reward,
                not self.alive,
                next_features_raw,
                next_view.feats,
                next_view.mask,
            )
        else:
            self.brain.store_transition(
                brain_step["obs_tensor"],
                brain_step["hidden_in"],
                brain_step["action_tensor"],
                brain_step["log_prob"],
                brain_step["value"],
                effective_reward,
                not self.alive,
                next_features_raw,
            )
```

- [ ] **Step 4: Implementierung simulation.py — Terminal am Todes-Aggregationspunkt**

In `remove_dead` den v2-Zweig (Zeilen 289–292) ersetzen:

```python
            self._broadcast_death_knowledge(agent)
            if self.physics_v2:
                # v2 (C2/C3): echte Terminal-Transition — done=True erreicht den
                # Buffer, r_death (−3.0) wird GENAU EINMAL gemünzt, der
                # Restbuffer wird geflusht und trainiert. remove_dead ist der
                # designierte, ursachen-agnostische Todes-Aggregationspunkt (B3).
                brain = getattr(agent, "brain", None)
                if brain is not None and getattr(brain, "physics_v2", False):
                    brain.finalize_terminal()
                # v2 (Spec B3): der Tod münzt genau EIN Kadaver-Objekt —
                # kein add_carcass-Credit auf Zell-Pools, kein Loot.
                self._spawn_carcass(agent)
            else:
                add_carcass(self.world, *agent.pos, CORPSE_ENERGY)
```

- [ ] **Step 5: Tests laufen lassen — grün**

Run: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest tests/test_physics_v2_reward.py tests/test_physics_v2_brain_integration.py tests/test_physics_v2_death.py -q`
Expected: `test_physics_v2_reward.py: 5 passed`; Bestands-Tests grün.

- [ ] **Step 6: Volle Suite + ruff + Commit**

```bash
rm -f checkpoint.pkl
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 303)
../venv/bin/python -m ruff check --fix artificial_society/agents/agent.py artificial_society/simulation.py tests/test_physics_v2_reward.py
../venv/bin/python -m ruff format artificial_society/agents/agent.py artificial_society/simulation.py tests/test_physics_v2_reward.py
git add artificial_society/agents/agent.py artificial_society/simulation.py tests/test_physics_v2_reward.py
git commit -m "feat(3b C3/C4): reiner Ueberleben+Neugier-Reward, Neugier-Assemblierung, Terminal-Transition in remove_dead"
```

---

### Task 16: Checkpoint-Versionierung (`CHECKPOINT_FORMAT_VERSION = 2`)

C5: v2-Brain-Formen sauber versioniert — der Loader wirft eine harte, verständliche Fehlermeldung statt eines stillen Frisch-Starts/Absturzes. **Team-ratifizierte Semantik (Koordinator-Entscheid zu Auflösung Nr. 12):** die 3a-Garantie „Legacy-Payload (ohne Version-Key) lädt mit `physics_v2=False`" BLEIBT bestehen — C5s Schutzabsicht gilt den v2-Brain-Tensor-Formen, nicht dem v1-Bestand (eine laufende v1-Sim muss nach Code-Update weiter resumen können). Konkret:

- Speichern schreibt IMMER `format_version = 2`.
- Laden: fehlender Key ⇒ Version 1. Version 1 + `physics_v2=False` ⇒ lädt normal — der bestehende 3a-Test `test_alt_checkpoint_ohne_key_laedt_nur_mit_flag_aus` bleibt UNVERÄNDERT und pinnt genau das, inklusive „Version 1 keyless + Flag an ⇒ Fehler" (das erledigt der bestehende 3a-Flag-Guard: Legacy defaultet `physics_v2=False ≠ True`; dafür ist kein neuer Code-Pfad nötig).
- Version ∉ {1, 2} ⇒ harter `CheckpointIncompatibleError` VOR dem broad-except, re-raise (3a-Muster).
- **v2-Brain-Formen-Guard (Kern von C5):** im v2-Modus prüft der Loader die geladenen Brains (Flag, 192er GRU, 29 Köpfe) VOR der `ensure_fields`-Schleife — sonst würde der `[compat]`-Rebuild einen Fremd-Checkpoint still „reparieren" (Gewichtsverlust) bzw. fremde v2-Formen später im Forward stumm crashen. Das fängt auch den Fall „3a-Ära-v2-Checkpoint" (Version 1 MIT `physics_v2=True` und v1-Brains — den der Flag-Guard allein durchlassen würde) mit klarer Meldung ab.

**Files:**
- Modify: `artificial_society/simulation.py` (`CHECKPOINT_FORMAT_VERSION`, `_save_checkpoint`, `_load_checkpoint`)
- Modify: `tests/test_physics_v2_checkpoint.py` (NUR Ergänzungen — KEIN 3a-Test wird ersetzt oder geändert)

**Interfaces:**
- Consumes: `CheckpointIncompatibleError` (bestehend); `ACTION_SIZE_V2`, `OBJ_CTX_DIM` aus brain.py (Task 8).
- Produces: `CHECKPOINT_FORMAT_VERSION = 2` (Modul-Konstante); Payload-Key `"format_version"`; Loader-Reihenfolge: Version-Check → Flag-Check → Zuweisungen → Brain-Formen-Guard → `ensure_fields`.

- [ ] **Step 1: Tests schreiben (nur Ergänzungen)**

In `tests/test_physics_v2_checkpoint.py` bleiben alle fünf 3a-Tests unverändert. Neue Tests anhängen:

```python
def test_unbekannte_format_version_wird_hart_abgewiesen(checkpoint_path):
    """C5 (3b): Version ∉ {1, 2} ⇒ harter Fehler VOR dem broad-except, bei
    beiden Flag-Stellungen — kein stiller Frisch-Start (Datenverlust).
    Version 1 (Legacy, auch ohne Key) lädt dagegen weiter mit Flag aus —
    das pinnt der unveränderte 3a-Test test_alt_checkpoint_ohne_key_...."""
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"format_version": 3, "agents": [], "tick": 5}, f)
    with pytest.raises(CheckpointIncompatibleError, match="format_version"):
        Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)
    with pytest.raises(CheckpointIncompatibleError, match="format_version"):
        Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)


def test_3a_aera_v2_checkpoint_wird_klar_abgewiesen(checkpoint_path):
    """C5-Kern: v2-Checkpoint mit fremden Netz-Formen ⇒ klare Fehlermeldung
    statt stillem [compat]-Rebuild (Gewichtsverlust) oder Forward-Absturz.
    Konstruiert: Version-1-Payload MIT physics_v2=True, aber v1-Brains
    (3a-Ära-Fall) — der Flag-Guard allein würde ihn durchlassen."""
    quelle = Simulation(seed=3, physics_v2=False, load_checkpoint=False, **_PARAMS)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {"agents": quelle.agents, "tick": 3, "physics_v2": True, "world": quelle.world},
            f,
        )
    with pytest.raises(CheckpointIncompatibleError, match="Brain-Formen"):
        Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)


def test_payload_traegt_format_version_2(checkpoint_path):
    sim = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim._save_checkpoint()
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    assert data["format_version"] == 2
    assert data["physics_v2"] is True


def test_roundtrip_erhaelt_v2_brain_formen(checkpoint_path):
    """C5: v2-Brain-Formen überleben den Roundtrip (29 Köpfe, 192er GRU)."""
    sim1 = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim1.step()
    sim1._save_checkpoint()
    sim2 = Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)
    assert all(a.brain.physics_v2 for a in sim2.agents)
    assert all(a.brain.gru.input_size == 192 for a in sim2.agents)
    assert all(a.brain.policy_mean.out_features == 29 for a in sim2.agents)
    assert all(a.causal_model is not None for a in sim2.agents)
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest tests/test_physics_v2_checkpoint.py -q`
Expected: die 4 neuen Tests FAIL (`format_version` fehlt im Payload; unbekannte Version lädt still über den broad-except „frisch"; der 3a-Ära-v2-Payload wird per `[compat]`-Rebuild still repariert statt abgewiesen); die 5 unveränderten 3a-Tests PASS.

- [ ] **Step 3: Implementierung**

In `artificial_society/simulation.py`:

(a) Konstante neben `CHECKPOINT_PATH` (Zeile 38):

```python
CHECKPOINT_PATH = "checkpoint.pkl"
# C5 (Plan 3b): Format-Version des Pickle-Payloads. v2 führt neue Brain-Formen
# (29 Köpfe, 192er GRU), Causal-Model und Transition-Felder ein — Version 2.
# Fehlender Key ⇒ Version 1 (Legacy bis 3a): lädt WEITERHIN mit physics_v2=False
# (3a-Garantie — v1-Bestand bleibt resumierbar); unbekannte Versionen re-raisen
# hart statt still frisch zu starten.
CHECKPOINT_FORMAT_VERSION = 2
```

(b) In `_save_checkpoint` als ersten Payload-Key ergänzen:

```python
                pickle.dump(
                    {
                        "format_version": CHECKPOINT_FORMAT_VERSION,
                        "agents": self.agents,
```

(c) Import oben in `simulation.py` ergänzen (nach dem bestehenden `agents.agent`-Import-Block):

```python
from artificial_society.agents.brain import ACTION_SIZE_V2, OBJ_CTX_DIM
```

(d) In `_load_checkpoint` direkt nach `data = pickle.load(f)` und VOR dem `physics_v2`-Check einfügen:

```python
            version = int(data.get("format_version", 1))
            if version not in (1, CHECKPOINT_FORMAT_VERSION):
                raise CheckpointIncompatibleError(
                    f"checkpoint format_version={version} unbekannt "
                    f"(unterstützt: 1 = Legacy/3a, {CHECKPOINT_FORMAT_VERSION} = aktuell) — "
                    "Checkpoint löschen oder mit der passenden Code-Version laden"
                )
```

(e) Den Brain-Formen-Guard direkt NACH `self.agents = data.get("agents", [])` und VOR allen weiteren Zuweisungen/`ensure_fields` einfügen:

```python
            self.agents = data.get("agents", [])
            if self.physics_v2:
                # C5-Kern: v2-Brain-Formen-Guard VOR ensure_fields — sonst
                # würde der [compat]-Rebuild fremde Checkpoints still
                # „reparieren" (Gewichtsverlust) bzw. fremde v2-Formen später
                # im Forward stumm crashen. Fängt auch 3a-Ära-v2-Checkpoints
                # (Version 1 mit physics_v2=True und v1-Brains) klar ab.
                for agent in self.agents:
                    brain = getattr(agent, "brain", None)
                    if brain is None:
                        continue
                    if (
                        not getattr(brain, "physics_v2", False)
                        or brain.gru.input_size != 128 + OBJ_CTX_DIM
                        or brain.policy_mean.out_features != ACTION_SIZE_V2
                    ):
                        raise CheckpointIncompatibleError(
                            f"Agent {agent.id}: Brain-Formen passen nicht zur "
                            "v2-Architektur (Plan 3b: 29 Köpfe, 192er GRU) — "
                            "Checkpoint stammt aus einer anderen Code-Version; "
                            "löschen oder mit passender Version laden"
                        )
```

(Alle Guards laufen INNERHALB des try und damit VOR dem broad-except-Fallback — `CheckpointIncompatibleError` wird dort bereits explizit re-raised, Zeilen 443–444.)

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest tests/test_physics_v2_checkpoint.py tests/test_phase3_checkpoints.py -q`
Expected: alle grün (`9 passed` in test_physics_v2_checkpoint.py: 5 unveränderte 3a-Tests + 4 neue). `tests/test_phase3_checkpoints.py` braucht KEINE Anpassung mehr: Version-1-Payloads (ohne Key) laden mit Flag aus weiterhin normal; über `_save_checkpoint` erzeugte Roundtrips tragen Version 2. Sollte dort dennoch etwas rot werden, ist das ein Bug im Guard (Reihenfolge/Default prüfen) — NICHT die Fixtures ändern.

- [ ] **Step 5: Volle Suite + ruff + Commit**

```bash
rm -f checkpoint.pkl
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 307)
../venv/bin/python -m ruff check --fix artificial_society/simulation.py tests/test_physics_v2_checkpoint.py
../venv/bin/python -m ruff format artificial_society/simulation.py tests/test_physics_v2_checkpoint.py
git add artificial_society/simulation.py tests/test_physics_v2_checkpoint.py
git commit -m "feat(3b C5): CHECKPOINT_FORMAT_VERSION=2 — Legacy-v1 laedt weiter, Formen-Guard statt stillem Absturz"
```

### Task 17: D4-3b-Metriken-Tests + kurzer Suite-Smoke

Die drei „erst nach 3b"-Metriken-Familien (D4) sind in Tasks 14/15 implementiert worden (reines Logging, kein Verhalten): DiscoveryV2-Events je Agent, Verb-Raten aus der Policy, Neugier-Zerlegung (drei Quellen). Dieser Task beweist sie über `metrics_snapshot()` und ergänzt einen kurzen End-to-End-Smoke (12 Ticks) in der Suite.

**Files:**
- Test: `tests/test_physics_v2_brain_integration.py` (anhängen)

**Interfaces:**
- Consumes: `ObjectLayer.metrics_snapshot()` (3a), Metrik-Keys aus Tasks 14/15: `verbs_fired` (Versuche), `verbs_failed`, `verbs_noop`, `discovery_events_by_agent`, `curiosity_sums`.
- Produces: nichts Neues — Beweis-Tests.

- [ ] **Step 1: Failing/beweisende Tests schreiben**

An `tests/test_physics_v2_brain_integration.py` anhängen:

```python
def test_d4_metriken_verbs_und_discovery_je_agent():
    """D4 (erst nach 3b): Verb-Raten + DiscoveryV2-Events je Agent im Snapshot."""
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    agent.body.strength = 1.0
    hammer = make_object("granite", 1.5)
    flint = make_object("flint", 0.8)
    agent.hands.held.append(hammer)
    sim.world.objects.ledger["spawned"] += 1.5
    sim.world.objects.add(flint, agent.pos, source="spawned")
    view = build_slots(agent, sim.world.objects)
    _erzwinge_verb(agent, 2, view.objs.index(flint), tool_slot=view.objs.index(hammer))

    snap = sim.world.objects.metrics_snapshot()
    assert snap["verbs_fired"]["strike"] == 1  # Versuche (F4)
    assert snap.get("verbs_failed", {}).get("strike", 0) == 0  # erfolgreicher Schlag
    assert snap["discovery_events_by_agent"].get(agent.id, 0) >= 1, (
        "frische Fragmente sind neue Eigenschafts-Punkte → DiscoveryV2-Event je Agent"
    )


def test_d4_neugier_zerlegung_wird_geloggt():
    """D4: die drei Neugier-Quellen werden separat geloggt (Pilot-Diagnostik)."""
    sim = _sim_v2()
    for _ in range(3):
        sim.step()
    snap = sim.world.objects.metrics_snapshot()
    assert set(snap["curiosity_sums"]) == {"nextslot", "causal", "novelty"}
    assert all(np.isfinite(v) for v in snap["curiosity_sums"].values())
    assert snap["curiosity_sums"]["nextslot"] > 0.0  # Vorhersagefehler früh > 0
    for agent in sim.agents:
        assert set(agent.curiosity_last) == {"nextslot", "causal", "novelty"}


def test_v2_smoke_12_ticks_keine_nans():
    """Kurzer Suite-Smoke: 12 Ticks v2 — keine Exceptions, keine NaNs, Buffer wachsen."""
    sim = _sim_v2(seed=7)
    for _ in range(12):
        sim.step()
    for agent in sim.agents:
        assert np.isfinite(agent.last_reward)
        assert np.isfinite(agent.energy) and np.isfinite(agent.health)
        assert not torch.isnan(agent.hidden_state).any()
    assert sum(len(a.brain.rollout) for a in sim.agents) > 0
    haende = sum(
        a.hands.carried_mass_kg() for a in sim.agents if getattr(a, "hands", None) is not None
    )
    lhs, rhs = sim.world.objects.conservation_terms(held_mass_kg=haende)
    assert abs(lhs - rhs) < 1e-6, "Massen-Ledger hält auch unter Policy-Aktionen"
```

- [ ] **Step 2: Tests laufen lassen**

Run: `rm -f checkpoint.pkl && ../venv/bin/python -m pytest tests/test_physics_v2_brain_integration.py -q`
Expected: `9 passed`. (Diese Tests sind Beweis-Tests über in Task 14/15 gebaute Metriken — falls einer FAIL zeigt, ist das ein echter Bug in der Metrik-Verdrahtung: fixen, nicht den Test anpassen. Der Erhaltungs-Assert nutzt 1e-6 statt 1e-9, weil über `sim.step()` auch Verwesungs-Rundung über viele Objekte akkumuliert.)

- [ ] **Step 3: Volle Suite + ruff + Commit**

```bash
rm -f checkpoint.pkl
../venv/bin/python -m pytest -q          # exakte Zahl reporten (erwartet: 310)
../venv/bin/python -m ruff check --fix tests/test_physics_v2_brain_integration.py
../venv/bin/python -m ruff format tests/test_physics_v2_brain_integration.py
git add tests/test_physics_v2_brain_integration.py
git commit -m "test(3b D4): Metriken-Beweise — Verb-Raten, DiscoveryV2 je Agent, Neugier-Zerlegung + 12-Tick-Smoke"
```

---

### Task 18: Abschluss — 500-Tick-v2-Rauchlauf (manuell, nicht committen)

Policy random-initialisiert; erwartbar: Aktionen feuern, keine Exceptions, Neugier-Logging plausibel, kein NaN in Losses. Läuft lokal (CPU-Brains, ~2–5 min; Bash-`timeout` auf 600000 setzen). Kein Commit — der Lauf ist Verifikation, das Skript bleibt Wegwerf-Code im Scratchpad.

**Files:** keine (Skript im Scratchpad, z. B. `/private/tmp/.../scratchpad/rauchlauf_3b.py`).

- [ ] **Step 1: Rauchlauf ausführen**

```bash
cd /Users/moritzbecker/projekt/as-lern-kopplung-3b && rm -f checkpoint.pkl
../venv/bin/python - <<'PY'
import math

import numpy as np
import torch

from artificial_society.simulation import Simulation

sim = Simulation(
    headless=True, load_checkpoint=False, physics_v2=True,
    seed=7, grid_w=30, grid_h=20, initial_population=10,
)
losses = []
for t in range(500):
    sim.step()
    for a in sim.agents:
        if a.last_loss:
            losses.append(a.last_loss)

snap = sim.world.objects.metrics_snapshot()
verbs = snap.get("verbs_fired", {})
cur = snap.get("curiosity_sums", {})
haende = sum(a.hands.carried_mass_kg() for a in sim.agents if a.hands is not None)
lhs, rhs = sim.world.objects.conservation_terms(held_mass_kg=haende)

print("population:", len(sim.agents))
print("verbs_fired:", verbs)
print("verbs_noop:", snap.get("verbs_noop", {}))
print("verbs_failed:", snap.get("verbs_failed", {}))
print("discovery_je_agent:", snap.get("discovery_events_by_agent", {}))
print("curiosity_sums:", {k: round(v, 2) for k, v in cur.items()})
print("ledger:", {k: round(v, 3) for k, v in snap["ledger"].items()})
print("trainings:", len(losses), "letzte losses:", [round(x, 4) for x in losses[-5:]])
print("ledger_delta:", abs(lhs - rhs))

assert verbs, "mindestens ein Verb muss in 500 Ticks gefeuert haben (Init-Bias +0.2)"
assert losses, "PPO muss mindestens einmal trainiert haben (128er-Buffer)"
assert all(math.isfinite(x) for x in losses), "kein NaN/Inf in den Losses"
assert all(math.isfinite(v) for v in cur.values()) and cur, "Neugier-Logging plausibel"
assert all(np.isfinite(a.last_reward) for a in sim.agents)
assert all(not torch.isnan(a.hidden_state).any() for a in sim.agents)
assert abs(lhs - rhs) < 1e-4, f"Massen-Ledger driftet: {lhs} != {rhs}"
print("RAUCHLAUF OK")
PY
rm -f checkpoint.pkl
```

Expected (letzte Zeile): `RAUCHLAUF OK`, davor plausible Zähler — `verbs_fired` mit mehreren Verben > 0, `trainings ≥ 10`, `curiosity_sums` alle drei Quellen > 0 (`causal` kann klein sein — es zahlt nur nach aufgelösten Pendings), `ledger_delta` ~1e-9-Bereich.

- [ ] **Step 2: Befund reporten + Golden-Schlussprobe**

```bash
rm -f checkpoint.pkl
../venv/bin/python -m pytest -q          # finale exakte Zahl reporten (erwartet: 310)
git status --short                       # muss leer sein bis auf docs/superpowers/plans/ (Plan NICHT committen)
```

Die Zahlen des Rauchlaufs (Verb-Zähler, Trainings, Neugier-Summen, Ledger-Delta) im Abschluss-Report festhalten — sie sind die Baseline-Diagnostik für den Pilot (Plan 5).

---

## Hinweise für die Ausführung

- **Reihenfolge ist bindend** (spätere Tasks konsumieren frühere Interfaces); insbesondere: Task 14 vor 15 (Reward-Umbau ersetzt die Task-14-Übergangsverdrahtung), Task 13 vor 15 (`finalize_terminal`).
- **Erwartete Suite-Zahlen** („erwartet: N") sind Sollwerte aus der Task-Planung; maßgeblich ist: KEIN Test rot, Zahl monoton wachsend, exakte Zahl im Report. Weicht die Zahl ab (z. B. weil ein Test parametrisiert ist), Zahl reporten und weiter.
- **Golden rot = Stopp.** Niemals den Golden-/Digest-Test anpassen; die Änderung hat dann v1-Verhalten berührt (meist: ein RNG-Draw außerhalb eines `physics_v2`-Gates).
- Läuft ein Schritt in einen echten Spec-Konflikt, nicht raten — Task stoppen und die Frage im Report eskalieren.






