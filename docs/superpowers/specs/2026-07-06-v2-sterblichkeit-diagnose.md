# v2-Sterblichkeit — Diagnose (v1-vs-v2-Demografie-Vergleich)

Datum: 2026-07-06
Datenbasis: `sweep_v1v2/{v1,v2}_{baseline,best}/*.jsonl` (je 4 Seeds, `nolearn`,
5000 Ticks, snapshot_interval 250). Konfigs:
- `*_baseline`: 40×30, pop30, min-pop 8, respawn-count 6, keine Knobs.
- `*_best`: 24×18, pop??, min-pop 16, respawn-count 8, age-structured,
  plant-ceiling-scale 3.4.

Code READ-ONLY geprüft; Zahlen aus den Roh-JSONL gerechnet.

---

## TL;DR

1. **Die „v2-Übersterblichkeit" ist ein Messartefakt, kein echtes Signal.** Das
   `deaths`-Feld liest für v1 baukonstruktiv **exakt 0** über den ganzen Lauf,
   weil der v1-Pfad Tote **vor** dem gezählten `remove_dead`-Wrapper wegfiltert
   (`simulation.py:626`). Pfad-unabhängige Sterblichkeits-Proxies (`respawns`,
   `ids_seen_total`, `births_cum`) zeigen in v1 und v2 **nahezu identischen
   Turnover** und **identische Populationskurven**.
2. Die **einzige echte v1/v2-Mechanik-Differenz** im Energiehaushalt ist ein
   **Nahrungsquellen-Gap** (nicht eine Kosten-Kalibrierung): v2 zwingt alle
   Agenten in reines Pflanzen-Foraging und sperrt den Fleisch-/Aas-Kalorienpfad,
   den in v1 ~50 % der Agenten (diet ≥ 0) nutzen. Metabolische Kosten
   (`move_cost` etc.) sind **bit-identisch** (gemeinsamer Code vor der
   `physics_v2`-Verzweigung).
3. Dieser Gap manifestiert sich in den Daten **nicht** als Demografie-Divergenz,
   weil **beide** Pfade ohnehin am Respawn-Boden kleben (`emergency_respawn`
   trägt die Population, nicht die Geburten). Der Boden maskiert jeden echten
   Unterschied.

---

## 1. Messvalidität (kritisch, zuerst)

### 1.1 Der Widerspruch ist real und aufgeklärt

Beobachtung im Sweep: v1 zeigt `deaths_H2 ≈ 0` bei gleichzeitig `respawns_H2 > 0`.
Respawns setzen voraus, dass die Population unter `MIN_POPULATION` fiel — also
müssen Agenten gestorben sein. Beides zusammen ist nur erklärbar, wenn der
`deaths`-Zähler v1 nicht erfasst.

**Ursache — Datei:Zeile:**

- Der `deaths`-Zähler ist ein Wrapper auf `sim.remove_dead`
  (`scripts/m1_pilot.py:677-683`):
  ```py
  def _counting_remove_dead():
      before = len(sim.agents)
      result = _orig_remove_dead()
      _death_state["count"] += before - len(sim.agents)
  ```
  Er zählt also, wie viele Agenten **genau in `remove_dead`** aus `sim.agents`
  verschwinden.
- Im **v1-Pfad** filtert `Simulation.step()` Tote aber **schon vor**
  `remove_dead` weg (`simulation.py:625-627`):
  ```py
  if not self.physics_v2:
      self.agents = [a for a in self.agents if a.alive]   # 626
  self.remove_dead()                                       # 627
  ```
  Wenn der Wrapper läuft, sind die Toten bereits raus → `before - len == 0`.
  Der `remove_dead`-Rumpf (`simulation.py:313-333`) sieht nur noch Überlebende.
  ⇒ **`deaths` ist in v1 strukturell 0, unabhängig von der tatsächlichen
  Sterblichkeit.**
- Im **v2-Pfad** ist `simulation.py:625` False (kein Pre-Filter). Tote erreichen
  `remove_dead` (bewusst, wegen Kadaver-Objekt/Erhaltung, Spec B3) → der Zähler
  erfasst v2 korrekt.

**Bestätigung in den Daten** (`deaths`-Feld, kumuliert, ganzer Lauf):

| cfg | deaths (Feld) |
|---|---|
| v1_baseline | **0.0** |
| v1_best | **0.0** |
| v2_baseline | 194.8 |
| v2_best | 418.0 |

Exakt 0 in **beiden** v1-Konfigs über 5000 Ticks bei nicht-trivialer Population
ist physisch unmöglich und beweist den Artefakt-Charakter.

### 1.2 Robuste, pfad-unabhängige Proxies

Diese Zähler haben **keine** `physics_v2`-Verzweigung und sind daher
cross-path-vergleichbar:

- `respawns` = Aufrufe von `emergency_respawn`; Schwellen-Check
  `simulation.py:639` ist ungated.
- `ids_seen_total` = |{alle je in einem Snapshot gesehenen `Agent.id`}|
  (`m1_pilot.py:546`); wächst mit jedem neuen (respawnten/geborenen) Agenten.
- `births_cum` = Wrapper auf `spawn_child_from_parent` (`m1_pilot.py:684-692`),
  pfad-unabhängig.

**H2 (Tick 2500→5000), Mittel über 4 Seeds:**

| cfg | pop@5000 | respawn-Calls | respawnte Agenten | births | **ids_seen (Turnover)** | mean_age | mean_E |
|---|---|---|---|---|---|---|---|
| v1_baseline | 10.5 | 9.8 | 58.5 | 18.2 | **71.5** | 227.0 | 93.7 |
| v2_baseline | 10.2 | 11.5 | 69.0 | 16.8 | **77.5** | 217.3 | 75.4 |
| v1_best | 20.0 | 22.0 | 176 | 18.5 | **171.5** | 156.5 | 56.3 |
| v2_best | 19.8 | 23.0 | 184 | 31.8 | **178.8** | 126.2 | 65.1 |

In einer pop-stabilen Phase gilt Zufluss = Abfluss, d. h.
**Gesamttode ≈ ids_seen (H2)**. Diese liegen v1 vs v2 nur ~4–8 % auseinander —
also **nahezu identisch**, während das `deaths`-Feld einen Faktor ∞ vortäuscht.

**Ganzer Lauf (Populationsdynamik):**

| cfg | min_pop-Setting | popMean | minPop erreicht | meanE (Traj) |
|---|---|---|---|---|
| v1_baseline | 8 | 14.0 | 8.0 | 84.7 |
| v2_baseline | 8 | 13.9 | 8.0 | 68.4 |
| v1_best | 16 | 20.4 | 16.0 | 57.8 |
| v2_best | 16 | 21.2 | 16.0 | 61.0 |

Die Populationskurven sind **ununterscheidbar** (v2_best liegt sogar minimal
höher). Beide Pfade fallen exakt auf das jeweilige `min_pop` (8 bzw. 16) und
werden von `emergency_respawn` gehalten — nicht von Geburten (Respawn-Agenten
58–184 ≫ Geburten 18–32 in H2).

### 1.3 Fazit Messvalidität

> **Die v2-Übersterblichkeit ist praktisch vollständig ein Messartefakt.** Das
> `deaths`-Feld ist für v1 durch den Pre-Filter (`simulation.py:626`) blind;
> die robusten Proxies zeigen identischen Turnover und identische
> Populationskurven. Die im Auftrag vermutete Erwartung („echtes Signal") wird
> von den Daten **nicht** gestützt. Für jede weitere Demografie-Auswertung ist
> `deaths` unbrauchbar — nutze `ids_seen_total`/`respawns`/`births_cum`.

Auch `mean_age` ist nur ein schwaches, kein tragendes Signal: v2_best 126 <
v1_best 156, aber v2_best hat gleichzeitig **mehr** Geburten (31.8 vs 18.5).
Jüngerer Altersschnitt = schnellere Generationenfolge bei mehr Nachwuchs, nicht
höhere Sterberate — die Turnover-Summe (ids_seen) bleibt gleich.

---

## 2. Was unterscheidet den v2-Energiehaushalt real?

Da die Demografie identisch ist, lautet die präzise Frage: Wo weicht der
v2-Energiehaushalt mechanisch von v1 ab (auch wenn der Boden es maskiert)?

### 2.1 Metabolische Kosten sind identisch (kein Kalibrierungs-Gap)

`move_cost`, `hydration_loss`, der `energy<=0 → health-Drain` und der Hungertod
stehen in `agent.py:1325-1361` **vor** der `physics_v2`-Verzweigung und laufen in
beiden Pfaden gleich:
```py
move_cost = 0.5 * mods["move_cost_mult"] * stage["move_cost_mult"] * cold_factor  # 1325
self.energy = max(0.0, self.energy - move_cost)                                    # 1335
...
if self.energy <= 0:  self.health -= 1.5     # 1354
if self.health <= 0:  self.alive = False     # 1359  ← Hungertod
```
Die Embodiment-Kosten (`carry_tick`-Fatigue `body.py:71-76`, `strike`) erhöhen
**Fatigue**, nicht direkt `energy`. ⇒ **Auf der Ausgabenseite gibt es keinen
v2-spezifischen Mehrverbrauch.** Die Kandidaten-Hypothese „Embodiment-
Energiekosten ∝ Masse töten v2" ist widerlegt.

### 2.2 Der echte Gap: v2 hat weniger Nahrungsquellen (Mechanik-Gap)

`_forage` (`agent.py:611-671`):
```py
diet = self.genes.get("diet_preference", 0.0)       # 611  (uniform(-1,1), genetics.py:55)
if diet < 0 or self.physics_v2:                       # 618  ← v2 IMMER Pflanzen-Zweig
    # nur plant_food-Zellpool (PLANT_ENERGY=30)
else:
    # carcasses (CORPSE-Pool) → meat_food (MEAT_ENERGY=45) → plant
```
Folgen für v2:
- **~50 % der Agenten verlieren ihren Fleisch-/Aas-Pfad.** `diet_preference` ist
  uniform(−1, 1) (`genetics.py:17,55`); in v1 essen die ~50 % mit diet ≥ 0
  zusätzlich Kadaver-Zellpool + `meat_food` (45 vs 30 Energie/Portion). In v2 ist
  dieser Zweig für **alle** gesperrt (`agent.py:618`).
- **Der Tod-→Nahrung-Recyclingpfad ist in v2 gekappt.** In v1 kreditiert
  `remove_dead` den Kadaver in den **Zellpool**:
  `add_carcass(world, *pos, CORPSE_ENERGY=36)` (`simulation.py:332`) — sofort
  per `_forage` von diet-≥0-Agenten essbar. In v2 wird der Tod stattdessen zu
  **genau einem Kadaver-OBJEKT** (`_spawn_carcass`, `simulation.py:331,335-349`),
  und die Fleisch-Zellpools sind aus (Spec B6). Dessen Energie ist nur über die
  verkörperte **Schneiden→Essen-Kette** zugänglich, die eine Zufallspolicy
  (`nolearn`) praktisch nie ausführt (`cuts_with_tool`/`cuts_bare_hand` in den
  Snapshots = 0). ⇒ Die aus Toten recycelte Energie geht der zugänglichen
  Nahrungsversorgung in v2 verloren.

### 2.3 Grober Netto-Energiehaushalt pro Tick

Konstanten (`agent.py:68-99`): MAX_ENERGY 240, PLANT_ENERGY 30, MEAT_ENERGY 45,
CORPSE_ENERGY 36, move_cost ≈ 0.5/Tick.

- **Ausgaben** (v1 = v2): ~0.5 Energie/Tick Grundbewegung + Hydration/Alter/
  Krankheit-Drains → Größenordnung ~0.5–1.5/Tick.
- **Einnahmen v2-Agent:** nur wenn `action.forage` feuert **und** die Zelle
  `plant_food` > 0 hat → bis zu 30·eff. Ein einziger Pflanzen-Bissen deckt also
  ~20–60 Ticks Grundkosten. Der begrenzende Faktor ist die **Pflanzen-
  Tragfähigkeit** (Regrow/Ceiling), nicht die Kosten.
- **Einnahmen v1-Agent (diet ≥ 0, ~50 %):** zusätzlich Kadaver-Pool (36) und
  `meat_food` (45) — de facto **~1,5× breitere Kalorienbasis** und ein
  geschlossener Tod→Nahrung-Kreislauf.

⇒ v2 hat eine **strikt kleinere zugängliche Kalorienbasis** als v1. Dass sich das
demografisch nicht zeigt, liegt allein am Respawn-Boden: `emergency_respawn`
injiziert frische Agenten schneller, als das Nahrungsdefizit die Population unter
`min_pop` drücken kann. Ohne diese Krücke würde v2 vermutlich ein reales Defizit
zeigen (siehe §3, Lever 3 = Entscheidungsexperiment).

**Dominante „Ursache" — Einordnung:** Es gibt keine v2-spezifische *Todes*ursache
im Sinne von Übersterblichkeit (die existiert nicht). Der reale, belegbare
v2/v1-Unterschied ist ein **Nahrungsquellen-Mechanik-Gap**
(`agent.py:618` + `simulation.py:331`), der aktuell demografisch maskiert ist.

---

## 3. Priorisierte Stellschrauben

### Lever 1 (P0, MUSS zuerst) — Instrument reparieren, nicht die Sim
**Problem:** `deaths` ist für v1 blind → jeder v1/v2-Demografievergleich über
`deaths` ist korrupt.
**Fix (rein Harness, kein Core):** In `scripts/m1_pilot.py` die Sterblichkeit
pfad-unabhängig zählen — z. B. `Simulation.step` wrappen und
`sum(1 for a in sim.agents if not a.alive)` **vor** dem Pre-Filter
(`simulation.py:626`) erfassen, oder den bereits validen `ids_seen`-basierten
Turnover als kanonische Sterblichkeitsmetrik ausweisen und `deaths` als
„v1-invalid" markieren.
**Wertebereich:** n/a (Instrumentierung).
**Erwarteter Effekt:** v1- und v2-Sterblichkeit werden erstmals vergleichbar;
bestätigt (Prognose) die Turnover-Gleichheit aus §1.2. **Ohne Lever 1 sind
Lever 2/3 nicht sauber messbar.**

### Lever 2 (P1) — Nahrungsquellen-Gap adressieren
Zwei Varianten, bewusst getrennt nach *Mechanik* vs *Kalibrierung*:

**2a — Mechanik-Fix (HOT-Core-FLAG):** Den v2-Zwang zu Pflanzen-only lockern
bzw. den Tod→Nahrung-Kreislauf wieder schließen.
- Stelle: `agent.py:618` (`if diet < 0 or self.physics_v2:`) — den
  `or self.physics_v2` entfernen/flaggen, **oder** in `_spawn_carcass`
  (`simulation.py:331`) zusätzlich einen Zellpool-Kredit (analog
  `add_carcass(..., CORPSE_ENERGY)`, `simulation.py:332`) für v2 setzen.
- Nicht harness-patchbar (Kernpfad) → als **HOT-Core-FLAG** an
  `agent.py:611-618` / `simulation.py:331-332` markieren.
- **Erwarteter Effekt:** stellt die v1-Kalorienbreite in v2 her; hebt (Prognose)
  Geburten und Population über den Boden, wenn v2 food-limitiert ist.

**2b — Kalibrierungs-Workaround (rein Harness):** Pflanzen-Tragfähigkeit für v2
anheben, um die fehlende Fleisch-Linie zu kompensieren. Knob existiert bereits:
`--plant-ceiling-scale` (`m1_pilot.py`, skaliert `world._bio["plant_ceiling"]`).
- **Wertebereich:** 1.0 → {1.5, 2.0, 2.5} (v2_baseline hat 1.0; v2_best schon
  3.4 — dort ggf. 3.4 → {4.5, 6.0}).
- **Erwarteter Effekt:** wenn v2 food-limitiert, steigen Geburten und mean_energy
  und die Population löst sich vom `min_pop`-Boden.

### Lever 3 (P1, Entscheidungsexperiment) — Respawn-Krücke entfernen
**Zweck:** Der Respawn-Boden maskiert jeden echten v1/v2-Unterschied. Nur ohne
ihn zeigt sich, ob der Nahrungs-Gap real beißt.
**Stellschraube (rein Harness, pfad-unabhängig):** `--min-pop` und
`--respawn-count` klein setzen, z. B. `--min-pop 2 --respawn-count 1` (statt
8/6 bzw. 16/8).
**Erwarteter Effekt / Lesart:**
- Laufen v1 und v2 auch ohne Boden gleich → bestätigt endgültig „kein reales
  v2-Defizit" (Gap ist folgenlos).
- Fällt v2 unter v1 → der Nahrungsquellen-Gap aus §2.2 ist real und Lever 2 ist
  die Priorität.

**Kein Lever nötig für „Kostenkonstante runter":** Es gibt keinen v2-spezifischen
Mehrverbrauch (§2.1) — die reine Kosten-Kalibrierung ist gegenstandslos.

---

## Anhang — Datei:Zeile-Index

- `scripts/m1_pilot.py:677-683` — `deaths`-Wrapper (misst nur `remove_dead`-Delta)
- `scripts/m1_pilot.py:659-666` — `respawns`-Wrapper (pfad-unabhängig)
- `scripts/m1_pilot.py:684-692` — `births_cum`-Wrapper (pfad-unabhängig)
- `scripts/m1_pilot.py:546` — `ids_seen_total`-Aufbau (pfad-unabhängig)
- `artificial_society/simulation.py:625-627` — v1-Pre-Filter vor `remove_dead` (Artefakt-Ursache)
- `artificial_society/simulation.py:313-333` — `remove_dead` (v2 Kadaver-Objekt, v1 `add_carcass`)
- `artificial_society/simulation.py:331-349` — `_spawn_carcass` (v2: Objekt statt Zellpool)
- `artificial_society/simulation.py:639` — `MIN_POPULATION`-Check → `emergency_respawn` (ungated)
- `artificial_society/agents/agent.py:611,618` — `_forage`: v2 erzwingt Pflanzen-only
- `artificial_society/agents/agent.py:1325-1361` — Metabolik/Hungertod (gemeinsam v1/v2)
- `artificial_society/agents/agent.py:68-99` — Energiekonstanten
- `artificial_society/agents/genetics.py:17,55` — `diet_preference` uniform(−1,1)
- `artificial_society/environment/physics/body.py:71-76` — `carry_tick` (Fatigue, nicht Energie)
