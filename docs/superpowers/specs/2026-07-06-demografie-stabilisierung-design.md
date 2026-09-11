# Demografie-Stabilisierung — Design (Stabilisierungs-Pass vor M1-Pilot)

**Datum:** 2026-07-06 · **Branch:** `integration/pop-fix` (HEAD 263118f) · **Lane:** core (HOT: agent.py, ggf. world.py/resources.py)
**Vorgeschichte:** Kalorien-Fix (AK/AL, APPROVE_WITH_MINORS) hat den Hungertod behoben; die Population ist aber
**nicht selbsttragend** — entscheidender Sweep (AM): ohne Respawn-Krücke 4/4 extinct, Mechanismus = **Malthus
Boom-Bust**. Bestätigung (AN): Partner-Taxis netto schädlich (opt-in, bleibt aus). Instrumentierung (AO,
`scratchpad/overshoot-mechanism.md`) hat die Overshoot-Ursache kausal festgenagelt.

## Problem (instrumentell bestätigt)

Zwei verkettete Treiber erzeugen den Overshoot 30→~55 @t≈200 → Massenhunger → Crash:

1. **Synchroner Gründer-Puls.** Alle 30 Gründer starten fertil (`reproduction_cooldown==0`); **54–64 % aller
   Konzeptionen fallen in Ticks 0–149**, Peak bei Tick 0. `--age-structured-founders` desynchronisiert NICHT
   (setzt nur `age`/`birth_tick`, nie den Cooldown; >97 % der Gründer überschreiten `MIN_REPRODUCTION_AGE=60`
   ohnehin). Der als Overshoot-Fix gedachte Flag ist gegen den Puls wirkungslos.
2. **Fruchtbarkeits-Gate auf momentanem Bestand.** `_local_food_per_capita` (agent.py:988) summiert den
   *aktuellen* `cell["food"]` und teilt durch Münder. Bei t0 steht die Welt auf vollem Bestand → per-capita
   97–102 gegen Floor 8 → das ganze fertile Kohorte konzipiert auf einem **einmaligen, nicht erneuerbaren
   Startpuffer**, frisst ihn in ~50 Ticks leer, dann trägt der Regrowth-**Fluss** (nur ~3–5 food/tick fürs ganze
   Gitter) den Nachwuchs nicht → Hungerwelle.

**Diskriminierendes Signal (AO-C):** carrying-capacity-per-capita ≈575 (quasi-statisch, bindet nie — NICHT
verwenden). **Regrowth-Fluss-pro-Mund**: 0,08 bei pop 55 vs 0,40 bei pop 12 — trennt Overshoot sauber von
gesunder Niedrigdichte-Erholung. Die Erholung bei niedriger Dichte MUSS erhalten bleiben (sie rettet seed41).

## Änderungen (drei, koordiniert)

### C1 — Gründer-Cooldown-Staffelung (billig, klar korrekt, complement für das t0-Fenster)

Bei der Gründer-Initialisierung jedem Gründer einen **initialen `reproduction_cooldown`** ziehen, gleichverteilt
über `[0, REPRODUCTION_COOLDOWN)` (=[0,99]), damit nicht alle bei t0 gleichzeitig feuern. Das zerlegt den
einmaligen synchronen t0-Puls in einen gleichmäßigen Zufluss — genau das Fenster, das C2 am schlechtesten
abdeckt (bei voller Welt ist der Fluss ≈0, das Fluss-Gate also bei t0 uninformativ).

- **Ort (Review I-2, BINDEND):** die **Harness**-Funktion `scripts/m1_pilot.py:_age_structure_founders`, direkt
  nach dem bestehenden `a.age`/`a.birth_tick`-Draw. Das ist der Pfad, den `--age-structured-founders` bedient;
  er mutiert bereits konstruierte Agenten NACH dem `Simulation(...)`-Bau. **KEIN Edit an `simulation.py`/`agent.py`**
  (das würde den Golden riskieren; die Harness importiert weder `compute_trajectory` noch der Digest-Test sie).
- **Determinismus (Review I-1, KORRIGIERT):** aus demselben **globalen** `random`-Strom ziehen, den die
  Schwester-Zeile schon nutzt: `a.reproduction_cooldown = random.randint(0, REPRODUCTION_COOLDOWN - 1)`.
  `rng.seed_all` IST `random.seed(...)` (die globalen Generatoren sind die seed-Quelle der Wahrheit); bare
  `random` NACH `seed_all` ist hier kanonisch und deterministisch. KEIN zweites RNG-Objekt anlegen.
- **Golden-Sicherheit:** Die Harness liegt außerhalb des Golden-/Digest-Pfads → C1 kann beide Artefakte unter
  keiner Gating nicht berühren. Die Flag-Kopplung ist Gürtel-und-Hosenträger, der echte Grund ist die Lage.

### C2 — Fruchtbarkeits-Gate auf erneuerbare Versorgung-pro-Mund umstellen (Kern-Hebel)

Das Konzeptions-Gate (agent.py:1003) von *momentanem Bestand-pro-Mund* auf *nachhaltige Erneuerung-pro-Mund*
umstellen. Der Zähler muss (a) t0-robust (nicht ≈0 bei voller Welt), (b) dichte-diskriminierend und (c)
bestands-unabhängig sein (darf nicht vom grasbaren Startpuffer getäuscht werden).

**ENTSCHIEDEN (Design-Review): Option A — geglätteter realisierter Regrowth-EMA.** (B = `soil_fertility·moisture`
ist ein umskaliertes quasi-statisches Carrying-Capacity-Signal — genau das, wovor AO warnte: die Diskriminierung
läge fast ganz auf `/Münder`. A behält eine zweite, physisch-reale Achse: der realisierte Fluss wird bei Dichte
über `stress_factor` (resources.py:536) gedämpft und erholt sich, wenn die Dichte fällt → A **entspannt doppelt**
bei niedriger Dichte, die sicherere Wahl für den Erhalt der Erholung.) Das Routing-Argument gegen A war falsch:
das Feld kommt via `cell_store.FLOAT_FIELDS` (NICHT HOT) + `initial_cell_state`/`regrow_grid` (env-Lane) — **kein
`world.py`-Edit**; der Headless-Digest hasht `world.F` nie → das Feld kann ihn nicht brechen.

**Exakte Formel:**
- env-Lane `environment/resources.py`: `"plant_renewal_ema"` in `initial_cell_state(...)` + in `cell_store.FLOAT_FIELDS`
  (cell_store.py:202,209 iteriert diese Tuple in `build_arrays`). In `regrow_grid` NACH dem finalen `plant_gain`
  (post-headroom/post-biome, ~:569–579, vor dem `F["plant_food"]`-Write :599):
  ```
  ALPHA = 0.05    # ~20-Tick-Gedächtnis; sweepbar
  F["plant_renewal_ema"] = ALPHA * plant_gain + (1.0 - ALPHA) * plant_renewal_ema0
  ```
  mit `plant_renewal_ema0 = F["plant_renewal_ema"].copy()` wie die anderen `*0`-Snapshots (:507–520).
- **t0-Seed (löst „Fluss≈0 bei vollem Bestand"):** den EMA mit dem *pre-headroom*-Potenzial `plant_gain`
  (resources.py:563, VOR dem `* plant_headroom` :569) initialisieren — deterministisch, KEIN RNG.
- core/HOT `agents/agent.py` `_try_reproduce` Gate — **MUSS `physics_v2`-gegated sein (Review C-1, KRITISCH):**
  ```
  if self.physics_v2:
      r = REPRODUCTION_SENSE_RADIUS
      renewal = Σ world.get_cell(cx,cy)["plant_renewal_ema"] über die (2r+1)²-Box
      mouths  = len(self._nearby_cached(agents, r)) + 1.0
      if renewal / mouths < REPRODUCTION_MIN_RENEWAL_PER_CAPITA:
          return None
  else:
      if self._local_food_per_capita(world, agents) < REPRODUCTION_MIN_FOOD_PER_CAPITA:
          return None      # v1-Pfad UNVERÄNDERT, byte-für-byte
  ```

- **Floor-Kalibrierung:** neuer Floor in anderer Skala (Fluss food/tick, nicht Bestand) → per Sweep einstellen
  (Ziel: bindet bei pop≳20, öffnet bei pop≲12 wo der gemessene Fluss/Mund ≈0,4 ist, damit die Erholung lebt).
  **Neuer Harness-Flag `--min-renewal-per-capita`** (Review M-2: NICHT `--min-food-per-capita` repurposen — das
  regelt weiter das v1-Gate + den v2-Fallback).
- **Fallback (nur falls core-lead ein neues Feld vetoed):** refined-B `Σ carrying_capacity·(0.4+soil_fertility/100)
  /(Münder+1)` (cap·fert ohne Moisture-Rauschen, agent.py-only), ebenfalls `physics_v2`-gegated.
- **Kadaver-Subvention:** das Gate zählt nur Pflanzen-Erneuerung; die (jetzt großzügige) Kadaver-Kalorie geht
  bewusst NICHT ins Fruchtbarkeits-Gate ein — Fleisch ist ein Puls (Tod→Nahrung), keine nachhaltige Basis;
  Reproduktion soll an der erneuerbaren Pflanzen-Basis hängen. (Falls A/B später zeigt, dass Fleisch-Subvention
  den Overshoot wieder anheizt → Biss-Kadenz drosseln, separater Knopf.)

### C3 — Kalorien-Biss auf `carcass`-only (behebt HOT-Review-Befund I1, A/B-Gradient-Risiko)

Der innate Kadaver-Handbiss (agent.py:792) akzeptiert aktuell `("carcass","raw_meat")`. `raw_meat` ist das
Produkt **gelernter** Zerlegung und trägt Toxin, das der **gelernte** eat-Verb zahlt — der toxin-freie innate
Biss dominiert ihn also strikt und droht den learn>nolearn-Gradient des M1-A/B zu verflachen. Fix: Zeile 792 auf
`("carcass",)` einschränken. Der innate Biss bleibt der v1-Aas-Zellpool-Ersatz (Kadaver, kein Toxin); jedes
`raw_meat` muss durch den gelernten eat-Verb (mit Toxin/Risiko) → Gradient bleibt erhalten.

## Determinismus & Contract

- **v1-Golden byte-identisch, KEIN Rebake (Review C-1 + I-3):** C1 liegt in der Harness (außerhalb des
  Golden-/Digest-Pfads). C3 ist bereits `physics_v2`-gegated (:789). C2 ist NICHT von Natur aus v2-only — das
  Gate `_try_reproduce` ist pfadunabhängig — daher MUSS C2 explizit `if self.physics_v2:`-gegated werden (v1
  behält das exakte Bestand-Gate). Damit ändert sich der v1-Default-Pfad um NULL → v1-Golden + Headless-Digest
  bleiben grün OHNE Rebake. Es existiert **kein v2-Golden-Artefakt** → nichts zu rebaken; die absichtliche
  v2-Verhaltensänderung wird nur vom Verifikations-Sweep erfasst. C1-Zufall: globaler `random`-Strom (I-1).
- Brain-Contract (act_v2, store_transition_v2, Slots, Masken) unberührt — C1/C2/C3 fassen nur Reproduktions- und
  Forage-Logik an, keine Aktionsräume. Kein off-policy-Bias (nichts überschreibt eine gesampelte Aktion).
- **§4b:** Reproduktion/Fruchtbarkeit/Gründer-Init sind **Basis-Mechanik** (wie Essen/Altern) — legitim eingebaut,
  KEINE gescriptete „Fähigkeit". Es wird keine Navigation/Sprache/Werkzeug-Nutzung gescriptet.

## Test-Plan (TDD)

- **C1:** (a) mit `--age-structured-founders` sind die initialen Cooldowns über [0,100) gestreut (nicht alle 0);
  (b) Determinismus: gleicher Seed → identische Cooldown-Folge; (c) ohne Flag: alle Cooldowns 0 (Default/v1
  unverändert). (d) Digest-/Golden-Test grün.
- **C2:** (a) hohe Dichte < Floor (Gate schließt); (b) **niedrige Dichte ≥ Floor — Gate OFFEN** (Review M-1,
  BINDEND: sperrt den Floor gegen Erholungs-Unterdrückung; bei pop≈12/erholtem Fluss ≈0,4 muss konzipiert werden
  können); (c) t0-Robustheit: bei voller Welt/pop30 Signal < Floor (kein Overshoot-Freibrief), NICHT ≈0; (d)
  **v1-Neutralität (Review M-3/C-1): mit `physics_v2=False` läuft weiter das Bestand-Gate, `plant_renewal_ema`
  wird NIE gelesen**; (e) Determinismus.
- **C3:** (a) innate Biss auf `raw_meat`: KEIN innate Biss (fällt durch zum gelernten Pfad); (b) auf `carcass`:
  Biss wie bisher; (c) Massenerhaltung/energy-only-Guards grün (bestehende `test_v2_carcass_gnaw.py`).
- **Golden/Digest:** v1 byte-identisch + Headless-Digest grün OHNE Rebake verifizieren. Review M-4: prüfen, dass
  kein v1-Lesepfad (means/diffuse `world.py:309–315`, Stats, Checkpoint) `plant_renewal_ema` so liest, dass es
  die v1-Trajektorie stört.

## Verifikation (nach Impl)

Entscheidender Sweep neu (respawn OFF, notaxis, seeds 41–44, 5000 Ticks, Grid 24×18, pop 30,
`--age-structured-founders`): Ziel = **robuste Selbsttrag** (≥3/4 Seeds pop>Floor bei t5000, births/deaths→~1,
kein Extinct), Overshoot-Peak gedämpft (pop-Peak <40 statt ~55). Floor-Kalibrier-Miniserie falls nötig.

## Scope / YAGNI

Kein Repro-Radius-Ausbau (Teleport-Paaren, §4b-Verstoß). Kein Partner-Taxis (bleibt aus). Keine Biss-Kadenz-
Drossel jetzt (erst wenn A/B es fordert). Keine v1-Pfad-Änderung.
