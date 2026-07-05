# Demografie-Diagnose: Warum der Etappe-1-Kalibrier-Sweep am Gate scheitert

Datum: 2026-07-05
Autor: Diagnose-Analyst (Fable 5)
Datengrundlage: `sweep_e1/cfgA..cfgF`, je 3 Seeds (41/42/43), nolearn, 5000 Ticks,
age-structured founders AN, min-pop 4 / respawn 2, Grid 40×30 (1200 Zellen),
30 Gründer. Code: `artificial_society/` @ Worktree `as-pilot`.

## Sweep-Matrix (Belege: `sweep_e1/cmds.txt`, meta-Records)

| cfg | regrow-scale | min-food-per-capita |
|-----|--------------|---------------------|
| A   | 1.0          | 6.0                 |
| B   | 1.5          | 6.0                 |
| C   | 2.0          | 6.0                 |
| D   | 1.0          | 3.0                 |
| E   | 1.5          | 3.0                 |
| F   | 2.0          | 3.0                 |

Median über 3 Seeds (peakPop, finPop, births/deaths/respawns final, mean_energy
im Steady-State ab Tick 2500, mean_age final, births/deaths in der 2. Laufhälfte):

| cfg | rgrw | minf | peakPop | finPop | births | deaths | resp | me_settle | age_fin | 2h_births | 2h_deaths |
|-----|------|------|---------|--------|--------|--------|------|-----------|---------|-----------|-----------|
| A   | 1.0  | 6.0  | 59      | 4      | 61     | 132    | 22   | 74.7      | 183     | 5         | 36        |
| B   | 1.5  | 6.0  | 63      | 4      | 84     | 142    | 21   | 97.9      | 220     | 5         | 33        |
| C   | 2.0  | 6.0  | 60      | 4      | 67     | 135    | 20   | 103.3     | 212     | 7         | 34        |
| D   | 1.0  | 3.0  | 61      | 5      | 56     | 118    | 19   | 70.9      | 198     | 2         | 34        |
| E   | 1.5  | 3.0  | 61      | 5      | 65     | 128    | 20   | 96.0      | 246     | 4         | 30        |
| F   | 2.0  | 3.0  | 61      | 4      | 73     | 137    | 15   | 94.9      | 313     | 10        | 30        |

Repräsentative Zeitreihe (cfgC seed41, 30 Gründer → Boom → Crash → Boden):

```
tick  pop b_cum deaths resp   me   age
 250   60    31      1    0  98.7  597   <- Boom: 30 Gründer haben sich verdoppelt
 500   48    50     32    0  36.4  629   <- Overshoot: mean_energy KOLLABIERT 99->36
 750   11    53     72    0  55.7  753   <- Massensterben (40 Tode in 500 Ticks)
1000    6    53     77    0 120.9  927   <- Boden; Überlebende SATT (me 120)
1500    4    53     79    0 136.3 1546   <- Geburten eingefroren seit Tick 500
2500    4    53     89    5  66.5  288   <- Pop am Respawn-Floor 4-5 festgenagelt
5000    4    58    124   20  19.5  212   <- nur +5 Geburten in 4500 Ticks; resp 0->20
```

## Befund 1 — Todesursache: Overshoot-Crash + individuelle Verhungerung, NICHT Alter

Zwei getrennte Sterbe-Regime, beide über `energy<=0 → health-Drain → health<=0`
(`agent.py:1354-1360`: `if self.energy <= 0: self.health -= 1.5`; `if self.health
<= 0: self.alive = False`), NICHT über das Alterslimit.

- **Kein Alterstod.** `AGE_LIMIT = 5000` (`agent.py:93`), `_age_tick`
  (`agent.py:919-925`) tötet erst bei `age >= 5000`. Gemessene `mean_age` final
  183–313 (Tabelle) — Faktor ~20 unter dem Limit. Senescence-Health-Decay startet
  erst bei `AGE_HEALTH_DECAY_START = 3500`. Tod durch Alter ist ausgeschlossen.
- **Regime 1 (dominante Todeswelle): Malthusianischer Overshoot-Crash.** 30 Gründer
  brüten bis Tick 250 auf pop 60. Bei Tick 500 stürzt `mean_energy` 99→36 und die
  Tode springen 1→32 — die 60 Mäuler überziehen die wahre Tragfähigkeit der Welt,
  das stehende `plant_food` wird schneller befressen als es nachwächst, die Kohorte
  verhungert kollektiv (pop 60→11 zwischen Tick 250 und 750). ~40 der ~130 Tode
  fallen in dieses Fenster.
- **Regime 2 (Rest): individuelle Verhungerung verstreuter Agenten am Boden.** Nach
  dem Crash liegt `mean_energy` der Überlebenden bei 100–140 (Tick 1000–1500), also
  weit über jeder Not — trotzdem klettern die Tode weiter (79→124 über Tick
  1500–5000). Das sind KEINE Hungertode der satten Kern-Überlebenden, sondern der
  per `emergency_respawn` an **zufälligen** Positionen (`simulation.py:305-311`,
  `world.random_land_position()`) eingesetzten Neu-Agenten, die verstreut in
  nahrungsarme Spots geraten und einzeln verhungern, bevor sie beitragen. `mean_age`
  bei Tod fluktuiert 105–350 → junge, kürzlich respawnte/geborene Individuen.

Netto-Energiehaushalt (bestätigt: Nahrung ist im Steady-State NICHT der Engpass):
Grundumsatz `move_cost = 0.5 × mods × stage` (`agent.py:1336-1341`) plus
Health-nichts bei energy>0 → ~0.5 Energie/Tick. Eine erfolgreiche Forage liefert
`take = min(plant_available, PLANT_ENERGY × eff)` mit `PLANT_ENERGY = 30`
(`agent.py:625-627`, `97`) — also bis zu **+30 gegen −0.5**, ein 60:1-Überschuss,
SOFERN die Zelle Nahrung trägt. Der Tod ist damit rein **räumlich** (leere Zelle),
nicht bilanziell: wo Nahrung steht, ist Energie im Überfluss (me 100–140); wer
keine erreicht, stirbt. Das erklärt hohe `mean_energy` UND fortlaufende Tode
gleichzeitig.

## Befund 2 — Regrow-Ceiling-Klemme: regrow-scale hebt me leicht, Tragfähigkeit NICHT

Der Review-Vorbehalt ist im Kern bestätigt, mit einer Präzisierung.

- **Was regrow-scale patcht:** `_scale_regrowth` (`m1_pilot.py:272-310`) multipliziert
  NUR die **Zufluss-Raten** `FOOD_SCARCITY_FACTOR` (Default 0.50) und
  `MEAT_SCARCITY_FACTOR` (0.55). Beide werden in `regrow_grid` zur Call-Zeit im
  Globals-Dict von `resources.py` nachgeschlagen (`resources.py:521`, `539`) →
  wirklich patchbar.
- **Was die Decke setzt und NICHT gepatcht wird:** Der stehende Zielbestand ist
  `plant_target = plant_ceiling × capacity + farm_bonus` (`resources.py:547`), und
  `plant_headroom = max(0, 1 − plant_food/plant_target)` (`resources.py:549`)
  klemmt den Zuwachs gegen 0, je näher der Standbestand an `plant_target` kommt. Die
  Decke `plant_ceiling` stammt aus `SCARCITY_CEILING_FACTOR = 0.35`
  (`resources.py:45`, via `biome_scarcity_ceiling`, `resources.py:78-79`) und ist
  vom Zufluss-Faktor **entkoppelt**: regrow-scale beschleunigt nur, WIE SCHNELL der
  Standbestand seine feste Decke erreicht, hebt die Decke selbst nicht.
- **Datenbeleg:** `mean_energy` steigt mit regrow-scale messbar (cfgA 74.7 → cfgC
  103.3; cfgD 70.9 → cfgF 94.9, ~+35 %), ABER peakPop (~60), Crash-Tiefe (alle auf
  4–5) und finPop sind über alle regrow-scales **identisch**. Auflösung: am Boden
  (pop 4–5) ist der Befressungsdruck ~null, die Zellen sitzen ohnehin an ihrer
  Decke; höherer Zufluss füllt befressene Zellen schneller nach → jede Forage bringt
  mehr → satte Überlebende. Die **Tragfähigkeit bei hoher Dichte** (die den
  Overshoot-Crash und die nachhaltige Populationsgröße bestimmt) hängt aber allein
  an der Decke und bleibt unverändert. → **Für die Demografie ist regrow-scale ein
  schwacher Knopf; der Tragfähigkeits-Hebel ist die Decke, und die exponiert die
  Harness nicht.**

## Befund 3 — Reproduktions-Engpass: Allee-Falle am Boden, NICHT Energie/Food-Gate

- **Energie bindet nicht.** `can_reproduce` verlangt `energy >= REPRODUCTION_ENERGY
  = 60` (`agent.py:76`, `505`). Die Überlebenden liegen bei me 100–140 (2× Schwelle).
  Trotzdem froren die Geburten von Tick 500 (b_cum 50) bis 5000 (b_cum 58) auf **+8
  in 4500 Ticks** ein. Energie ist es nicht.
- **Das Food-Gate bindet nicht.** `_try_reproduce` blockt bei
  `_local_food_per_capita < REPRODUCTION_MIN_FOOD_PER_CAPITA` (`agent.py:820`). cfgD/E/F
  senken das Gate 6.0→3.0 — **ohne Effekt** auf die Geburten (2h-births cfgA 5 vs
  cfgD 2; cfgF 10 nur wegen regrow 2.0, nicht wegen des Gates). Bei satten Zellen ist
  das Gate ohnehin erfüllt.
- **Was bindet: Partnerfindung bei niedriger Dichte (Allee-Effekt).**
  `_try_reproduce` (`agent.py:817-857`) verlangt ein fruchtbares Weibchen UND ein
  fruchtbares Männchen `abs(dx)<=5 and abs(dy)<=5` (ein 11×11 = 121-Zellen-Kasten)
  mit `trust >= -0.2`. Bei pop 4–5 auf 1200 Zellen, ~je zur Hälfte m/f, treffen sich
  fruchtbare Partner im ±5-Kasten fast nie → Geburten ≈ 0. `emergency_respawn` setzt
  zwar bei pop<4 zwei Agenten nach, aber an **zufälligen, verstreuten** Positionen
  mit age 0 (müssen erst `MIN_REPRODUCTION_AGE = 60` erreichen, `agent.py:91`), die
  bei ~360–400 Tick Lebensdauer meist sterben, bevor sie einen Partner finden →
  respawns feuern endlos (0→20 über den Lauf) → **Gate `respawns→0` unerreichbar**.
- **Lebensdauer vs. Reproduktionszyklus:** Steady-State-Lebensdauer ~360–400 Ticks
  >> `MIN_REPRODUCTION_AGE 60 + GESTATION_TIME 40 + REPRODUCTION_COOLDOWN 100 = 200`
  (`agent.py:91-92`, `78`). Der Zyklus frisst die Fenster also NICHT auf — ein
  überlebender Agent hätte 1–2 Reproduktionsfenster. Der Engpass ist rein die
  **Partner-Dichte**, nicht die Zeit oder die Energie.

## Synthese 4 — Priorisierte Knob-Liste für den nächsten Sweep

Kernbefund: Die gescweepten Knöpfe (regrow-scale, min-food-per-capita) adressieren
KEINE der beiden bindenden Ursachen. Die zwei echten Constraints sind (A) die von
`SCARCITY_CEILING_FACTOR` gesetzte Tragfähigkeit (steuert Overshoot-Tiefe und
nachhaltige Popgröße) und (B) die Partner-Dichte am Boden (Allee-Falle).
Patchbarkeit wurde am Code geprüft und ist je Knopf annotiert.

### Prio 1 (höchster Hebel): Pflanzen-Tragfähigkeit anheben (Ceiling-Knob)

Adressiert direkt die Tragfähigkeit → macht pop ~60 nachhaltig statt zum Overshoot,
hält `food_per_capita` über dem Gate → Geburten laufen weiter.

- **Patch-Surface — ACHTUNG, NICHT wie regrow-scale:** Im vektorisierten Live-Pfad
  `regrow_grid` ist die Decke eine **vorberechnete Pro-Zell-Array**
  `plant_ceiling = bio["plant_ceiling"]` (`resources.py:546`), NICHT das Modul-Global
  `SCARCITY_CEILING_FACTOR`. Ein Patch von `resources_mod.SCARCITY_CEILING_FACTOR`
  nach dem Welt-Bau bliebe **wirkungslos** (regrow_grid liest das Array, nicht die
  Konstante). Call-time-patchbar ist stattdessen die Array auf der Welt-Instanz:
  `sim.world._bio["plant_ceiling"] *= f` direkt nach dem `Simulation`-Bau (das Array
  ist statisch, `capacity`/`_bio` werden pro Tick nicht neu gebaut —
  `resources.py:490` "never written here"). Das hebt konsistent Zielbestand
  (`:547`) UND Hard-Cap (`plant_hard_cap`, `:580`, nutzt dasselbe Array).
  Verifizieren: exakter Feldname `_bio["plant_ceiling"]` und dass er nicht pro Tick
  regeneriert wird (per Kurz-Run prüfen). `MEAT_CEILING_FACTOR` (`:548`) bleibt
  separat call-time-patchbar als Modul-Global, falls Fleisch relevant wird.
- **Sweep-Wertebereich:** Faktor `f ∈ {1.7, 2.5, 3.4}` (effektive Decke ~0.6 / 0.9 /
  1.2 gegenüber Basis 0.35). Erwartung: Overshoot verschwindet, pop hält sich um
  40–60, respawns fallen auf 0.

### Prio 2 (bricht die Allee-Falle): Dichte über die Paarungsschwelle heben

Selbst bei unbegrenzter Nahrung brüten 4–5 Agenten auf 1200 Zellen nicht. Zwei
call-time-patchbare Hebel (kombinierbar), plus ein Knopf, der einen Source-Edit
braucht:

- **2a — Respawn-Floor & min-pop hochsetzen (patchbar):**
  `simulation_mod.MIN_POPULATION` und `RESPAWN_COUNT` sind call-time gesetzt
  (`m1_pilot.py:509-510`). Statt Boden 4/2 den Boden so hoch legen, dass natürliche
  Geburten die Dichte übernehmen und respawns DANN auf 0 fallen. Sweep:
  `min_pop ∈ {12, 16}`, `respawn_count ∈ {6, 8}`. (Gegen-Intuition beachten: das
  Gate misst `respawns→0`; ein hoher Floor hilft nur, wenn er die Population über die
  Paarungsschwelle hebt und danach selbsttragend wird — sonst maskiert er nur.)
- **2b — Grid verkleinern = Dichte erhöhen (patchbar, CLI):** `--grid-w/--grid-h`
  (`m1_pilot.py:649-650`, Default 40×30). Von 1200 auf ~24×18=432 oder 20×15=300
  Zellen bei gleicher Gründerzahl → 2,5–4× Dichte → Partner treffen sich im
  ±5-Kasten. Vorsicht: kleineres Grid konzentriert auch den Befressungsdruck →
  am besten mit Prio 1 (höhere Decke) koppeln.
- **2c — Paarungsradius (braucht Source-Edit, FLAG an Core):** Der eigentliche
  Allee-Hebel ist der Mate-Such-Kasten `abs(...) <= 5` — ein **hartkodiertes Literal**
  in `_try_reproduce` (`agent.py:830-835`), KEINE benannte Konstante → aktuell NICHT
  call-time-patchbar. Empfehlung: als Modul-Konstante `MATE_SEARCH_RADIUS`
  extrahieren (analog `REPRODUCTION_SENSE_RADIUS`, `agent.py:88`), dann als Knopf
  sweepbar (z. B. 5 → 8 → 12). Bis dahin über 2a/2b lösen.

### Prio 3 (sekundär, nur falls 1+2 nicht reichen): Overshoot-Amplitude dämpfen

Die Alters-Staffelung `[0,2000)` (`m1_pilot.py:_age_structure_founders`) lässt immer
noch ~½ der 30 Gründer bei t=0 fruchtbar (age≥60) → sofortiger ~15er-Boom. Falls
Prio 1 den Boom nicht tragfähig macht: Gründerzahl senken (falls exponierbar — via
`spawn_initial_population`, aktuell KEIN CLI-Knopf, Verifikation nötig) oder die
Staffel-Obergrenze so weiten, dass weniger Gründer gleichzeitig fruchtbar sind.
Niedrigerer Hebel als 1+2.

### De-priorisiert (Datenbeleg: binden nicht)

- **regrow-scale:** ceiling-geklemmt (Befund 2). Als milder Helfer bei 1.5 halten,
  nicht als primärer Tragfähigkeits-Knopf.
- **min-food-per-capita:** nicht bindend (Befund 3); auf 6.0 lassen, um das
  Dichte-Feedback zu erhalten (Senkung brachte nichts).
- **REPRODUCTION_ENERGY / REPRODUCTION_COOLDOWN:** Energie/Zyklus binden nicht
  (Befund 3, me 100–140, Lebensdauer >> Zyklus). Senken würde nichts freischalten.

## Empfohlener nächster Sweep (kompakt)

Kern: **Ceiling × Dichte** statt regrow × food-gate.
`ceiling_factor ∈ {1.7, 2.5, 3.4}` × `{(min_pop 12, respawn 6) ODER grid 24×18}`,
regrow-scale fix 1.5, min-food-per-capita fix 6.0, age-structured founders AN,
3 Seeds, 5000 Ticks. Vorab: `sim.world._bio["plant_ceiling"] *= f`-Patchpfad per
1-Tick-Smoke-Test verifizieren (Feldname + Persistenz).
