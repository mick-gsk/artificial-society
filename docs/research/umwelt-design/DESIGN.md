# DESIGN.md — Path-A Mechanismus-Design (C1–C5, korrigiert, post-Red-Team)

Dieses Dokument spezifiziert das falsifizierbare Path-A-Design zur Untersuchung der Frage, unter welchen Bedingungen das **gekoppelte System** (Lerner + Umwelt) demonstrierbare offene kumulative Akkumulation **funktionaler** Komplexität gegen eine **compute-gematchte** Zufalls-/Rekombinations-Baseline zeigt. Es geht **nicht** um "überlegene Spezies", und es enthält **keinerlei** Garantie-/Unvermeidbarkeits-Sprache. Ob eine Bedingung hinreichend ist, entscheidet ausschließlich das Experiment.

## 0. Lese- und Geltungsregeln

- **belegt** = code-verifiziert mit `file:line`-Anker. **vermutet** = Hypothese, durch das Experiment zu prüfen.
- Jeder Mechanismus ist ein **Vier-Tupel**: (1) Mechanismus auf den Code abgebildet; (2) compute-gematchte, null-kalibrierte Baseline (definiert **vor** dem Mechanismus); (3) hack-resistente **funktionale** Metrik; (4) **pre-quantifiziertes Kill-Kriterium** (die Messung, die die Hypothese widerlegt).
- Jeder Mechanismus trägt das Label **notwendig-nicht-hinreichend** mit Begründung gegen die Diagnose und eine C1–C5-Abbildung.
- Verbotene Wörter (Garantie-Sprache): zwangsläufig, führt zu, garantiert, erzeugt automatisch, stellt sicher, löst das Problem. Erlaubt: "ist plausibel notwendig für X", "entfernt einen bekannten Blocker", "macht X möglich, ohne es zu garantieren".
- Compute-Regime (Default): paired seeds 1001–1012 (`run_pilot.py:27`), 1500 ticks, grid 30×20, pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`, GPU-PC via SSH. 250-tick-Smokes lügen (Transient); nur 1500-tick-Endpunkte zählen. Gepaarter Bootstrap-CI via `analyze_gate._bootstrap_mean_ci` (`analyze_gate.py:41`, `N_BOOTSTRAP=10000` `analyze_gate.py:38`).

## 0.1 Korrigierte Designprinzipien C1–C5 (Kurzform)

- **C1**: Achse ist **statisch vs. nicht-stationär** (an die eigene Population gekoppelt), **nicht** endogen-vs-exogen; mit Hall-of-Fame + Diversity-Pressure als Disengagement-Garde.
- **C2**: **Ausdruckskraft** (indirekte Kodierung) statt roher Kettenlänge; diskrete/Bottleneck-Kanäle treiben Kompositionalität (Resnick 2020, VERIFIZIERT) und sind ein Feature; Ausdruckskraft ≠ Erreichbarkeit (Trittsteine nötig); mehr Raum ist **nicht** der primäre Engpass.
  - *Die zwei früher-falschen C2-Teilsätze, explizit umgedreht (Audit-Spur):* **(i)** FRÜHER behauptet (**FALSCH**): „enge/diskrete Kanäle verhindern reiche Sprache." → **KORREKT:** enge/diskrete Kanäle **treiben** Kompositionalität (Resnick et al. 2020, VERIFIZIERT) — Bottleneck als Feature. **(ii)** FRÜHER behauptet (**FALSCH**): „fixe Genomlänge = fixer Komplexitätsdeckel." → **KORREKT:** entscheidend ist die **Ausdrucksmächtigkeit** der Repräsentation (indirekte/CPPN-artige Kodierung), **nicht** die rohe Länge. Beide falschen Formen erscheinen in diesem Dokument ausschließlich in dieser umgedrehten Fassung.
- **C3**: Nischenkonstruktion ist eine **ermöglichende** Bedingung, richtungsneutral; verlangt Persistenz + Vererbbarkeit (hier **belegt PASSED**, Diagnose Befund 5).
- **C4**: kumulative Kultur via Demographie (effektive Populationsgröße + Konnektivität) + hochfidele Transmission ist der unbestrittene Kern; MLS ist umstritten und **nicht** als Haupt-Lever verkauft; die drei Paare (kulturell-vs-genetisch, Gruppe-vs-Individuum, Koordination-vs-IQ) werden **nie** verwechselt.
- **C5**: Offenheit ist konstitutiv eine Eigenschaft des **gekoppelten** Lerner+Umwelt-Systems; die Dichotomie "Umwelt statt Agent" ist falsch (Hughes 2024, VERIFIZIERT).

## 0.2 Abhängigkeits- und Reihenfolge-Graph

Das Red-Team hat code-belegt gezeigt, dass **M1, M2, M3, M4 und M6 auf der ungerepairten Messung scheitern oder uninterpretierbar werden** (DV2-Floor=2 in beiden Armen aller bisherigen Läufe; DV3 `computable=False` wegen `discovered_by=-1`; DV1 strukturell leer). **M5 ist deshalb ein harter, vorgezogener Blocker.** Verbindliche Reihenfolge:

1. **M5** (Messreparatur) — Vorbedingung für alle übrigen. Floor-Test (graded-Metrik > 2 unter positiv-synthetischem Input) **muss bestehen**, bevor irgendein Null-Resultat von M1–M4/M6 interpretiert wird.
2. **M2** (embodiment-gematchte Baseline) — gated auf M5; definiert die Kontrolle vor jeder Lernbewertung.
3. **M1** (selektive Generierung) — gated auf M5; gemessen gegen internen embodiment-gematchten Null + extern orientierend gegen M2.
4. **M3** (Diversitäts-Erhaltung) — gated auf M5; **nur** Within-Arm-A/B über M1, **keine** Headline.
5. **M4** (Nicht-Stationarität) — gated auf M5; abhängig von M1/M3 für Erreichbarkeit.
6. **M6** (Fidelity-Sweep) — gated auf M5 **und** M1-ON-Kontrollzelle; herabgestuft auf bestätigend.

---

## M5 — MESSREPARATUR (vorgezogen, harter Blocker)

**Non-gefloorte, arm-symmetrische funktionale Metrik + adoptions-basierte Instrumentierung + per-agent `discovered_by` im Null.**

### Mechanismus (belegt + Red-Team-korrigiert)

Drei Teile; zwei davon wurden umgeschrieben, weil das Red-Team die Originalform code-belegt widerlegte.

**(a) Non-gefloorte Metrik statt "per-Agent-Mittel".** Die DV2-Familie misst gegen `TASK_BASIS` (`metrics.py:261`) = 6 Utility-Funktionen, jede ein Produkt geclampter Property-Dims (z. B. `_u_edible_safe = edibility*(1-toxicity)`, `_u_cutting_tool = sharpness*hardness`). Diese Utilities **sättigen bei ~1.0 (belegt)**. Daher **kein** Aggregationstrick: ein per-Agent-Mittel wäre arm-asymmetrisch (der Recombiner ist "1 Agent"; Red-Team M5-Angriff 3). Stattdessen eine echte tiefen-auflösende Metrik **`graded_useful_advance`**: pro Task **nicht** binär (`> frontier + margin`), sondern die **Summe der margin-normierten Frontier-Vorsprünge über alle Strukturschichten** (`accumulated_useful_depth` `metrics.py:290` erweitert — statt `per_task_maxdepth` den kumulierten Vorsprungs-Betrag pro Schicht zählen), sodass der Score **nicht** bei Floor=2 deckelt, sobald 6 Tasks je einmal getroffen sind.
- *vermutet:* dass diese Reformulierung dem embodied Arm Headroom über Floor=2 verschafft — wird durch den Floor-Test (Akzeptanzkriterium a) **geprüft**, nicht angenommen.

**(b) `record_use` auf Adoptions-Events, NICHT auf Inventar-Konsum.** Red-Team M1-Angriff 4 / M5-Angriff 4 ist **code-verifiziert**: die einzige echte Dekrement-/Konsum-Stelle ist `sharp_stone` in `agents/agent.py:752` — **kein** entdecktes `mat_XXXX` wird je homeostatisch verbraucht. Ein auf Inventar-Konsum gehängter `record_use`-Hook **feuert ins Leere** → DV1 würde "sauber aber strukturell leer". Daher hängt `record_use` an **Adoptions-Events**: `culture.receive_transmitted` (`culture.py:80`) + `brain.imitate_from` (`brain.py:186`) als Adoptions-Proxy, **ohne Verhaltens-Edit der Golden-Trajektorie**.

**(c) `discovered_by`-Durchreichung im Export.** `discovered_by` ist im Registry vorhanden (`materials.py`), wird aber im JSON-Export gedroppt → DV3 (`transmitted_frontier_advances` `metrics.py:415`) ist `computable=False`. Fix: `discovered_by` im Export durchreichen → DV3 wird computable (`>=0`).

### Baseline (compute-gematcht, vor dem Mechanismus)

Bestehende provisional/non-computable DVs auf **denselben** Pilot-Exports: DV1 `provisional=True`, DV3 `computable=False`, DV2 floored. Teile (a) DV2-Tiefenauflösung und (c) `discovered_by` sind **offline** auf existierenden Pilot-Daten validierbar (kein Re-Run nötig). Teil (b) `record_use`-via-Adoption erfordert Re-Run: paired seeds 1001–1012, 1500 ticks, Standard-Regime.

### Metrik (selbst-referentiell — die Metrik *ist* das Artefakt)

Drei Knock-out-validierte (`metrics.py:27-30`) Akzeptanzkriterien:
- **(a)** `graded_useful_advance` gibt unter kontrolliert positiv-synthetischem Tiefen-Input einen Wert **strikt > Floor=2** aus **und** ist arm-symmetrisch (identische Funktion auf learned- und Recombiner-Entries, **keine** per-Agent-Aggregation, die nur einem Arm zusteht).
- **(b)** DV1-Adoptions-Gewicht (`weight_source='adoption'`) ist **> 0** für **beide** Arme **oder** der Befund wird explizit als arm-asymmetrisch deklariert.
- **(c)** DV3 `computable=True` (`metrics.py` Pfad), weil `discovered_by >= 0`.

### Kill-Kriterium (pre-quantifiziert, vier Teile)

- **(a)** Wenn `graded_useful_advance` unter positiv-synthetischem Input strukturell **nicht > 2** ausgeben kann, ist sie kein Fortschritt gegenüber DV2 → **Floor-Reparatur fehlgeschlagen**, M1–M4 bleiben uninterpretierbar.
- **(b)** (Red-Team-Angriff 4) Wenn `record_use`-via-Adoption das `total_weight` des **learned** Arm über 1500 ticks ~0 lässt (Hook ins Leere), ist DV1 weiter strukturell leer → **DV1-Reparatur fehlgeschlagen**, DV1 wird endgültig als nicht-rettbar markiert und aus allen Gates entfernt.
- **(c)** Wenn der embodiment-gematchte Null (M2) trotz Adoptions-Events DV1=0 bleibt (kein `discovered_by>=0`, kein `record_use`), ist die Instrumentierung **nicht arm-symmetrisch** → M2-DV1-Vergleich verworfen.
- **(d)** Wenn DV3 für **keinen** seed computable wird, ist der Export-Fix unvollständig.

### Label: notwendig-nicht-hinreichend (HARTER BLOCKER)

Das Red-Team zeigte, dass M1 (Floor-Confound), M2 (am Floor uninterpretierbares Kill), M3 (DV2-Cap inert), M4 (kein Headroom) und M6 (DV3 non-computable) **alle** auf der ungerepairten Messung scheitern. M5 entfernt diesen Mess-Blocker und ist deshalb **vorgezogen** als Vorbedingung. **Nicht hinreichend:** bessere Messung erzeugt keine Komplexität; Diagnose (A) belegt, dass das Defizit die DV2-Reform **überlebt** (learned DV2~2 vs. Recombiner~32) → **M5 dreht das Vorzeichen nicht.**

### C-Mapping

Querschnitt (Messhygiene für die C5-Bewertung): erlaubt erst die saubere Trennung learner-vs-environment-Effekt und behebt die "fairness proof"-Frage (DV1 floored den Recombiner aus Instrumentierungs-, nicht Verhaltensgrund). Die graded-Metrik adressiert direkt das DV2-Floor (Diagnose Caveat A).

### Adressierte Diagnose-Befunde

Befund 1 (DV1 uses-Pollution / "fairness proof"), Befund (A)-Caveat (DV2-Floor-Saturation), DV3-Non-Computability.

### Red-Team-Zusammenfassung (Verdikt: fragil)

| Angriff | Kern | Antwort / Status |
|---|---|---|
| 1 — DV3 arm-asymmetrisch (Recombiner `uses=0`/`discovered_by=-1`) | **does-not-survive** in Originalform | Behoben **nur** wenn M2 als attribuierter, konsumierender Null gebaut wird (Kill-Kriterium c fängt es); der `record_use`-Teil auf der bestehenden disembodied `recombiner.py` fällt durch sein eigenes Kill-Kriterium |
| 2 — versteckte statische Fitness (TASK_BASIS fix) | trifft DV2-Familie, nicht M5 spezifisch | Teil-Survival: margin macht churn-immun; M5 macht arm-symmetrischer, **nicht** offen |
| 3 — per-Agent-Aggregation fabriziert Signal | berechtigt | Behoben durch Floor-Test (struktur. Kapazität) + Präregistrierung + **arm-symmetrische** Aggregation |
| 4 — `record_use` ins Leere (kein `mat_XXXX`-Konsum) | **does-not-survive** in Originalform | **Korrigiert:** Adoptions-Proxy statt Inventar-Konsum + Kill-Kriterium (b) |
| 5 — 250-tick-Transient | reine Offline-Metrik auf finaler append-only Registry | immun, sofern kumulative Lesart + ≥1500 ticks |
| 6 — Baseline weder compute- noch embodiment-matched für fairness-Anspruch | berechtigt | Konditionales Survival: ja als Instrumentierung, fairness-Auflösung hängt an korrekt attribuiertem M2 |

**Fragile Stellen (markiert):** (i) der fairness-Anspruch trägt **nur** zusammen mit einem korrekt attribuierten M2; (ii) die statische 6-Task-Basis-Decke bleibt — M4 adressiert sie, M5 nicht.

---

## M2 — EMBODIMENT-GEMATCHTE Baseline (attribuierter, konsumierender Null), gated auf M5

**Gedrosselter Recombiner unter Lokalität / Teilinformation / per-tick-Limit, mit per-agent `discovered_by` und Adoptions-`record_use`.**

### Mechanismus (belegt + Red-Team-korrigiert)

Der bestehende `recombiner.py:run_recombiner` ist ein disembodied perfect-memory **Monopolist**: globaler monoton wachsender Pool (unconditional append `recombiner.py:104`), `discovered_by=-1` (`recombiner.py:99`), `uses=0` (`recombiner.py:101`), Seed = alle 24 `MATERIALS`, RNG save/restore (`recombiner.py:54-55,113`, **belegt**). **Mechanismus:** `run_recombiner_embodied` (neue Funktion, berührt keine Hot-Files): K=pop **disjunkte lokale Pools** (Fragmentierung), per-tick-Attempt-Budget, **Teilinformation** (nur Materialien in Zelle/Inventar), Pool-Austausch nur über sozialen Kanal mit Fidelity `FIDELITY_BASE=0.72` (`social_learning.py:26`). Uniform action/material wie Original (`recombiner.py:78,76`); **einziger** Unterschied zum learned Arm bleibt die fehlende Policy.

**Red-Team-Korrektur (M2-Angriff 1 / M5-Angriff 1):** der Null **muss** attribuiert + konsumierend sein, sonst ist der DV1/DV3-Vergleich eingebaut-asymmetrisch. Daher: per-Agent `discovered_by >= 0` setzen **und** `record_use`-via-Adoptions-Events erzeugen (analog M5-Instrumentierung), sodass DV1/DV3 für den Null strukturell **≠ 0** sein kann.

**Red-Team-Korrektur (M2-Angriff 4):** RNG als pro-Agent-Streams aus dem Seed deriviert, deterministische tick/agent-Reihenfolge, unit-getestet (sonst bricht der gepaarte Bootstrap).

### Baseline (M2 IST die Baseline — Kontrolle, vor jeder Lernbewertung)

Drei-Arm pre-registriert: (1) learned (mit M1), (2) embodiment-**freier** Recombiner (bestehend), (3) embodiment-**gematchter + attribuierter** Recombiner (neu). Compute-Match: Arm (3) fährt dieselbe Gesamtzahl **real gezählter** `combine_vectors` wie der learned Arm.

**Red-Team-Korrektur (M2-Angriff 6):** wenn M1 einen ungezählten `_combine_pure`-Imaginationssuchlauf nutzt, bekommt Arm (3) denselben ungezählten Bewertungs-Twin, aber mit **uniformer** statt policy-gewichteter Auswahl → "nur die Auswahlregel unterscheidet sich" bleibt ehrlich.

### Metrik

- **PRIMÄR:** `graded_useful_advance` (M5-non-gefloorte Metrik) je Arm, **nicht** das gefloorte DV2 (M2-Angriff 5: am Floor ist learned-vs-(3)~0 nicht wegen Embodiment, sondern wegen Mess-Saturation).
- **SEKUNDÄR:** DV3 `transmitted_frontier_advances` (nach M5 computable + attribuiertem Null).
- Gepaarter Bootstrap-CI (`analyze_gate.py:41`, `N_BOOTSTRAP=10000`) für learned-vs-(2) und learned-vs-(3).

### Kill-Kriterium (pre-quantifiziert, billigste + härteste Falsifikation, gated auf M5)

**Nur** ausführbar, **nachdem** M5s graded-Metrik den Floor-Test (>2) bestanden hat (sonst durch Floor-Saturation konfundiert; M2-Angriff 5 **belegt**). Dann: wenn der embodiment-gematchte Null (3) `graded_useful_advance >= learned` erreicht (gepaarter CI learned-(3) **nicht > 0**, `d_lo <= 0`), ist der "Lerndefizit" ein **Embodiment-Artefakt** → Lernhypothese für diese Metrik **widerlegt**, Befund als Konfundierung berichtet (Path B). Echtes Lernsignal **nur**, wenn learned > **beide** Nullen (2) **und** (3) über gepaarten CI auf der non-gefloorten Metrik.

### Label: notwendig-nicht-hinreichend

Die Diagnose nennt explizit den **Embodiment-Confound**. Ohne diese Kontrolle ist jeder learned-vs-recombiner-Vergleich uninterpretierbar. **Nicht hinreichend:** M2 erzeugt keine Komplexität, reine Messhygiene/Falsifikation; es **macht die Lernhypothese falsifizierbar, ohne sie zu garantieren.**

### C-Mapping

C5/C1: trennt die Eigenschaft des gekoppelten Systems (embodied, fragmentiert, nicht-stationär) vom disembodied Monopolisten; klärt "Umwelt vs. Agent" als falsche Dichotomie. Direkt an der Fairness-/Confound-Frage (Diagnose C).

### Adressierte Diagnose-Befunde

Embodiment-Confound, Lever-Priorität #2.

### Red-Team-Zusammenfassung (Verdikt: fragil)

| Angriff | Kern | Antwort / Status |
|---|---|---|
| 1 — DV2-Mengen-/Churn-Hack | DV2 churn-immun by design | fair, überlebt |
| 5 — DV2-Floor macht Kill uninterpretierbar | **does-not-survive** ohne M5 | **Korrigiert:** Kill **gated auf M5**; primär auf graded-Metrik gemessen |
| 6 — Compute-Match-Defekt unter M1 (ungezähltes Imaginationsbudget) | teilweise | **Korrigiert:** uniformer ungezählter Twin auch für den Null, pre-registriert |
| 2 — versteckte statische Fitness (TASK_BASIS) | symmetrisch über Arme | fair; Geltungsgrenze "nur Akkumulation Richtung 6 Ziele" dokumentieren |
| 3 — notwendig-mit-hinreichend verwechselt | nein | sauber gelabelt |
| 4 — RNG-Ordering bricht paired Bootstrap | Implementierungs-Hazard | **Korrigiert:** pro-Agent-RNG-Streams + Unit-Test |

**Fragile Stelle (markiert):** das Kill-Kriterium ist **nur** nach bestandenem M5-Floor-Test valide; die statische 6-Task-Decke bleibt eine Geltungsgrenze.

---

## M1 — Policy-gekoppelte SELEKTIVE Generierung (seeded-softmax + Value-of-Information), gated auf M5

**RNG-isolierter ungezählter imaginierter Zwilling; gemessen auf der M5-Metrik als CO-PRIMÄR.**

### Mechanismus (belegt + Red-Team-korrigiert)

Die PPO-Policy entscheidet das **WANN** (`need_driven_invention.py:351` `if need_magnitude < eff_threshold`), das **WAS** ist quasi-zufällig (`_select_materials_by_need`/`_select_action_by_need`, `need_driven_invention.py:305-310`, softmax + `np.random.choice`). **Mechanismus:** ungezählter imaginierter Zwilling `_combine_pure` (reine Funktion über `combine_vectors` `materials.py:430`, **ohne** `DISCOVERY_REGISTRY.register`/`uses`-Inkrement), bewertet ein kleines Kandidatenset nach `value(_need_score)` + `voi_weight*novelty` + `RATCHET_GAIN*marginal(Frontier-Vorsprung)`; Auswahl per **seeded-softmax** (**nicht** argmax); dann **eine** real gezählte `combine_vectors` + register. Gegated per env-Flag (`AS_PATHA_CD`-Muster); OFF byte-identisch zur Golden.

**Red-Team-Korrekturen, alle code-verifiziert:**

- **(i) Angriff 6 — RNG-Leck (belegt):** `combine_vectors` konsumiert den globalen `random`-Stream (rub/ignite `materials.py:452-455` `random.random()`/`random.uniform`). Der Zwilling **muss** `random.getstate()` **vor** dem K-Kandidaten-Sweep snapshotten und `random.setstate()` **danach** (`recombiner.py:54/113`-Muster), sodass die imaginierte Suche **netto 0** RNG-Draws verbraucht und die **eine** reale `combine_vectors` aus demselben Stream-Punkt wie OFF zieht. **Unit-Test:** globaler RNG-State **byte-identisch** vor/nach dem imaginierten Sweep (nicht nur "reale Aufrufe == 0").
- **(ii) Angriff 4 — `uses`-Pollution (belegt):** der Zwilling **muss** Vektoren **ohne** `uses`-Inkrement lesen (`get_vector` `materials.py:294-298` inkrementiert `uses` bei **jedem** Lookup) → nebenwirkungsfreier Getter / gecachte Vektoren.
- **(iii) Angriff 1 — Metrik-Kopplung (Tautologie-Risiko):** der `RATCHET_GAIN`-Term koppelt den Generator an die DV2-Scoring-Funktion. **Korrektur:** die primäre Metrik ist eine **held-out Task-Basis disjunkt** von der, gegen die der Generator optimiert (Generator nutzt nur `value`+`novelty`; Bewertung gegen einen **zweiten**, im Generator nicht verwendeten Task-Satz), **oder** `RATCHET_GAIN=0`.

### Baseline (compute-gematcht, vor dem Mechanismus)

**Primäre Entscheidungs-Baseline (Red-Team-Angriff 6b — das Label "Primär=Recombiner" war irreführend, korrigiert):** der **embodiment-gematchte interne Null** = derselbe embodied Pfad mit `voi_weight=0` und `RATCHET_GAIN=0` (uniformes Kandidaten-Sampling), gleiche Anzahl imaginierter + realer Aufrufe. Der Recombiner (M2 Arm 2/3) ist **nur** orientierender **externer** Null.

Compute-Match-Integrität per Unit-Test: (1) imaginierte `combine_vectors` real gezählt **== 0**; (2) ON-Arm reale Aufrufe **<= OFF**; (3) NEU: globaler RNG-State **identisch** vor/nach dem imaginierten Sweep. Paired seeds 1001–1012, 1500 ticks, Standard-Regime.

### Metrik (CO-PRIMÄR — ein gefloortes Null-Resultat widerlegt nichts, Red-Team M1-Angriff 2)

`graded_useful_advance` (M5 non-gefloorte, tiefen-auflösende Metrik) **UND** als Anti-Greedy-Co-Kriterium `n_functional_clusters` (`metrics.py:144`) **gemeinsam**. Held-out-Task-Basis für die graded-Metrik (disjunkt vom Generator-Ziel). Das gefloorte Original-DV2 wird **nur** noch als Floor-Diagnostik mitgeführt, nicht als Entscheidungs-DV. Gepaarter Bootstrap-CI (`analyze_gate._bootstrap_mean_ci`, `N_BOOTSTRAP=10000`).

### Kill-Kriterium (pre-quantifiziert, gated auf M5)

**Nur** auswertbar, **nachdem** M5s graded-Metrik den Floor-Test (>2) bestand (Red-Team-Angriff 2 = does-not-survive-on-its-own-terms, jetzt behoben). Dann **widerlegt**, wenn über 12 paired seeds bei 1500 ticks der gepaarte CI von `graded_useful_advance(VoI-ON) − graded(VoI-OFF intern)` **nicht > 0** (`d_lo <= 0`).

**Zusatz-Kill (Anti-Greedy, gegen den C+D-Kollaps):** selbst bei Anstieg widerlegt, wenn `n_functional_clusters(ON) < n_functional_clusters(OFF)` über gepaarten CI.

**POSITIV-KONTROLLE PFLICHT:** vor jeder Null-Interpretation muss die graded-Metrik unter **irgendeiner** Intervention nachweislich > Floor beweglich sein, sonst ist der Null uninterpretierbar. Smoke-Tests (250 ticks) zählen **nicht** (Transient lügt; C+D war +35% bei 250, kollabierte bei 1500).

### Label: notwendig-nicht-hinreichend

Diagnose (C) identifiziert die fehlende **WAS-Kopplung** als bindenden Engpass; ohne sie ist der learned Arm ein compute-gedrosselter fragmentierter random recombiner. M1 entfernt diesen Blocker. **Nicht hinreichend:** der plan-getreue Lever (seeded-softmax + VoI) ist laut gonogo-Doc **ungetestet**; greedy-argmax (getestet) verschlechterte die Effizienz 0.127→0.029. Zudem adressiert M1 **nur** die Generator-Seite, nicht Transmission/Akkumulation über die Population (Red-Team-Angriff 3) → das Floor-Kill auf der graded-Metrik widerlegt Suffizienz korrekt. C3-Persistenz ist unabhängig **belegt PASSED**.

### C-Mapping

C5: Generierung ist konstitutiv Teil des gekoppelten learner+environment-Systems (Hughes 2024). C2: VoI/novelty nutzt die offene Material-Rekombination (adjacent-possible `materials.py:430`) als Ausdrucksraum, nicht rohe Kettenlänge; seeded-softmax statt argmax respektiert, dass **Erreichbarkeit über Trittsteine** der Engpass ist, nicht die Raumgröße.

### Adressierte Diagnose-Befunde

(C) ACTUAL BINDING ENGPASS, Lever-Priorität #1. Korrigiert greedy-argmax aus Path-A Phase D.

### Red-Team-Zusammenfassung (Verdikt: fragil)

| Angriff | Kern | Antwort / Status |
|---|---|---|
| 6 — RNG-Leck im imaginierten Zwilling (belegt) | Compute-Match-Test blind für RNG-Konsum | **Korrigiert:** getstate/setstate-Snapshot + RNG-State-Unit-Test |
| 1 — Generator optimiert die DV2-Scoring-Größe (Tautologie) | teils offen | **Korrigiert:** held-out-Task-Basis / `RATCHET_GAIN=0`; paired interner Null neutralisiert Saturations-Hack |
| 5 — 250-tick-Transient kehrt um | gut verteidigt | Kill schließt Smokes explizit aus, 1500-tick-CI + Anti-Greedy-Co-Kriterium |
| 6b — "Primär"-Baseline war der disembodied Recombiner | irreführendes Label | **Korrigiert:** interner embodiment-gematchter Null ist die Entscheidungs-Baseline |
| 3 — notwendig mit hinreichend verwechselt | nein | sauber gelabelt; Floor-Kill widerlegt Suffizienz |
| 2 — DV2-Floor macht Null uninterpretierbar | **does-not-survive** ohne M5 | **Korrigiert:** M5-Gating + Pflicht-Positiv-Kontrolle + CO-PRIMÄR graded-Metrik |
| 4 — `uses`-Pollution durch `get_vector` im Zwilling (belegt) | Instrumentierungs-Leck | **Korrigiert:** nebenwirkungsfreier Getter |

**Fragile Stellen (markiert):** (i) Tautologie-Risiko bei aktivem `RATCHET_GAIN` — nur sauber mit held-out-Basis; (ii) der gesamte Test ist erst nach bestandenem M5-Floor-Test interpretierbar; (iii) Suffizienz nicht beansprucht — Floor-Kill ist die korrekte Refutation.

---

## M3 — DIVERSITÄTS-ERHALTENDE Generierung (QD / MAP-Elites / Novelty), Within-Arm-Lever, FRAGIL

**Über Skill-Deskriptor; NUR als Within-Arm-A/B-Lever, NICHT als Open-Endedness-Headline.**

### Mechanismus (belegt + Red-Team-eingeschränkt, FRAGIL)

Das C+D-NO-GO zeigte: greedy means-ends herdet auf wenige high-value combos (Diagnose Befund 4). **Mechanismus:** MAP-Elites-Archiv (Mouret & Clune 2015, **VERIFIZIERT**) über Deskriptor = (dominante aktive Property-Dim via `mean_active_dims` `metrics.py:156`, Action-Klasse aus `PRIMITIVE_ACTIONS` `invention.py:52`). In M1s imaginierter Suche wird der seeded-softmax um einen **Novelty/Archiv-Bonus** erweitert (Lehman & Stanley 2011, **VERIFIZIERT**); Elite-Ersetzung pro Zelle nach bester Task-Utility. Neues Modul `systems/qd_archive.py` (self-registered, `registry.py:125`), gespeist aus `DISCOVERY_REGISTRY.entries`; greift **nur** in M1s Auswahl-Score ein (**kein** Hot-File-Edit, **keine** Registry-Mutation → C3-Persistenz intakt, Red-Team-Angriff 6 sauber beantwortet).

**Residual-Risiko (Red-Team-Angriff 4, belegt):** der Deskriptor ist **statisch + niedrigdimensional** (1 Property-Dim + 8 fixe Actions); die echte Ausdruckskraft liegt in **offener Material-Rekombination** (Diagnose Befund 7). Ist das fixe Gitter abgedeckt, geht der Novelty-Bonus auf 0 und QD degeneriert zum value-VoI-Sampler → nur **endliche** Diversitäts-Pressung.

### Baseline (compute- UND embodiment-gematcht, vor dem Mechanismus)

M1 mit reinem value-VoI-Sampler **ohne** Archiv/Novelty-Bonus (`voi_weight>0`, `novelty_archive_weight=0`), compute-gematcht auf identische imaginierte + reale `combine_vectors`. Das ist eine **Within-Arm-A/B** (gleicher embodied Learner) und damit compute- **und** embodiment-gematcht (Red-Team-Angriff 5: dies ist der **einzige** Kontrast, den M3 sauber beantwortet). Der Recombiner (M2) wird für M3 **nicht** als Headline-Null benutzt (Red-Team-Angriff 1: auf `n_functional_clusters` out-clustert der Monopolist jede fragmentierte Population by construction).

### Metrik (Within-Arm, hack-resistent)

**PRIMÄR:** `n_functional_clusters(QD-ON)` vs. `n_functional_clusters(value-VoI-OHNE-Archiv)` über gepaarten CI, **gekoppelt** an die M5 graded-Metrik (**nicht** das gefloorte DV2; Red-Team-Angriff 3 **belegt**: am Floor ist "kein DV2-Verlust" trivial erfüllt → Anti-Churn-Cap inert). Gültiger Erfolg verlangt **beide**: mehr funktionale Cluster **ohne** `graded_useful_advance`-Verlust.

### Kill-Kriterium (pre-quantifiziert, gated auf M5)

Gated auf M5 (Red-Team-Angriff 3: ohne M5-Floor-Reparatur feuert M3s eigene Floor-Klausel sofort durch Bestandsdaten). **Widerlegt**, wenn:
- QD-ON gegen value-VoI-OHNE-Archiv **keinen** gepaarten CI-Anstieg von `n_functional_clusters > 0` zeigt (`d_lo <= 0`); **oder**
- der Cluster-Zuwachs mit einem `graded_useful_advance`-**Rückgang** einhergeht (gepaarter CI < 0 → QD erkauft nutzlose Vielfalt); **oder**
- (übernommen) die M5-Metrik trotz QD am Floor bleibt → QD irrelevant für den bindenden Engpass.

### Label: notwendig-nicht-hinreichend für M1, FRAGIL

Diagnose Befund 4 zeigt, dass greedy Generierung ohne Diversitätsschutz kollabiert; QD entfernt diesen spezifischen Kollaps-Blocker **innerhalb** des learned Arm. **Nicht hinreichend und explizit FRAGIL:**
1. M3 kann **keine** Open-Endedness-Headline gegen einen compute/embodiment-gematchten Null liefern, weil `n_functional_clusters` den Monopolisten begünstigt (Red-Team-Angriff 1 = does-not-survive als Headline-DV); M3 ist auf die enge Within-Arm-Frage beschränkt: "fügt das Archiv funktionale Diversität über denselben Learner hinzu?".
2. Der Anti-Churn-Anspruch ist **nur** nach M5 valide (Angriff 3).
3. Die Diversitäts-Pressung **versiegt** bei abgedecktem fixem Gitter (Angriff 4).

### C-Mapping

C2: Trittsteine/Erreichbarkeit über QD-Deskriptoren; Novelty Search adressiert, dass nicht die Raumgröße, sondern das Finden von Trittsteinen der Engpass ist. C1: Diversity-Pressure als Anti-Disengagement-Garde für M1/M4. **Einschränkung (Angriff 4):** über die offene Material-Achse liefert das fixe Gitter **keine** dauerhafte Diversitäts-Pressung.

### Adressierte Diagnose-Befunde

Befund 4 (greedy collapse), Lever-Priorität #3; schützt M1 vor Modus-Kollaps.

### Red-Team-Zusammenfassung (Verdikt: fragil)

| Angriff | Kern | Antwort / Status |
|---|---|---|
| 1 — Monopolist gewinnt `n_functional_clusters` als Headline | **does-not-survive als Headline-DV** | nur Within-Arm-A/B; **keine** Open-Endedness-Headline |
| 2 — 250-tick-Transient (kumulativer monotoner Count) | **fragil** | 1500-tick-CI; aber DV2 als Durable-Check ist floored → M5-Gating + graded-Kopplung |
| 3 — DV2-Floor macht Anti-Churn-Cap inert | **does-not-survive** ohne M5 | **Korrigiert:** an M5 graded-Metrik gekoppelt + M5-Gating |
| 4 — statischer niedrigdim. Deskriptor / endliche Pressung | **fragil** (bounded) | Residual-Risiko explizit markiert; nur Kollaps-Blocker bis Gitter abgedeckt |
| 5 — Baseline nicht embodiment-matched | sauber beantwortet | Within-Arm-A/B ist compute- **und** embodiment-gematcht |
| 6 — C3-Persistenz / Elite-Replacement löscht Trittsteine | sauber beantwortet | **keine** Registry-Mutation; Archiv nur Auswahl-Bias |

**Fragile Stellen (markiert):** nur Within-Arm-Lever, **nicht** Headline; M5-abhängig; endliche Pressung über fixen Deskriptor.

---

## M4 — KOEVOLUTIONÄRE NICHT-STATIONARITÄT (C1), FRAGIL, M5-abhängig

**Mit OFFENER Task-Basis-Erweiterung (nicht nur Margin-Anhebung) + korrekt instrumentierter Anti-Disengagement-Garde.**

### Mechanismus (belegt + Red-Team-neukonstruiert, FRAGIL)

`TASK_BASIS` ist ein statisches 6-Ziel-Set (`metrics.py:261`), jede Utility ein Produkt geclampter Dims → Sättigung bei ~1.0 (**belegt**, Red-Team-Angriff 2). **Kernkorrektur:** bloße Anhebung von `ADVANCE_MARGIN`/Bedrohung hebt **nur die Latte, nicht das erreichbare Maximum** (utility ≤ 1.0 by construction) → kann den Floor strukturell **nicht** lösen. Daher koppelt M4 die Nicht-Stationarität an eine **offene Erweiterung der Task-Basis selbst:** neue Task-Dimensionen erscheinen, wenn die lebende Population die bestehenden sättigt (z. B. Komposit-Tasks aus erreichten Frontier-Artefakten, gespeist aus `DISCOVERY_REGISTRY` + `culture.population_sequences` `culture.py:120`) → **populations-gekoppelt** (C1; Red Queen Van Valen 1973 / POET Wang 2019, beide VERIFIZIERT). Neues `systems/coevolution.py` (self-registered, `registry.py:125`). Hall-of-Fame: Archiv bester je-erreichter Frontier-Artefakte.

**Wichtig (Red-Team-Angriff 7 + 1):** der nicht-stationäre Druck muss **sowohl** den In-Run-Reward **als auch** die offline-**Mess**-Frontier koppeln, sonst misst die Metrik ein statisches Objekt, während nur der Reward variiert.

### Baseline (compute- und embodiment-gematcht, vor dem Mechanismus)

Identisches System mit **statischer** Basis (coevolution-Flag OFF, byte-identisch zur Golden). Compute-gematcht: **keine** zusätzlichen `combine_vectors` (gleiches Erfindungs-Budget); die per-tick-Frontier-Berechnung **muss amortisiert/gecacht** sein (**nicht** 30k Registry-Entries pro tick re-clustern; Red-Team-Angriff 6 Feasibility). Regime ist **fixe tick-Zahl (1500)**, nicht wall-clock → ON verliert keine Ticks. Embodiment-gematchter Recombiner (M2) fährt unter **derselben** offenen Basis. Paired seeds 1001–1012, 1500 ticks.

### Metrik

M5 `graded_useful_advance` (**nicht** gefloortes DV2; Red-Team-Angriff 7: am Floor maskiert fehlendes Headroom einen echten Effekt als false-KILL) **über die Zeit** (Akkumulationssteigung).

**Anti-Disengagement-Check korrigiert (Red-Team-Angriff 4, belegt):** `useful_depth_max` über die **append-only** `DISCOVERY_REGISTRY` ist by construction monoton nicht-fallend → vacuous. Der Disengagement-Check wird daher über die **lebende Population** gemessen (`culture.population_sequences` `culture.py:120`, die bei Tod **schrumpft**), nicht über die unsterbliche Registry.

### Kill-Kriterium (pre-quantifiziert, gated auf M5)

Gated auf M5 (Floor/Headroom-Confound; Red-Team-Angriff 7). **Widerlegt**, wenn:
- die graded-Akkumulationssteigung unter offener nicht-stationärer Basis **nicht größer** ist als unter statischer Basis (gepaarter CI der End-Differenz `<= 0` über 12 seeds); **oder**
- Disengagement auftritt, gemessen über die **lebende Population** (`culture.population_sequences` fällt im letzten Drittel ggü. mittlerem Drittel, gepaart) → Hall-of-Fame hat sein Ziel verfehlt.

Smoke-Tests lügen (Transient); nur 1500-tick. Die Steigung- + Spät-Fenster-Instrumentierung fängt explizit den C+D-Reversal-Modus (Red-Team-Angriff 5 sauber beantwortet).

### Label: notwendig-nicht-hinreichend, FRAGIL

Eine statische 6-Ziel-Basis sättigt bei utility ≤ 1.0 → kein Druck für weitere Tiefe (plausible DV2-Floor-Ursache in **beiden** Armen). Offene Nicht-Stationarität entfernt diesen Sättigungs-Blocker. **Nicht hinreichend:** ohne erreichbare Trittsteine (M1/M3) erzeugt Nicht-Stationarität nur unerreichbare Ziele (Red-Team-Angriff 3, korrekt als notwendig-nicht-hinreichend gelabelt). **FRAGIL:** (1) die offene-Basis-Erweiterung ist neu und ungetestet; (2) der ganze Test ist erst nach M5 (Headroom) interpretierbar (Angriff 7); MLS/Gruppenselektion wird **nicht** als Haupt-Lever verkauft (C4-Regel).

### C-Mapping

C1 (Kernachse, korrigiert): **statisch vs. nicht-stationär** (an die eigene Population gekoppelt), mit Hall-of-Fame + Diversity-Pressure (M3) als Disengagement-Garde. C5: Koevolution macht Offenheit zur Eigenschaft des gekoppelten Systems. C2: offene Basis-Erweiterung statt bloßer Margin-Anhebung respektiert, dass das attainable maximum, nicht die Latte, der Engpass ist.

### Adressierte Diagnose-Befunde

C1 (Nicht-Stationarität), DV2-Floor-Saturation-Hypothese (Caveat A); liefert die Anti-Disengagement-Garde für M1/M3.

### Red-Team-Zusammenfassung (Verdikt: fragil)

| Angriff | Kern | Antwort / Status |
|---|---|---|
| 1 — Mess-Frontier bleibt statisch (`_base_frontier`), nur Reward variiert | **does-not-survive** in Originalform | **Korrigiert:** Reward **und** Mess-Frontier gemeinsam gekoppelt |
| 2 — versteckte statische Fitness (Ceiling utility ≤ 1.0) | **does-not-survive** in Originalform | **Korrigiert:** offene Basis-Erweiterung statt Margin-Anhebung |
| 3 — notwendig mit hinreichend verwechselt | korrekt gelabelt | überlebt als notwendig-nicht-hinreichend (nach Fix von 1+2) |
| 4 — Anti-Disengagement-Check über append-only Registry vacuous | **does-not-survive** in Originalform | **Korrigiert:** Check über lebende `population_sequences` |
| 5 — 250-tick-Transient/Reversal | gut verteidigt | Steigung + Spät-Fenster-Check über 1500 ticks fängt genau diesen Modus |
| 6 — versteckte per-tick-Compute-Asymmetrie | tick-gematcht, nicht wall-clock | überlebt; Feasibility-Caveat: Frontier-Berechnung amortisieren/cachen |
| 7 — Kill misst falsche (statische) Frontier; Floor maskiert Effekt | **does-not-survive** ohne M5 | **Korrigiert:** graded-Metrik + M5-Gating + gekoppelte Mess-Frontier |

**Fragile Stellen (markiert):** offene-Basis-Erweiterung neu/ungetestet; vollständig M5-abhängig (Headroom); Disengagement-Check zwingend über lebende Population.

---

## M6 — ERROR-THRESHOLD / Evolvabilitäts-Balance der Transmissions-Fidelity (bestätigender Sweep, herabgestuft), FRAGIL

**Kill-Kriterium auf DV3 verschoben; M1/M5-gated mit Positiv-Kontrolle.**

### Mechanismus (belegt + Red-Team-korrigiert, FRAGIL, bestätigend)

Fidelity ist bereits **hoch** und laut Diagnose (B) **nicht** bindend: `FIDELITY_BASE=0.72` (+`0.18*trust` → 0.90, `social_learning.py:26-27`), `INHERIT_FIDELITY=0.70` (`simulation.py:34`), `DEATH_BROADCAST_FIDELITY=0.45` (`simulation.py:35`), Korruption `culture.receive_transmitted` (`culture.py:80` `if random.random() > fidelity`), erfolgs-gewichtetes `sample_for_transmission` (`culture.py:64-76`). **Mechanismus (kein neuer Lever):** `FIDELITY_BASE`-Sweep über {0.45, 0.60, 0.72, 0.85, 0.95}, um die Eigen-Error-Threshold-Kurve (Eigen 1971, VERIFIZIERT) zu kartieren. Reiner Parameter-Sweep an `social_learning.py:26` + `simulation.py:34`, gekreuzt mit M1-ON.

**Einschränkung (Red-Team-Angriff 4, belegt):** `sample_for_transmission` gewichtet nach `successes` (`culture.py:64-76`) = **statische** Selektions-Pressung **unabhängig** von `FIDELITY_BASE` → der Sweep testet Copy-Noise gegen ein **stationäres** Transmissions-Ziel, nicht die Ratchet-Kopplung (C1 = M4s Aufgabe). Diese Geltungsgrenze **muss** in DIAGNOSE.md dokumentiert werden.

### Baseline (compute-gematcht, vor dem Mechanismus)

Live-Wert `FIDELITY_BASE=0.72` (`social_learning.py:26`) als Within-Arm-Referenz; compute-gematcht (Sweep ändert kein Erfindungs-Budget).

**Red-Team-Korrektur (Angriff 6):** zusätzliche **Kontroll-Zelle** M1-ON / Fidelity-fest-0.72, damit ein graded-Anstieg der **Fidelity-Achse** zugeschrieben werden kann und **nicht** M1 selbst (crossed-only kann beide nicht trennen). Paired seeds 1001–1012, 1500 ticks.

### Metrik (PRIMÄR auf DV3 verschoben)

Red-Team-Angriff 2 (**belegt**): das gefloorte DV2 ist durch Floor-Saturation überdeterminiert flach → bestätigt "B" aus dem **falschen** Grund. Daher **primär** DV3 `transmitted_frontier_advances` (`metrics.py:415`, fidelity-sensitiv by design, `k>=2`-Adoption `metrics.py:445`) als Funktion von `FIDELITY_BASE`, **plus** die M5 graded-Metrik als Sekundär.

**POSITIV-KONTROLLE PFLICHT:** DV3 muss bei **irgendeinem** Sweep-Wert nachweislich auf Fidelity reagieren, bevor ein flaches Resultat interpretiert wird.

### Kill-Kriterium (pre-quantifiziert, gated auf M5 UND M1-ON-Kontrollzelle)

Gated auf M5 (DV3 computable; Angriff 3) **und** M1-ON-Kontrollzelle (Angriff 6).
- Hypothese "Fidelity **nicht** bindend" (Diagnose B) **bestätigt** (M6 als Lever **verworfen**), wenn DV3 über den Sweep {0.45..0.95} flach bleibt (kein Wert hebt den gepaarten DV3-Median über 0.72, alle CIs überlappen) **und** die Positiv-Kontrolle zeigte, dass DV3 überhaupt fidelity-beweglich ist (sonst uninterpretierbar).
- **Umgekehrt (überraschend):** hebt ein Nicht-0.72-Wert DV3 über gepaarten `d_lo > 0` **und** schließt die M1-Kontrollzelle M1 als Ursache aus, ist (B) **teilweise falsch** und Fidelity wird aufgewertet.

Pre-quantifiziert: relevanter Effekt erst ab gepaartem `d_lo > 0`. Per-Arm-Computability von DV3 vor jeder Auswertung **asserten** (Angriff 3: Recombiner ohne Agenten wäre arm-asymmetrisch).

### Label: notwendig-nicht-hinreichend, HERABGESTUFT zu bestätigend, FRAGIL

Diagnose (B) belegt: Fidelity ist hoch und nicht bindend. M6 testet **nur**, dass die Fidelity-Achse kein Sub-Optimum ist; entfernt keinen primären Blocker. **FRAGIL:** (1) ohne M5 (DV3 computable + non-gefloortes DV2) ist M6 **nicht** ausführbar (Angriff 2 + 3, beide does-not-survive ohne M5); (2) die statische Selektions-Pressung (successes-Gewichtung) bedeutet, M6 misst Copy-Noise gegen ein statisches Ziel (Angriff 4, Geltungsgrenze). Hinreichendkeit **nicht** beansprucht.

### C-Mapping

C4: kumulative Kultur via hochfideler Transmission (unbestritten Kern) mit Eigen-Error-Threshold-Balance; **ohne** die drei Paare zu verwechseln (kulturell-vs-genetisch, Gruppe-vs-Individuum, Koordination-vs-IQ). MLS **nicht** als Haupt-Lever.

### Adressierte Diagnose-Befunde

(B) (Fidelity nicht bindend) als bestätigender Sweep; verankert die Eigen-Error-Threshold-Balance.

### Red-Team-Zusammenfassung (Verdikt: fragil)

| Angriff | Kern | Antwort / Status |
|---|---|---|
| 1 — blind recombiner ist fidelity-invariant, keine echte Null-Kalibrierung | within-arm-paired-Sweep | überlebt literalen Hack; Null-Kalibrierung bleibt Within-Arm-Referenz |
| 2 — DV2-Floor macht Bestätigung unfalsifizierbar | **does-not-survive** auf DV2 | **Korrigiert:** Kill auf DV3 verschoben + Positiv-Kontrolle |
| 3 — DV3 non-computable / arm-asymmetrisch | **does-not-survive** ohne M5 | **Korrigiert:** harte M5-Vorbedingung + per-Arm-Computability-Assert |
| 4 — statische Selektions-Pressung (successes) | Geltungsgrenze | dokumentieren in DIAGNOSE.md; scoped confirmatory survives |
| 5 — 250-tick-Transient/Reversal | 1500-tick-Regime gepinnt | überlebt |
| 6 — reverse branch confound mit M1 | crossed-only kann nicht trennen | **Korrigiert:** M1-ON/0.72-Kontrollzelle + `d_lo>0`-Schwelle |

**Fragile Stellen (markiert):** ohne M5 nicht ausführbar; statische-Selektions-Geltungsgrenze; nur bestätigend, kein primärer Blocker.

---

## Zusammenfassung: Mechanismus-Status

| ID | Rolle | Verdikt | Headline-fähig? | Harte Vorbedingung |
|---|---|---|---|---|
| **M5** | Messreparatur | fragil (HARTER BLOCKER) | — (Querschnitt) | keine (vorgezogen) |
| **M2** | Embodiment-gematchte Baseline | fragil | ja (Confound-Falsifikation) | M5-Floor-Test |
| **M1** | Selektive Generierung (Lever #1) | fragil | ja (CO-PRIMÄR) | M5-Floor-Test + Positiv-Kontrolle |
| **M3** | Diversitäts-Erhaltung (Within-Arm) | fragil | **nein** (nur Within-Arm-A/B) | M5 |
| **M4** | Nicht-Stationarität (C1) | fragil | bedingt | M5 + M1/M3 |
| **M6** | Fidelity-Sweep (bestätigend) | fragil | nein (herabgestuft) | M5 + M1-ON-Kontrollzelle |

Alle sechs Mechanismen tragen das Label **notwendig-nicht-hinreichend** und sind nach Red-Team-Korrektur als **fragil** klassifiziert. Kein Mechanismus beansprucht Hinreichendkeit; jeder trägt ein pre-quantifiziertes Kill-Kriterium, das seine spezifische Hypothese widerlegt. M5 ist der vorgezogene harte Blocker, ohne den M1–M4 und M6 uninterpretierbar bleiben.
