# EXPERIMENT-PLAN.md — Stage 0a: Open-ended kumulative funktionale Komplexitaet (gekoppeltes System)

> **Operatives Ziel.** Bedingungen identifizieren und testen, unter denen das GEKOPPELTE System (Learner + Umwelt) eine nachweisbare open-ended kumulative Akkumulation FUNKTIONALER Komplexitaet zeigt, gemessen gegen eine COMPUTE-GEMATCHTE Random-/Rekombinations-Baseline. KEINE "ueberlegene Spezies". Keine Garantie-Sprache an irgendeiner Stelle.

> **Bindende Schreibregeln (gelten fuer dieses Dokument).** Keine Garantie-/Unvermeidlichkeits-Sprache. Jeder Test ist eine Hypothese mit dem Vier-Tupel (1) Mechanismus auf den Code abgebildet, (2) null-kalibrierte compute-gematchte Baseline VOR dem Mechanismus definiert, (3) vordefinierte hack-resistente FUNKTIONALE Metrik, (4) PRE-QUANTIFIZIERTES Kill-Kriterium. Jeder Mechanismus ist `notwendig-nicht-hinreichend` gelabelt. "belegt" (code-verifiziert) wird strikt von "vermutet" getrennt.

---

## 0. Lese-Konventionen

- **belegt** = an einer realen `file:line`-Stelle code-verifiziert (Anker am Dokumentende).
- **vermutet** = plausible Hypothese, NICHT code-verifiziert; durch genau diesen Plan zu pruefen.
- **Compute-Regime (Default).** 12 gepaarte Seeds (1001–1012, `run_pilot.py:27` *belegt*), 1500 Ticks, Grid 30×20, Pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`, Ausfuehrung auf der GPU-PC via SSH. **250-Tick-Smokes luegen** (dokumentierter Transient: C+D lag bei 250 Ticks +35% vorne, kehrte sich bis 1500 um — `stage0a-cd-coupling-gonogo-2026-06-30.md`).
- **Gepaarte Statistik.** Gepaarter Bootstrap-CI ueber `analyze_gate._bootstrap_mean_ci` (`analyze_gate.py:41`, `N_BOOTSTRAP=10000` `analyze_gate.py:38` *belegt*). "Effekt" gilt erst ab gepaartem `d_lo > 0`.
- **Compute-Match-Achse (Direktive).** Gematcht wird die Zahl der REAL gezaehlten `combine_vectors`-Aufrufe (`materials.py:430` *belegt*), NICHT Wall-Clock. Bei fixer Tick-Zahl verliert kein Arm Ticks.

---

## 1. Prioritaets- und Abhaengigkeitslogik (billigste + haerteste Falsifikation zuerst)

Reihung folgt zwei Kriterien gleichzeitig: (i) Falsifikationskraft/Kosten-Verhaeltnis, (ii) harte Vorbedingungs-Kette. Die Kette ist code-belegt:

- **DV2 ist am Floor (= 2)** fuer den learned arm in JEDEM Arm/Seed (`stage0a-cd-coupling-gonogo-2026-06-30.md`; Diagnose-Caveat A). *belegt (Doc).* Das macht ein gefloortes Null-Resultat **uninterpretierbar**: "Mechanismus gescheitert" ist nicht von "Metrik hat keinen Headroom" unterscheidbar.
- **DV3 ist `computable=False`** (alle `discovered_by=-1` im Export; `transmitted_frontier_advances` setzt `computable=False`, `metrics.py:415/423`). *belegt.*
- Daraus folgt die Red-Team-Konsequenz: **M1, M3, M4, M6 sind auf der ungerepairten Messung uninterpretierbar bzw. ihr eigenes Kill-Kriterium feuert durch Bestandsdaten.** Deshalb ist **E1 (Messdiagnose) + die Messreparatur (M5) ein HARTER BLOCKER** und wird vorgezogen.

**Ausfuehrungs-Reihenfolge (Gate-Kette):**

| Reihenfolge | Experiment | Mechanismus | Kostenklasse | Vorbedingung |
|---|---|---|---|---|
| 1 | **E1** DV2-Floor-Diagnose OFFLINE | M5 (Teil a/c) | sehr billig (kein Re-Run) | keine |
| 2 | **E0** Embodiment-matched Rekombiner-Baseline | M2 (+ M5-Instrumentierung) | billig–mittel (1 Re-Run-Satz) | E1 bestanden (Floor-Test > 2) |
| 3 | **E2** Fidelity-Sweep (bestaetigend) | M6 (+ M5 fuer DV3) | mittel (5 Zellen × 12 Seeds) | E1 (DV3 computable) |
| 4 | **E3** Plan-treue seeded-softmax+VoI-Generierung | M1 | mittel | E1 (Floor-Test) + E0 (interner Null definiert) |
| 5 | **E4** QD/MAP-Elites-Generierung | M3 | mittel | E1 + E3 (Within-Arm A/B auf M1-Pfad) |
| 6 | **E5** Koevolutionaere Nicht-Stationaritaet | M4 | teuer | E1 + E3 |

> **Warum diese Reihung "hart" ist.** E1 ist quasi kostenlos und entscheidet, ob ueberhaupt eine interpretierbare Messung existiert — ohne sie sind E0/E3/E4/E5 nicht auswertbar. E0 ist der billigste + haerteste Confound-Test: faellt er, ist das gesamte "Lerndefizit" als Embodiment-Artefakt entlarvt (Path B), bevor teure Generierungs-Lever (E3–E5) gebaut werden.

---

## 2. Querschnitt-Garden (gelten in JEDEM Experiment)

Vor jedem Mechanismus VOR-definiert; in allen Experimenten als kontrollierte Confounds gefuehrt:

1. **Embodiment-Match.** Jeder Vergleich, der eine Lern-Aussage tragen soll, MUSS einen embodiment-gematchten internen Null besitzen (gleicher embodied Pfad, nur Auswahlregel/Parameter veraendert). Der disembodied Rekombiner (`recombiner.py:run_recombiner`) ist NUR orientierender EXTERNER Null, nie alleinige Entscheidungsbasis (Red-Team M1-Angriff 6b; M2-Angriff 5). *belegt: `recombiner.py:99` `discovered_by=-1`, `:101` `uses=0`, `:104` unconditional append.*
2. **Compute-Match.** Gezaehlte `combine_vectors` zwischen Armen identisch. Wenn ein Arm einen UNGEZAEHLTEN imaginierten Bewertungs-Twin nutzt, bekommt der Null-Arm denselben ungezaehlten Twin — aber mit UNIFORMER statt policy-gewichteter Auswahl, sodass "nur die Auswahlregel unterscheidet sich" ehrlich bleibt (Red-Team M2-Angriff 6, M1-Angriff 6).
3. **Metrik-Hackbarkeit.** Primaere DV ist eine non-gefloorte, arm-symmetrische funktionale Metrik (M5), KEINE per-Arm-Aggregation, die nur einem Arm zusteht. Mengen-/Churn-Hacking ist durch den Frontier-Margin-Mechanismus (`ADVANCE_MARGIN=0.02` `metrics.py:47`) ausgeschlossen — Masse redundanter Artefakte zaehlt 0 (`accumulated_useful_depth` `metrics.py:290`). *belegt.*
4. **Determinismus.** OFF-Pfad ist byte-identisch zur Golden-Trajektorie. Jeder imaginierte Suchlauf MUSS den globalen RNG-State snapshotten (`random.getstate()` / `random.setstate()`, Muster `recombiner.py:54/113`) und NETTO 0 RNG-Draws verbrauchen. *belegt: `combine_vectors` konsumiert den globalen Stream, `materials.py:452–455` `random.random()`/`random.uniform()`.*
5. **Keine `uses`-Pollution.** Imaginierte Vektor-Lookups duerfen NICHT `get_vector` (`materials.py:294–298`, inkrementiert `uses` bei JEDEM Lookup *belegt*) mit Seiteneffekt nutzen — nebenwirkungsfreier Getter / gecachte Vektoren.
6. **Transient-Sperre.** Verdikt nur aus terminalen 1500-Tick-Endpunkten ueber 12 gepaarte Seeds. Smokes zaehlen NICHT.

---

## E1 — DV2-Floor-Diagnose (OFFLINE) + Messreparatur-Validierung

**Testet Mechanismus:** **M5** (Messreparatur, vorgezogen als harter Blocker).
**Adressiert Diagnose:** Caveat A (DV2-Floor-Saettigung), Befund 1 (DV1 `uses`-Pollution / "fairness proof"), DV3 non-computability.

### Hypothese
*vermutet:* Das beobachtete DV2 = 2 fuer den learned arm ist (mindestens teilweise) ein **Messartefakt** — entweder Floor-Saettigung gegen die statische 6-Task-`TASK_BASIS` (`metrics.py:261` *belegt*, Utilities saettigen by construction bei ~1.0, z.B. `_u_edible_safe = edibility*(1-toxicity)` `metrics.py:237`) oder Fragmentierung ueber embodied Agenten. Eine tiefenaufloesende, arm-symmetrische Metrik (`graded_useful_advance`) kann unter positivem Tiefen-Input STRIKT > 2 ausgeben; `discovered_by`-Durchreichung macht DV3 `computable`.

### Manipulierte Achse
Nur die **Mess-/Instrumentierungsschicht** auf den BEREITS existierenden Pilot-Export-JSONs (n = 12, `stage0a-pilot-2026-06-29`). KEIN Re-Run fuer Teil (a)/(c). **Kein Verhaltens-Edit der Golden-Trajektorie.**

### Kontrollierte Confounds
- **Embodiment:** Die neue Metrik wird arm-symmetrisch definiert (gleiche Funktion auf learned-Entries UND Rekombiner-Entries; KEINE per-Agent-Mittelung, die nur dem learned arm zusteht — Red-Team M5-Angriff 3).
- **Metrik-Hackbarkeit:** `graded_useful_advance` erbt den Margin-/Churn-Schutz von `accumulated_useful_depth` (`metrics.py:290–343`); zaehlt pro Task die SUMME margin-normierter Frontier-Vorspruenge ueber ALLE Strukturschichten, statt binaer (>Frontier+Margin) zu deckeln — so kein Deckel bei Floor = 2, sobald 6 Tasks 1× getroffen sind.
- **Statische Fitness (Geltungsgrenze, dokumentiert):** Die Metrik bleibt eine Annaeherung an 6 vorregistrierte Tasks; sie misst "Akkumulation Richtung dieser 6 Ziele", nicht offene Akkumulation generell (Red-Team M5-Angriff 2). MUSS als Geltungsgrenze in `DIAGNOSE.md` stehen.

### Compute-gematchte Baseline (VOR dem Mechanismus)
Selbst-referentiell, da die Metrik DAS Artefakt ist: die bestehenden provisional/non-computable DVs (DV1 `provisional=True`, DV3 `computable=False`, DV2 floored) auf denselben Pilot-Exports. Es ist ein **Instrumentierungs-Unit-Test**, kein Wissenschaftsvergleich (Red-Team M5-Angriff 6).

### Erwartete Falsifikations-Signatur
Wenn `graded_useful_advance` unter **kontrolliertem positiv-synthetischem Tiefen-Input** strukturell NICHT > 2 ausgeben kann, ist die Floor-Reparatur fehlgeschlagen.

### Kill-Kriterium (PRE-QUANTIFIZIERT, vier Teile)
- **(a)** `graded_useful_advance` gibt unter positiv-synthetischem Input NICHT > 2 aus → Floor-Reparatur FEHLGESCHLAGEN; M1/M3/M4 bleiben uninterpretierbar (Knock-out-Validierung gegen `metrics.py:27–30`).
- **(b)** *(erfordert Mini-Re-Run, siehe unten)* `record_use`-via-Adoption `total_weight` des LEARNED arm bleibt ueber 1500 Ticks ~0 → DV1 strukturell leer → **DV1 wird endgueltig als nicht-rettbar markiert und aus ALLEN Gates entfernt.** *(belegt-Risiko: der einzige echte Inventar-Dekrement ist `sharp_stone` `agents/agent.py:752`; KEIN `mat_XXXX` wird homeostatisch verbraucht — Red-Team M5-Angriff 4. Deshalb haengt `record_use` an ADOPTIONS-Events `culture.receive_transmitted` `culture.py:80` + `brain.imitate_from` `brain.py:186`, NICHT an Inventar-Konsum.)*
- **(c)** embodiment-matched Null (E0) bleibt trotz Adoptions-Events DV1 = 0 (kein `discovered_by>=0`, kein `record_use`) → Instrumentierung nicht arm-symmetrisch → M2-DV1-Vergleich verworfen.
- **(d)** DV3 wird fuer KEINEN Seed `computable=True` → Export-Fix unvollstaendig (`discovered_by` vorhanden `materials.py:278`, im JSON gedroppt).

### Budget & Ausfuehrung
- Teil (a)/(c)/(d): **OFFLINE**, kein Re-Run, lokal oder GPU-PC.
- Teil (b) `record_use`-via-Adoption: EIN Re-Run-Satz, 12 gepaarte Seeds, 1500 Ticks, Standard-Regime, GPU-PC via SSH.

### Akzeptanzkriterien (Mechanismus-Erfolg)
(a) `graded_useful_advance` > Floor unter positivem Input UND arm-symmetrisch; (b) `weight_source='adoption'` > 0 fuer BEIDE Arme ODER explizit als arm-asymmetrisch deklariert; (c) DV3 `computable=True` weil `discovered_by>=0`.

**Label:** `notwendig-nicht-hinreichend` und **HARTER BLOCKER**. Bessere Messung entfernt einen Mess-Blocker, **erzeugt keine Komplexitaet** und dreht das Vorzeichen der Diagnose NICHT (Defizit ueberlebt die DV2-Reform per Diagnose A).

---

## E0 — Embodiment-matched Rekombiner-Baseline (decisive confound control)

**Testet Mechanismus:** **M2** (embodiment-matched, attribuierter, konsumierender Null), nutzt **M5**-Instrumentierung.
**Adressiert Diagnose:** EMBODIMENT-CONFOUND, Lever-Prioritaet #2. **Billigste + haerteste Einzel-Falsifikation des Gesamtprojekts.**

### Hypothese
*vermutet:* Wenn ein auf die Agenten-Constraints (Lokalitaet, Teilinformation, per-Tick-Attempt-Limit, Fragmentierung) GEDROSSELTER Rekombiner die learned-Population auf `graded_useful_advance` weiterhin schlaegt oder einholt, dann ist das "Lerndefizit" ein **Embodiment-Artefakt**, kein Lernversagen — und die Lernhypothese fuer diese Metrik ist widerlegt (Path B).

### Manipulierte Achse
Embodiment des Null-Arms: vom disembodied perfect-memory MONOPOLISTEN (globaler monoton wachsender Pool, `recombiner.py:104` unconditional append, `:99` `discovered_by=-1`, `:101` `uses=0`, Seed = alle 24 `MATERIALS`, RNG save/restore `:54–55/113` *belegt*) zu `run_recombiner_embodied` (neue Funktion, beruehrt KEINE Hot-Files): K = Pop disjunkte lokale Pools (Fragmentierung), per-Tick-Attempt-Budget, Teilinformation (nur Materialien in Zelle/Inventar), Pool-Austausch nur ueber sozialen Kanal mit `FIDELITY_BASE=0.72` (`social_learning.py:26` *belegt*).

### Kontrollierte Confounds
- **Embodiment:** Drei-Arm pre-registriert: **(1)** learned (mit M1, spaeter), **(2)** embodiment-FREIER Rekombiner (bestehend, extern-orientierend), **(3)** embodiment-MATCHED + ATTRIBUIERTER Rekombiner (neu). Einziger Unterschied (3)↔learned bleibt die fehlende Policy.
- **Attribution/Konsum (Red-Team M2-Angriff 1, M5-Angriff 1):** Der Null MUSS attribuiert + konsumierend sein, sonst ist DV1/DV3-Vergleich eingebaut-asymmetrisch: per-Agent `discovered_by>=0` setzen UND `record_use`-via-Adoptions-Events erzeugen (analog M5).
- **Compute (Red-Team M2-Angriff 6):** Arm (3) faehrt dieselbe Gesamtzahl REAL gezaehlter `combine_vectors` wie der learned arm. Nutzt M1 einen ungezaehlten `_combine_pure`-Imaginations-Twin, bekommt (3) denselben ungezaehlten Twin mit UNIFORMER Auswahl.
- **RNG (Red-Team M2-Angriff 4):** Pro-Agent-RNG-Streams aus Seed deriviert, deterministische Tick/Agent-Reihenfolge, unit-getestet (analog `cc.n==0`-Test der Phase D).
- **Metrik-Hackbarkeit (Red-Team M2-Angriff 1):** PRIMAER `graded_useful_advance` (M5), churn-immun. Volumen-Hacking ausgeschlossen.

### Compute-gematchte Baseline (VOR dem Mechanismus)
**M2 IST die Baseline.** Sie ist VOR jeder Lernbewertung definiert. Der disembodied Rekombiner erreicht laut gonogo-Doc DV2 ~32 NICHT durch Hack, sondern durch unbegrenzte Attempts + Monopol-Pool — genau das, was E0 drosselt.

### Erwartete Falsifikations-Signatur
Gepaarter CI `learned − (3)` auf `graded_useful_advance` NICHT > 0 (`d_lo <= 0`) → Embodiment-Artefakt.

### Kill-Kriterium (PRE-QUANTIFIZIERT, GATED auf E1)
- **Vorbedingung:** NUR ausfuehrbar, nachdem E1s `graded_useful_advance` den Floor-Test (> 2) bestanden hat. Sonst ist das Kill durch Floor-Saettigung konfundiert (Red-Team M2-Angriff 5 *code-belegt*: am Floor ist `learned − (3)` ~ 0 wegen Mess-Saettigung, NICHT wegen Embodiment).
- **Kill:** Erreicht (3) `graded_useful_advance >= learned` (gepaarter CI `learned − (3)` mit `d_lo <= 0`), ist das "Lerndefizit" ein EMBODIMENT-ARTEFAKT → Lernhypothese fuer diese Metrik WIDERLEGT, Befund als Konfundierung berichtet (**Path B**).
- **Echtes Lernsignal** nur, wenn `learned > BEIDE` Nullen (2) UND (3) ueber gepaarte CI auf der non-gefloorten Metrik.

### Metriken
- PRIMAER: `graded_useful_advance` (M5) je Arm.
- SEKUNDAER: DV3 `transmitted_frontier_advances` (`metrics.py:415`, nach E1 computable + attribuiertem Null).
- Gepaarter Bootstrap-CI (`analyze_gate.py:41`) fuer `learned−(2)` und `learned−(3)`.

### Budget & Ausfuehrung
12 gepaarte Seeds (1001–1012), 1500 Ticks, Grid 30×20, Pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`, GPU-PC via SSH. Drei Arme.

**Label:** `notwendig-nicht-hinreichend`. Ohne diese Kontrolle ist jeder `learned`-vs-Rekombiner-Vergleich uninterpretierbar. E0 erzeugt KEINE Komplexitaet (reine Messhygiene/Falsifikation); es macht die Lernhypothese falsifizierbar, ohne sie zu garantieren.

---

## E2 — Fidelity-Sweep `FIDELITY_BASE` 0.45..0.95 (bestaetigend, herabgestuft)

**Testet Mechanismus:** **M6** (Error-Threshold / Evolvabilitaets-Balance der Transmissions-Fidelity).
**Adressiert Diagnose:** Befund (B) — Fidelity ist hoch und NICHT bindend — als bestaetigenden Sweep.

### Hypothese
*belegt-Ausgangslage:* Fidelity ist bereits HOCH (`FIDELITY_BASE=0.72` `social_learning.py:26`, `+0.18*trust → 0.90` `:27`; `INHERIT_FIDELITY=0.70` `simulation.py:34`; `DEATH_BROADCAST_FIDELITY=0.45` `simulation.py:35`; Korruption `culture.py:80` `if random.random() > fidelity`; erfolgs-gewichtetes `sample_for_transmission` `culture.py:64–76`). *vermutet:* Die Fidelity-Achse ist kein Sub-Optimum — `DV3` bleibt ueber den Sweep flach.

### Manipulierte Achse
Reiner Parameter-Sweep `FIDELITY_BASE ∈ {0.45, 0.60, 0.72, 0.85, 0.95}` an `social_learning.py:26` (+ `simulation.py:34`), gekreuzt mit M1-ON.

### Kontrollierte Confounds
- **M1-Confound (Red-Team M6-Angriff 6):** ZUSAETZLICHE Kontroll-Zelle **M1-ON / Fidelity-fest-0.72**, damit ein graded-Anstieg der FIDELITY-Achse zugeschrieben werden kann und NICHT M1 selbst. Crossed-only kann beide nicht trennen.
- **Metrik (Red-Team M6-Angriff 2 *code-belegt*):** PRIMAER NICHT das gefloorte DV2 (durch Floor-Saettigung ueberdeterminiert flach → wuerde "(B)" aus dem FALSCHEN Grund bestaetigen), sondern **DV3** (`metrics.py:415`, fidelity-sensitiv by design, `k>=2` Adoption `metrics.py:445`), plus M5 `graded` als Sekundaer.
- **Positiv-Kontrolle (PFLICHT):** DV3 muss bei IRGENDeinem Sweep-Wert nachweislich auf Fidelity reagieren, BEVOR ein flaches Resultat interpretiert wird; sonst uninterpretierbar.
- **Per-Arm-Computability (Red-Team M6-Angriff 3):** DV3-Computability VOR jeder Auswertung per-Arm asserten (Rekombiner ohne Agenten waere arm-asymmetrisch).
- **Statische Selektions-Pressung (Geltungsgrenze, Red-Team M6-Angriff 4 *code-belegt*):** `sample_for_transmission` gewichtet nach `successes` (`culture.py:64–76`) = STATISCHE Selektion unabhaengig von `FIDELITY_BASE` → der Sweep testet Copy-Noise gegen ein STATIONAERES Transmissions-Ziel, NICHT die Ratchet-Kopplung (C1 = E5s Aufgabe). MUSS in `DIAGNOSE.md` dokumentiert werden.
- **Transient (Red-Team M6-Angriff 5):** Hohe-Fidelity-Zelle (0.95) kann fruehen DV3-Bump zeigen, der bis 1500 kollabiert. Nur 1500-Tick-Endpunkte zaehlen.

### Compute-gematchte Baseline (VOR dem Mechanismus)
Live-Wert `FIDELITY_BASE=0.72` als Within-Arm-Referenz; compute-matched (Sweep aendert kein Erfindungs-Budget). Plus die M1-ON/0.72-Kontroll-Zelle (oben).

### Erwartete Falsifikations-Signatur
DV3 ueber {0.45..0.95} flach (kein Wert hebt den gepaarten DV3-Median ueber 0.72, alle CIs ueberlappen) — bei nachgewiesener Positiv-Kontrolle.

### Kill-Kriterium (PRE-QUANTIFIZIERT, GATED auf E1 + M1-ON-Kontrollzelle)
- Hypothese "Fidelity NICHT bindend" (Diagnose B) **BESTAETIGT** (M6 als Lever VERWORFEN), wenn DV3 ueber den Sweep flach bleibt UND die Positiv-Kontrolle zeigte, dass DV3 ueberhaupt fidelity-beweglich ist.
- **UMGEKEHRT (ueberraschend):** Hebt ein Nicht-0.72-Wert DV3 ueber gepaarten `d_lo > 0` UND die M1-Kontrollzelle schliesst M1 als Ursache aus → (B) teilweise falsch, Fidelity wird aufgewertet.
- Relevanter Effekt erst ab gepaartem `d_lo > 0`.

### Budget & Ausfuehrung
5 Sweep-Zellen + 1 M1-Kontrollzelle, je 12 gepaarte Seeds, 1500 Ticks, Standard-Regime, GPU-PC via SSH.

**Label:** `notwendig-nicht-hinreichend` und **HERABGESTUFT zu bestaetigend, FRAGIL.** Ohne E1 (DV3 computable + non-gefloortes DV2) NICHT ausfuehrbar. Misst Copy-Noise gegen ein statisches Ziel (Geltungsgrenze). Hinreichendkeit nicht beansprucht. MLS NICHT als Haupt-Lever (C4-Regel; die drei Paare cultural-vs-genetic, group-vs-individual, coordination-vs-IQ NICHT verwechseln).

---

## E3 — Plan-treue selektive Generierung: seeded-softmax + Value-of-Information

**Testet Mechanismus:** **M1** (policy-gekoppelte selektive Generierung, NICHT greedy-argmax).
**Adressiert Diagnose:** (C) ACTUAL BINDING ENGPASS (Policy waehlt WANN, nicht WAS), Lever-Prioritaet #1. Korrigiert den greedy-argmax aus Path-A Phase D.

### Hypothese
*belegt:* Die PPO-Policy entscheidet das WANN (`need_driven_invention.py:351` `if need_magnitude < eff_threshold`), das WAS ist quasi-zufaellig (`_select_materials_by_need`/`_select_action_by_need`, `need_driven_invention.py:305–310` softmax + `np.random.choice`). *vermutet:* Koppelt man das WAS plan-treu (imaginiertes means-ends ueber `value(_need_score) + voi_weight*novelty`, Auswahl per SEEDED-SOFTMAX statt argmax), steigt `graded_useful_advance` ueber den embodiment-matched internen Null — OHNE den greedy-Kollaps der Path-A Phase D (Effizienz fiel 0.127 → 0.029, gonogo-Doc; *Anmerkung: das ist Invention-EFFIZIENZ = n_discoveries / executed combine_vectors, OFF-Arm 0.127 vs ON-Arm 0.029, KEINE Diversitaets-/Temporalmetrik*).

### Manipulierte Achse
Ungezaehlter imaginierter Zwilling `_combine_pure` (reine Funktion ueber `combine_vectors` `materials.py:430`, OHNE `DISCOVERY_REGISTRY.register`/`uses`-Inkrement), bewertet kleines Kandidatenset, Auswahl per seeded-softmax → DANN EINE real gezaehlte `combine_vectors` + register. Gegated per env-Flag (`AS_PATHA_CD`-Muster), OFF byte-identisch zur Golden.

### Kontrollierte Confounds
- **Determinismus / RNG-Leck (Red-Team M1-Angriff 6 *code-belegt*):** `combine_vectors` konsumiert den globalen `random`-Stream (`materials.py:452–455`). Der Zwilling MUSS `random.getstate()` vor dem K-Kandidaten-Sweep snapshotten und `random.setstate()` danach (`recombiner.py:54/113`-Muster) → imaginierte Suche NETTO 0 RNG-Draws; die EINE reale `combine_vectors` zieht aus demselben Stream-Punkt wie OFF. **Unit-Test:** globaler RNG-State byte-identisch vor/nach dem Sweep (NICHT nur "reale Aufrufe == 0").
- **`uses`-Pollution (Red-Team M1-Angriff 4 *code-belegt*):** Zwilling liest Vektoren OHNE `uses`-Inkrement (`get_vector` `materials.py:294–298` inkrementiert bei JEDEM Lookup) — nebenwirkungsfreier Getter/gecachte Vektoren.
- **Metrik-Tautologie (Red-Team M1-Angriff 1):** Ein `RATCHET_GAIN`-Term wuerde den Generator an die DV2-Scoring-Funktion koppeln (Tautologie-Risiko). KORREKTUR: PRIMAERE Metrik ist eine **HELD-OUT Task-Basis disjunkt** von der, gegen die der Generator optimiert (Generator nutzt nur `value+novelty`), ODER `RATCHET_GAIN=0`.
- **Embodiment (Red-Team M1-Angriff 6b):** PRIMAERE Entscheidungs-Baseline ist der **embodiment-matched INTERNE Null** (= derselbe embodied Pfad mit `voi_weight=0` und `RATCHET_GAIN=0`, uniformes Kandidaten-Sampling, gleiche Anzahl imaginierter + realer Aufrufe), NICHT der Rekombiner. Der Rekombiner (E0 Arm 2/3) ist nur orientierender EXTERNER Null.
- **Floor-Confound (Red-Team M1-Angriff 2 = does-not-survive ohne E1):** GATED auf E1. Vor jeder Null-Interpretation **Positiv-Kontrolle PFLICHT**: die graded-Metrik muss unter IRGENDeiner Intervention nachweislich > Floor beweglich sein.

### Compute-gematchte Baseline (VOR dem Mechanismus)
Embodiment-matched interner Null (oben). Compute-Match-Integritaet per Unit-Test: (1) imaginierte `combine_vectors` real gezaehlt == 0; (2) ON-Arm reale Aufrufe <= OFF; (3) globaler RNG-State identisch vor/nach dem imaginierten Sweep.

### Erwartete Falsifikations-Signatur
Gepaarter CI `graded(VoI-ON) − graded(VoI-OFF intern)` NICHT > 0 (`d_lo <= 0`) → selektive Generierung fuer diese Metrik widerlegt; ODER `n_functional_clusters(ON) < n_functional_clusters(OFF)` (greedy-Herding).

### Kill-Kriterium (PRE-QUANTIFIZIERT, GATED auf E1, CO-PRIMAER)
- **Vorbedingung:** NUR auswertbar, nachdem E1s graded-Metrik den Floor-Test (> 2) bestand (Red-Team M1-Angriff 2).
- **WIDERLEGT**, wenn ueber 12 gepaarte Seeds bei 1500 Ticks der gepaarte CI von `graded_useful_advance(VoI-ON) − graded(VoI-OFF intern)` NICHT > 0 (`d_lo <= 0`).
- **ZUSATZ-Kill (Anti-Greedy, gegen C+D-Kollaps):** selbst bei Anstieg widerlegt, wenn `n_functional_clusters(ON) < n_functional_clusters(OFF)` (`metrics.py:144`) ueber gepaarten CI.
- **Smoke-Tests (250 Ticks) zaehlen NICHT** (Transient luegt; C+D war +35% bei 250, kollabierte bei 1500).

### Metriken
CO-PRIMAER (Red-Team M1-Angriff 2: ein gefloortes Null-Resultat widerlegt nichts): M5 `graded_useful_advance` (held-out-Task-Basis) UND `n_functional_clusters` (`metrics.py:144`) gemeinsam. Gefloortes Original-DV2 nur als Floor-Diagnostik. Gepaarter Bootstrap-CI (`analyze_gate.py:41`).

### Budget & Ausfuehrung
12 gepaarte Seeds, 1500 Ticks, Standard-Regime, GPU-PC via SSH. ON vs OFF-intern (mind. zwei Arme; Rekombiner-Arme aus E0 als orientierende externe Nullen mitgefuehrt).

**Label:** `notwendig-nicht-hinreichend`. (C) identifiziert die fehlende WAS-Kopplung als bindenden Engpass; ohne sie ist der learned arm ein compute-gedrosselter fragmentierter random recombiner. **NICHT hinreichend:** der plan-treue Lever ist laut gonogo-Doc UNGETESTET; greedy-argmax verschlechterte die Effizienz. M1 adressiert nur die Generator-Seite, nicht Transmission/Akkumulation ueber die Population (Red-Team M1-Angriff 3) → das Floor-Kill auf der graded-Metrik widerlegt Suffizienz korrekt. C3-Persistenz ist unabhaengig PASSED (`simulation.py:122` `_reset_accumulating_singletons()` einmalig, `:33` `INHERIT_SEQUENCES=10`, globales `DISCOVERY_REGISTRY` `materials.py:325`).

---

## E4 — Diversitaets-erhaltende Generierung (QD / MAP-Elites / Novelty)

**Testet Mechanismus:** **M3** (diversitaets-erhaltende Generierung), NUR als WITHIN-ARM A/B-Lever, FRAGIL.
**Adressiert Diagnose:** Befund 4 (greedy means-ends herdet auf wenige high-value combos), Lever-Prioritaet #3. Schuetzt E3/M1 vor Modus-Kollaps.

### Hypothese
*belegt-Ausgangslage:* Das C+D-NO-GO zeigte greedy-Herding (Diagnose Befund 4). *vermutet:* Ein MAP-Elites-Archiv (Mouret & Clune 2015, VERIFIZIERT) ueber Deskriptor = (dominante aktive Property-Dim via `mean_active_dims` `metrics.py:156`, Action-Klasse aus `PRIMITIVE_ACTIONS` `invention.py:52` *belegt*) fuegt, in M1s imaginierte Suche als Novelty/Archiv-Bonus eingespeist (Lehman & Stanley 2011, VERIFIZIERT), funktionale Diversitaet ueber DENSELBEN embodied Learner hinzu, OHNE `graded`-Verlust.

### Manipulierte Achse
In M1s imaginierter Suche wird seeded-softmax um einen Novelty/Archiv-Bonus erweitert; Elite-Ersetzung pro Zelle nach best task-utility. Neues Modul `systems/qd_archive.py` (self-registered, `registry.py:125`), gespeist aus `DISCOVERY_REGISTRY.entries`; greift NUR in M1s Auswahl-Score ein (KEIN Hot-File-Edit, KEINE Registry-Mutation → C3-Persistenz intakt, Red-Team M3-Angriff 6 sauber).

### Kontrollierte Confounds
- **Embodiment + Compute (Red-Team M3-Angriff 5 — der EINZIGE Kontrast, den M3 sauber beantwortet):** WITHIN-ARM A/B (gleicher embodied Learner) gegen M1 mit reinem value-VoI OHNE Archiv (`voi_weight>0`, `novelty_archive_weight=0`), compute-matched auf identische imaginierte + reale `combine_vectors`. Der Rekombiner (E0) wird fuer M3 NICHT als Headline-Null benutzt (Red-Team M3-Angriff 1 *code-belegt*: auf `n_functional_clusters` `metrics.py:144` out-clustert der Monopolist jede fragmentierte Population by construction).
- **Floor / Anti-Churn-Inertheit (Red-Team M3-Angriff 3 *code-belegt*):** GATED auf E1. Ohne E1-Floor-Reparatur ist die "kein DV2-Verlust"-Klausel am Floor trivial erfuellt → Anti-Churn-Cap inert. Daher an die M5 `graded`-Metrik gekoppelt, NICHT an gefloortes DV2.
- **Transient + monotone Cumulative-Count (Red-Team M3-Angriff 2):** `n_functional_clusters` ist ein kumulativer Registry-Count (Entries werden nie entfernt) → ein frueher QD-Bloom kann am Ende eingeloggt bleiben, auch wenn die Rate kollabierte. Daher gepaarter CI am 1500-Tick-Horizont UND Kopplung an `graded` als Durable-Check.
- **Statische Deskriptor-Fitness (Geltungsgrenze, Red-Team M3-Angriff 4 *code-belegt*):** Deskriptor ist statisch + niedrigdimensional (1 Property-Dim + 8 fixe Actions); echte Ausdruckskraft liegt in OFFENER Material-Rekombination (Diagnose Befund 7). Ist das fixe Gitter abgedeckt, geht der Novelty-Bonus auf 0 und QD degeneriert zum value-VoI-Sampler → nur ENDLICHE Diversitaets-Pressung. RESIDUAL-RISIKO, explizit markiert.

### Compute-gematchte Baseline (VOR dem Mechanismus)
M1 mit reinem value-VoI-Sampler OHNE Archiv/Novelty-Bonus, compute- UND embodiment-matched (Within-Arm).

### Erwartete Falsifikations-Signatur
QD-ON gegen value-VoI-OHNE-Archiv zeigt KEINEN gepaarten CI-Anstieg von `n_functional_clusters > 0`; ODER Cluster-Zuwachs mit `graded`-RUECKGANG; ODER `graded` bleibt trotz QD am Floor.

### Kill-Kriterium (PRE-QUANTIFIZIERT, GATED auf E1)
WIDERLEGT, wenn (a) `n_functional_clusters(QD-ON) − (value-VoI-OHNE-Archiv)` gepaarter CI `d_lo <= 0`; ODER (b) Cluster-Zuwachs geht mit `graded`-RUECKGANG einher (gepaarter CI < 0 → QD erkauft nutzlose Vielfalt); ODER (c) `graded` bleibt trotz QD am Floor → QD irrelevant fuer den bindenden Engpass. Gueltiger Erfolg verlangt BEIDE: mehr funktionale Cluster OHNE `graded`-Verlust.

### Metriken
PRIMAER (Within-Arm, hack-resistent): `n_functional_clusters(QD-ON)` vs OHNE-Archiv ueber gepaarten CI, GEKOPPELT an M5 `graded` (NICHT gefloortes DV2). Gepaarter Bootstrap-CI.

### Budget & Ausfuehrung
12 gepaarte Seeds, 1500 Ticks, Standard-Regime, GPU-PC via SSH. QD-ON vs value-VoI-OHNE-Archiv (Within-Arm).

**Label:** `notwendig-nicht-hinreichend`, **FRAGIL.** M3 entfernt den greedy-Kollaps-Blocker INNERHALB des learned arm. **NICHT hinreichend und EXPLIZIT FRAGIL:** (1) M3 kann KEINE Open-Endedness-Headline gegen einen compute/embodiment-matched Null liefern (`n_functional_clusters` beguenstigt den Monopolisten — Red-Team M3-Angriff 1 = does-not-survive als Headline-DV); beschraenkt auf "fuegt das Archiv funktionale Diversitaet ueber denselben Learner hinzu". (2) Anti-Churn-Anspruch nur valide nach E1. (3) Diversitaets-Pressung versiegt bei abgedecktem fixem Gitter.

---

## E5 — Koevolutionaere Nicht-Stationaritaet (C1) mit offener Task-Basis-Erweiterung

**Testet Mechanismus:** **M4** (population-gekoppelte Nicht-Stationaritaet + Anti-Disengagement-Garde), FRAGIL.
**Adressiert Diagnose:** C1 (STATISCH vs NICHT-STATIONAER), DV2-Floor-Saturation-Hypothese (Caveat A), Anti-Disengagement-Garde fuer E3/E4.

### Hypothese
*belegt:* `TASK_BASIS` ist statisches 6-Ziel-Set (`metrics.py:261` *belegt*), jede Utility ein Produkt geclampter Dims → Saettigung bei ~1.0 (`_u_edible_safe = edibility*(1-toxicity)` `metrics.py:237` *belegt*). *vermutet:* Eine blosse Anhebung von `ADVANCE_MARGIN`/Bedrohung hebt NUR die Latte, NICHT das erreichbare Maximum (`utility <= 1.0` by construction) und kann den Floor strukturell NICHT loesen (Red-Team M4-Angriff 2 *code-belegt*). Daher koppelt M4 die Nicht-Stationaritaet an eine OFFENE Erweiterung der Task-Basis selbst: neue Task-Dimensionen erscheinen, wenn die LEBENDE Population die bestehenden saettigt (Komposit-Tasks aus erreichten Frontier-Artefakten, gespeist aus `DISCOVERY_REGISTRY` + `culture.population_sequences` `culture.py:120` *belegt*) → population-gekoppelt (C1, Red Queen Van Valen 1973 / POET Wang 2019, VERIFIZIERT). Dies hebt die Akkumulationssteigung von `graded` ueber die statische Basis.

### Manipulierte Achse
Neues `systems/coevolution.py` (self-registered, `registry.py:125`): offene Task-Basis-Erweiterung + Hall-of-Fame (Archiv bester je-erreichter Frontier-Artefakte). **WICHTIG (Red-Team M4-Angriff 7/1):** der nicht-stationaere Druck MUSS SOWOHL den In-Run-Reward ALS AUCH die offline-MESS-Frontier koppeln, sonst misst die Metrik ein statisches Objekt, waehrend nur der Reward variiert.

### Kontrollierte Confounds
- **Metrik misst falsche Frontier (Red-Team M4-Angriff 1/7 *code-belegt*):** Der disembodied Rekombiner hat keinen Reward-Kanal — nicht-stationaerer Reward ist fuer ihn ein No-op. DV2s interne Frontier (`_base_frontier` aus `MATERIALS` `metrics.py:280–289`) ist STATISCH und in beiden Armen identisch. Deshalb: Reward UND Mess-Frontier gemeinsam koppeln; der embodiment-matched Rekombiner (E0) faehrt unter DERSELBEN offenen Basis.
- **Headroom / Floor (Red-Team M4-Angriff 7 *code-belegt*):** GATED auf E1. Am Floor maskiert fehlendes Headroom einen echten Effekt als false-KILL → M5 `graded` (non-gefloort) statt DV2.
- **Anti-Disengagement vacuous (Red-Team M4-Angriff 4 *code-belegt*):** `useful_depth_max` ueber die APPEND-ONLY `DISCOVERY_REGISTRY` (`materials.py:325`) ist monoton nicht-fallend by construction → der "darf nicht fallen"-Check ist unfalsifizierbar. KORREKTUR: Disengagement-Check ueber die LEBENDE Population (`culture.population_sequences` `culture.py:120`, schrumpft bei Tod), NICHT ueber die unsterbliche Registry.
- **Compute / Feasibility (Red-Team M4-Angriff 6):** KEINE zusaetzlichen `combine_vectors` (gleiches Erfindungs-Budget); per-Tick-Frontier-Berechnung MUSS amortisiert/gecacht sein (NICHT ~30k Registry-Entries pro Tick re-clustern). Regime ist FIXE Tick-Zahl (1500), nicht Wall-Clock → ON verliert keine Ticks.
- **Transient (Red-Team M4-Angriff 5):** Smokes luegen; Steigung + Spaet-Fenster-Instrumentierung faengt explizit den C+D-Reversal-Modus. Nur 1500-Tick.
- **Reachability (Red-Team M4-Angriff 3):** ohne erreichbare Trittsteine (E3/E4) erzeugt Nicht-Stationaritaet nur unerreichbare Ziele → notwendig-nicht-hinreichend.

### Compute-gematchte Baseline (VOR dem Mechanismus)
Identisches System mit STATISCHER Basis (coevolution-Flag OFF, byte-identisch zur Golden). Embodiment-matched Rekombiner (E0) faehrt unter DERSELBEN offenen Basis.

### Erwartete Falsifikations-Signatur
`graded`-Akkumulationssteigung unter offener nicht-stationaerer Basis NICHT groesser als unter statischer Basis (gepaarter CI der End-Differenz `<= 0`); ODER Disengagement (lebende `culture.population_sequences` faellt im letzten Drittel ggue. mittlerem Drittel).

### Kill-Kriterium (PRE-QUANTIFIZIERT, GATED auf E1)
WIDERLEGT, wenn (a) gepaarter CI der End-Differenz der `graded`-Akkumulationssteigung (offen − statisch) ueber 12 Seeds `<= 0`; ODER (b) Disengagement auftritt, gemessen ueber die LEBENDE Population (`culture.population_sequences` faellt im letzten Drittel ggue. mittlerem Drittel, gepaart) → Hall-of-Fame hat sein Ziel verfehlt. Smokes luegen; nur 1500-Tick.

### Metriken
M5 `graded_useful_advance` ueber die ZEIT (Akkumulationssteigung), NICHT gefloortes DV2. Anti-Disengagement-Check ueber lebende Population (`culture.population_sequences` `culture.py:120`).

### Budget & Ausfuehrung
12 gepaarte Seeds, 1500 Ticks, Standard-Regime, GPU-PC via SSH. Offene vs statische Basis (+ embodiment-matched Rekombiner unter offener Basis).

**Label:** `notwendig-nicht-hinreichend`, **FRAGIL.** Eine statische 6-Ziel-Basis saettigt bei `utility <= 1.0` → kein Druck fuer weitere Tiefe (plausible DV2-Floor-Ursache in BEIDEN Armen). **NICHT hinreichend:** ohne erreichbare Trittsteine (E3/E4) erzeugt Nicht-Stationaritaet nur unerreichbare Ziele. FRAGIL: die offene-Basis-Erweiterung ist neu und ungetestet; der ganze Test ist erst nach E1 (Headroom) interpretierbar. MLS/Gruppenselektion NICHT als Haupt-Lever (C4-Regel).

---

## 3. Experiment → Mechanismus → Diagnose-Mapping (Uebersicht)

| Exp | Mechanismus | Diagnose-Befund | Lever-Prio | Vorbedingung | Entscheidungs-DV | Verdikt-Modus |
|---|---|---|---|---|---|---|
| **E1** | M5 | Caveat A (Floor), Befund 1 (DV1), DV3 non-comp. | Blocker | — | `graded` > 2 unter Synthetik; DV3 `computable`; DV1-Adoption | Mess-Gate |
| **E0** | M2 (+M5) | EMBODIMENT-CONFOUND | #2 | E1 | `graded`: `learned − (3)` | Path-B-Falsifikation |
| **E2** | M6 (+M5) | (B) Fidelity nicht bindend | herabgestuft | E1 | DV3 vs `FIDELITY_BASE` + M1-Kontrollzelle | bestaetigend |
| **E3** | M1 | (C) WAS-Kopplung | #1 | E1, E0 | `graded(ON−OFF intern)` + `n_functional_clusters` | Lern-Hypothese |
| **E4** | M3 | Befund 4 (greedy collapse) | #3 | E1, E3 | `n_functional_clusters` + `graded` (Within-Arm) | Anti-Kollaps-Lever |
| **E5** | M4 | C1 + Caveat A | flankierend | E1, E3 | `graded`-Steigung + lebende Population | Nicht-Stationaritaet |

> **Gesamt-Entscheidungslogik.** Faellt **E0** (embodiment-matched Null `>=` learned auf der non-gefloorten Metrik), ist der "Lerndefizit"-Befund als Embodiment-Konfundierung berichtet → **Path B** (Methodik-Paper), die Generierungs-Lever E3–E5 werden NICHT als Open-Endedness-Headline weiterverfolgt. Besteht E0 (learned > BEIDE Nullen), liefern E3 (#1) → E4 (#3) → E5 (flankierend) die gestufte Pruefung der bindenden WAS-Kopplung, des greedy-Kollaps-Schutzes und der population-gekoppelten Nicht-Stationaritaet. E2 bleibt durchgehend ein bestaetigender Seitenarm (Fidelity nicht bindend).

---

## 4. Belegt vs vermutet — Zusammenfassung

**Belegt (code-verifiziert, Anker):** DV2-Floor in Bestandsdaten; `discovered_by=-1`/`uses=0`/unconditional add im Rekombiner; `combine_vectors` konsumiert globalen RNG-Stream; `get_vector` inkrementiert `uses`; einziger Inventar-Dekrement `sharp_stone`; `TASK_BASIS` statisch + saettigend; `_base_frontier` statisch; `useful_depth_max` ueber append-only Registry monoton; `sample_for_transmission` statisch successes-gewichtet; C3-Persistenz (kein Episode-Reset, einmaliges `_reset_accumulating_singletons`, Sequenz-Vererbung); Fidelity-Konstanten hoch.

**Vermutet (durch diesen Plan zu pruefen):** dass `graded_useful_advance` den Floor strukturell loest; dass das Defizit (teilweise) ein Embodiment-Artefakt ist; dass plan-treue seeded-softmax+VoI-Generierung `graded` ueber den internen Null hebt; dass QD funktionale Diversitaet ohne `graded`-Verlust hinzufuegt; dass offene Nicht-Stationaritaet die Akkumulationssteigung hebt; dass Fidelity nicht bindend ist.

---

## Anhang A — Code-Anker (Ground Truth)

| Schluessel | Datei:Zeile |
|---|---|
| `structural_depths` | `artificial_society/research/metrics.py:85` |
| `functional_depths` | `artificial_society/research/metrics.py:115` |
| `FUNC_TAU = 0.15` | `artificial_society/research/metrics.py:45` |
| `DEDUP_TAU = 0.08` | `artificial_society/research/metrics.py:44` |
| `ADVANCE_MARGIN = 0.02` | `artificial_society/research/metrics.py:47` |
| `accumulated_useful_depth` (DV2) | `artificial_society/research/metrics.py:290` |
| `_u_edible_safe`-Saettigung | `artificial_society/research/metrics.py:237` |
| `n_functional_clusters` / `mean_active_dims` | `artificial_society/research/metrics.py:144` / `:156` |
| `population_functional_value` (DV1) | `artificial_society/research/metrics.py:346` |
| `transmitted_frontier_advances` (DV3) | `artificial_society/research/metrics.py:415` (`k>=2` `:445`) |
| `TASK_BASIS` | `artificial_society/research/metrics.py:261` |
| `_base_frontier` (statisch) | `artificial_society/research/metrics.py:280` |
| `analyze_registry` | `artificial_society/research/metrics.py:450` |
| `N_BOOTSTRAP = 10000` / `_bootstrap_mean_ci` | `artificial_society/research/analyze_gate.py:38` / `:41` |
| Verdikte PATH_A / BORDERLINE / PATH_B | `analyze_gate.py:91` / `:93` / `:95` |
| Invention-Effizienz (0.127/0.029) | `artificial_society/research/analyze_learning.py:71` |
| Rekombiner: pool-grow / uniform action / mat_a/mat_b / dedup / unconditional add / `discovered_by=-1` / `uses=0` / RNG save | `recombiner.py:69` / `:78` / `:76` / `:84` / `:104` / `:99` / `:101` / `:54–55,113` |
| `DEFAULT_SEEDS` 1001–1012 | `artificial_society/research/run_pilot.py:27` |
| while-loop / einmaliges Reset / `INHERIT_SEQUENCES=10` / `INHERIT_FIDELITY=0.70` / `DEATH_BROADCAST_FIDELITY=0.45` / Sequenz-Vererbung | `simulation.py:493–513` / `:122` / `:33` / `:34` / `:35` / `:182` |
| `cultural_diversity` / successes-Gewichtung / Korruption / `population_sequences` | `culture.py:134` / `:68` / `:80` / `:120` |
| `FIDELITY_BASE=0.72` / `+0.18` trust / `OBSERVE_PROB` / `TEACH_PROB` / `social_learning_step` | `social_learning.py:26` / `:27` / `:24` / `:25` / `:39` |
| `imitate_from` | `agents/brain.py:186` |
| `PRIMITIVE_ACTIONS` (8-enum) | `systems/invention.py:52` |
| `agent_invent_from_need` / WANN-Gate / WAS-softmax | `need_driven_invention.py:318` / `:351` / `:305–310` |
| `tick_growth` (Bottleneck) | `environment/growth.py:113` |
| `DISCOVERY_REGISTRY` (append-only singleton) | `environment/materials.py:325` |
| `combine_vectors` / RNG-Konsum / `get_vector` `uses`-Inkrement / `discovered_by` Export | `materials.py:430` / `:452–455` / `:294–298` / `:278` |
| einziger Inventar-Dekrement (`sharp_stone`) | `agents/agent.py:752` |
| `tick_systems` (Self-Registration) | `systems/registry.py:125` |

## Anhang B — Zitations-Status (zero fabrication)

**VERIFIZIERT:** Resnick et al. 2020 (arXiv:1910.11424); Hughes et al. 2024 (arXiv:2406.04268, coupled-system); Baker et al. 2019/2020 (arXiv:1909.07528, ICLR 2020); POET Wang/Lehman/Clune/Stanley 2019 (arXiv:1901.01753) + Enhanced POET ICML 2020 (arXiv:2003.08536); Taylor 2016 (arXiv:1507.07403 / Artificial Life 22(3):408–423); Lehman & Stanley 2011 (Evol. Comp. 19(2):189–223); MAP-Elites Mouret & Clune 2015 (arXiv:1504.04909); QD-Survey Pugh/Soros/Stanley 2016; Niche Construction Odling-Smee/Laland/Feldman 2003; Henrich 2015/2016; Tomasello 1999; Error-Threshold Eigen 1971; Red Queen Van Valen 1973.

**UNVERIFIZIERT — HOHES FABRIKATIONSRISIKO:** Soros & Stanley 2014 (Chromaria, ALIFE 14) — exakte Formulierung der vier notwendigen Bedingungen NICHT online bestaetigt; nur als PARAPHRASE + `[UNVERIFIZIERT]`, kein woertliches Zitat, keine "Bedingung 4"-Behauptung.

**NICHT GEFUNDEN:** "Cook et al. 2024" (in repo-eigenen gonogo/path-b-Docs als full learned policy-coupling referenziert) — kein passendes Paper lokalisiert; in OPEN-QUESTIONS als Autoren-Klaerung markiert (moeglicherweise falsch erinnert).

**KORRIGIERTE FEHLATTRIBUTION:** "Chli & De Wilde 2004, Niche Construction and the Evolution of Complexity" → realer Autor ist Tim Taylor (ALIFE-IX 2004).
