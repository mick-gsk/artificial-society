# OPEN-QUESTIONS.md — Ungeprüfte Annahmen und Klärungsbedarf

Stand: 2026-06-30. Diese Datei sammelt die *code-/datenmäßig noch nicht abschließend verifizierten* Annahmen, auf denen das Mechanismus-Design (M1–M6) und die Diagnose ruhen, sowie die Zitate, die vor jeder Veröffentlichung verifiziert werden müssen. Jede offene Frage ist als Vier-Tupel formuliert, wo sie ein eigenständiges Risiko für eine Designentscheidung darstellt: (1) Mechanismus/Annahme im Code, (2) compute-gematchte Baseline bzw. Referenz, (3) hack-resistente funktionale Metrik, (4) pre-quantifiziertes Kill-/Auflösungskriterium. Trennung von **BELEGT** (code-verifiziert mit `file:line`) und **VERMUTET** (Hypothese, noch zu prüfen) ist durchgängig markiert.

Keine Garantie-/Unvermeidbarkeits-Sprache. Jede offene Frage ist als potenziell projekt-blockierend einzuordnen, NICHT als gelöst.

---

## 0. Lese-Reihenfolge / Abhängigkeitsketten (zuerst klären)

**BELEGT (aus Red-Team-Log):** Die schärfste übergreifende Erkenntnis ist, dass **M1, M2, M3, M4 und M6 auf der ungerepairten Messung entweder scheitern oder uninterpretierbar werden** — der DV2-Floor (`accumulated_useful_depth` `metrics.py:290`, gefloored bei 2 in allen Pilot-Armen/Seeds) und die Nicht-Berechenbarkeit von DV3 (`transmitted_frontier_advances` `metrics.py:415`, `computable=False` weil `discovered_by=-1`) konfundieren jedes Kill-Kriterium, das auf diesen DVs sitzt.

Daraus folgt die **harte, noch zu bestätigende** Reihenfolge-Annahme:

- **OQ-0 (VERMUTET, blockierend):** M5 (Messreparatur) muss *vor* M1–M4/M6 stehen und seinen Floor-Test (graded-Metrik strikt > 2 unter positiv-synthetischem Input) bestehen, sonst sind alle nachgelagerten Kill-Kriterien durch Floor-/Computability-Artefakte konfundiert. Diese Behauptung ist *plausibel notwendig*, aber selbst noch nicht empirisch gezeigt (siehe OQ-1). Sie *entfernt einen bekannten Blocker*, garantiert aber keine Interpretierbarkeit.

---

## 1. DV2-Floor-Saturation für embodied Arme (höchste Priorität)

**BELEGT:**
- `accumulated_useful_depth` (`metrics.py:290`) vergibt Tiefenkredit nur bei striktem Frontier-Vorsprung > `ADVANCE_MARGIN=0.02` (`metrics.py:47`).
- `TASK_BASIS` (`metrics.py:261`) = 6 fixe Utility-Funktionen, jede ein Produkt geclampter Property-Dims (z. B. `_u_edible_safe=edibility*(1-toxicity)`, `_u_cutting_tool=sharpness*hardness`), Sättigung bei ~1.0 — dies ist die **strukturelle Decke** (CODE-VERIFIZIERT laut Diagnose/Red-Team M4 Angriff 2, M5 Angriff 2).
- Beobachtet: DV2 = 2 (Floor) für den learned arm in OFF *und* ON, in allen Seeds (gonogo-Doc; Diagnose Caveat A).

**VERMUTET (zu klären):**
- Ob der Floor eine Eigenschaft (a) der *Embodiment/Fragmentierung*, (b) der *statischen 6-Task-Decke* (`utility<=1.0`), oder (c) einer echten *Generierungs-Insuffizienz* ist. Diese drei sind mit der aktuellen Metrik **nicht trennbar**.

**OQ-1 — Vier-Tupel (Floor-Beweglichkeit als Vorbedingung für alles):**
1. *Mechanismus/Annahme:* M5 `graded_useful_advance` (tiefenauflösende Erweiterung von `accumulated_useful_depth` `metrics.py:289-343`) kann strukturell Werte > 2 ausgeben.
2. *Baseline/Referenz:* dieselbe Funktion auf bestehenden Pilot-Exports (offline, kein Re-Run), arm-symmetrisch auf learned- *und* Recombiner-Entries angewandt (KEINE per-Agent-Aggregation, die nur einem Arm zusteht — Red-Team M5 Angriff 3).
3. *Metrik:* `graded_useful_advance` unter kontrolliert positiv-synthetischem Tiefen-Input.
4. *Kill:* Gibt die Metrik unter positiv-synthetischem Input strukturell NICHT > 2 aus, ist die Floor-Reparatur **fehlgeschlagen** und M1–M4 bleiben uninterpretierbar (Path B wird wahrscheinlicher).

**Status:** OFFLINE auf existierenden Pilot-Daten validierbar — **kein Re-Run nötig**. Dies ist die billigste und zuerst auszuführende Klärung. *notwendig-nicht-hinreichend:* eine bessere Metrik entfernt den Mess-Blocker, erzeugt aber keine Komplexität und dreht das Vorzeichen der Diagnose nicht.

---

## 2. Auflösung des Embodiment-Confounds (M2)

**BELEGT:**
- `recombiner.py:run_recombiner` ist ein disembodied perfect-memory Monopolist: unconditional append (`recombiner.py:104`), `discovered_by=-1` (`recombiner.py:99`), `uses=0` (`recombiner.py:101`), RNG save/restore (`recombiner.py:54-55,113`).
- Der disembodied Recombiner erreicht laut gonogo-Doc DV2~32 — gegenüber learned floor ~2.

**VERMUTET (zu klären):**
- Ob ein *embodiment-matched* Null (lokale disjunkte Pools, Teilinfo, per-tick-Budget, Transmission mit `FIDELITY_BASE=0.72` `social_learning.py:26`) den learned arm auf der **non-gefloorten** Metrik immer noch erreicht oder übertrifft. Falls ja: das „Lerndefizit" ist ein Embodiment-Artefakt, kein Lernversagen.

**OQ-2 — Vier-Tupel (billigste + härteste Falsifikation, GATED auf OQ-1):**
1. *Mechanismus/Annahme:* `run_recombiner_embodied` (neue Funktion, kein Hot-File-Edit) mit per-Agent `discovered_by>=0` UND record-use-via-Adoption, sodass DV1/DV3 für den Null strukturell ungleich 0 sein KANN (Red-Team M2 Angriff 1).
2. *Baseline:* M2 IST die Kontrolle; Drei-Arm pre-registriert: learned (M1) / embodiment-FREIER Recombiner (bestehend) / embodiment-MATCHED+attribuierter Recombiner (neu). Compute-Match auf real gezählte `combine_vectors`; falls M1 einen ungezählten `_combine_pure`-Twin nutzt, bekommt der Null denselben ungezählten Twin mit UNIFORMER statt policy-gewichteter Auswahl (Red-Team M2 Angriff 6).
3. *Metrik:* PRIMÄR `graded_useful_advance` (NICHT das gefloorte DV2 — Red-Team M2 Angriff 5); SEKUNDÄR DV3 nach M5. Gepaarter Bootstrap-CI (`analyze_gate.py:41`, `N_BOOTSTRAP=10000` `analyze_gate.py:38`).
4. *Kill:* NUR auswertbar nachdem OQ-1 bestanden ist. Erreicht der embodiment-matched Null `graded_useful_advance >= learned` (gepaarter CI learned-(3) NICHT >0, `d_lo<=0`), ist das Defizit ein **Embodiment-Artefakt** → Lernhypothese für diese Metrik widerlegt (Path B).

**Offene Unter-Frage (VERMUTET):** Ist ein attribuierter, konsumierender Null überhaupt ohne Verhaltens-Edit der echten Sim baubar? Red-Team M5 Angriff 4 zeigt code-belegt, dass es **keinen homeostatischen Konsum entdeckter `mat_XXXX`** gibt (einziger echter Dekrement: `sharp_stone` `agent.py:752`) — daher muss record_use an **Adoptions-Events** (`culture.py:80` `receive_transmitted`, `brain.imitate_from` `social_learning.py:88-99`) hängen, nicht an Inventar-Konsum. Ob diese Adoptions-Events für den embodiment-matched Null in ausreichender Zahl feuern, ist UNGEPRÜFT.

*notwendig-nicht-hinreichend:* M2 macht die Lernhypothese falsifizierbar, erzeugt aber keine Komplexität.

---

## 3. Reicht das 1500-Tick-Budget? (Transient-Warnung)

**BELEGT:**
- C+D-Retrofit zeigte +35 % VORSPRUNG bei 250 Ticks, der sich bis 1500 Ticks **umkehrte** (gonogo-Doc) — 250-Tick-Smokes LÜGEN.
- Default-Regime: 4 Seeds × 1500 Ticks (paired seeds 1001–1012 `run_pilot.py:27`), grid 30×20, pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`.
- Kontinuierliche while-Schleife (`simulation.py:493-513`), kein Episode-Reset; `_reset_accumulating_singletons()` einmal in `__init__` (`simulation.py:122`).

**VERMUTET (zu klären):**
- Ob 1500 Ticks ausreichen, um einen *echten* Akkumulationseffekt von einem Vor-Kollaps-Transienten zu trennen — insbesondere für M4 (Koevolution) und M1 (seeded-softmax+VoI), die laut Red-Team das C+D-Reversal-Risiko erben.
- Ob für M4 die Akkumulations**steigung** über 1500 Ticks stabil genug ist, oder ob längere GPU-Läufe (z. B. 8000 Ticks, vgl. `run_pilot.py` Smoke-Hinweis) nötig sind, um das Spät-Fenster (letztes vs. mittleres Drittel) sauber zu vermessen.

**OQ-3 — Vier-Tupel (Tick-Budget-Adäquatheit):**
1. *Mechanismus/Annahme:* 1500 Ticks erlauben es, terminale DV-Differenzen *und* eine Spät-Fenster-Disengagement-Prüfung zu vermessen.
2. *Baseline:* identischer Lauf bei gestaffelten Horizonten (z. B. 1500 vs. 3000 vs. 8000 Ticks) auf denselben Seeds; GPU-PC via SSH, da CPU world-update der Engpass ist (`tick_growth` `growth.py:113`).
3. *Metrik:* Vorzeichen-Stabilität des gepaarten CI der End-Differenz über die Horizonte; Disengagement gemessen über die **lebende** Population (`culture.population_sequences` `culture.py:120`, schrumpft bei Tod), NICHT über die append-only Registry (Red-Team M4 Angriff 4: Registry-Monotonie macht den Check vacuous).
4. *Kill:* Kehrt das Vorzeichen des gepaarten End-Differenz-CI zwischen 1500 und einem längeren Horizont, ist 1500 **unzureichend** und alle M1/M4-Verdikte auf 1500-Tick-Basis sind als potenzielle Transienten zu kennzeichnen.

**Status:** Erfordert einen gestaffelten GPU-Lauf (compute-intensiv → GPU-PC via SSH gemäß stehender Regel). Nicht offline lösbar. *notwendig-nicht-hinreichend:* ein ausreichender Horizont macht die Verdikte interpretierbar, garantiert kein positives Ergebnis.

---

## 4. Zitate, die vor Veröffentlichung verifiziert werden müssen

### 4.1 Soros & Stanley 2014 (Chromaria, ALIFE 14) — HÖCHSTE FABRIKATIONS-GEFAHR

**VERMUTET / UNVERIFIZIERT:** Die exakte Anzahl und Formulierung der „vier notwendigen Bedingungen" für Open-Endedness ist online NICHT bestätigt. Dies ist der stärkste *behauptete* Beleg für C2.

**OQ-4a — Auflösungskriterium:**
- *Aktion:* PDF beschaffen (ALIFE 14 Proceedings).
- *Bis dahin bindend:* nur als **PARAPHRASE + [UNVERIFIZIERT]** zitieren; KEINE wörtlichen Zitate, KEINE Behauptung „Bedingung 4".
- *Kill für die Verwendung:* Lässt sich die Vier-Bedingungen-Liste nicht im Original belegen, darf sie in DIAGNOSE.md/Paper nicht als Stützung von C2 angeführt werden; C2 muss dann allein auf den VERIFIZIERTEN Quellen (Lehman & Stanley 2011, Mouret & Clune 2015, Pugh/Soros/Stanley 2016) ruhen.

### 4.2 „Cook et al. 2024" — Identität ungeklärt

**NICHT GEFUNDEN:** In den repo-eigenen gonogo/Path-B-Dokumenten als Beleg für „full learned policy-coupling" referenziert; kein passendes Paper lokalisiert.

**OQ-4b — Auflösungskriterium:**
- *Aktion:* Autor:in/Originalquelle vom Projekt-Lead klären lassen (möglicherweise falsch erinnert).
- *Bis dahin:* in OPEN-QUESTIONS als ungeklärt führen; NICHT als Beleg verwenden.
- *Kill:* Lässt sich keine reale Quelle identifizieren, ist der Verweis ersatzlos zu streichen; die Policy-Coupling-Begründung (M1) ruht dann allein auf Hughes et al. 2024 (`arXiv:2406.04268`).

### 4.3 Korrigierte Misattribution (BELEGT, zur Kontrolle)

- „Chli & De Wilde 2004, Niche Construction and the Evolution of Complexity" → realer Autor ist **Tim Taylor** (ALIFE-IX 2004). In allen Dokumenten korrigiert führen.

### 4.4 Verifizierte Zitate (BELEGT — nur Liste, keine Neuverifikation nötig)

Resnick et al. 2020 (`arXiv:1910.11424`); Hughes et al. 2024 (`arXiv:2406.04268`, coupled-system framing — sec 2–3 für die konstitutive Definition noch gegenzulesen, siehe OQ-4d); Baker et al. 2019/2020 (`arXiv:1909.07528`); POET Wang/Lehman/Clune/Stanley 2019 (`arXiv:1901.01753`) + Enhanced POET ICML 2020 (`arXiv:2003.08536`); Taylor 2016 (`arXiv:1507.07403`); Lehman & Stanley 2011 (Evol. Comp. 19(2):189-223); MAP-Elites Mouret & Clune 2015 (`arXiv:1504.04909`); QD-Survey Pugh/Soros/Stanley 2016; Niche Construction Odling-Smee/Laland/Feldman 2003; Henrich 2015/2016; Tomasello 1999; Eigen 1971; Van Valen 1973.

**OQ-4d (VERMUTET, klein):** Hughes et al. 2024 sec 2–3 noch gegenlesen, dass die *konstitutive* Definition von Open-Endedness als Eigenschaft des gekoppelten learner+environment-Systems tatsächlich so getroffen wird (Stützung für C5). Bis dahin als Abstract-Paraphrase markieren.

---

## 5. Passen M1–M6 ins 1500-Tick-Budget? (Feasibility pro Mechanismus)

**VERMUTET (zu klären, pro Mechanismus):**

- **M1 (VERMUTET):** Der ungezählte imaginierte `_combine_pure`-Sweep läuft pro Erfindungsentscheidung. Ob K-Kandidaten-Bewertung × Erfindungsfrequenz × 1500 Ticks × 12 Seeds in vertretbarer Wall-Clock-Zeit bleibt, ist UNGEPRÜFT. **Harte technische Vorbedingung (BELEGT, Red-Team M1 Angriff 6):** `combine_vectors` (`materials.py:430`) konsumiert den globalen `random`-Stream (`materials.py:452-455`); der Twin MUSS `random.getstate()`/`setstate()` snapshotten (Muster `recombiner.py:54/113`), sonst zieht die EINE reale `combine_vectors` aus einem anderen Stream-Punkt als OFF → Golden geht RED. Unit-Test: globaler RNG-State byte-identisch vor/nach dem Sweep. Zusätzlich (Red-Team M1 Angriff 4): `get_vector` (`materials.py:294-298`) inkrementiert `uses` bei JEDEM Lookup → der Twin braucht einen nebenwirkungsfreien Getter, sonst pollutet er DV1.

- **M2 (VERMUTET):** K=pop disjunkte Pools + sozialer Austauschkanal pro Tick. Pro-Agent-RNG-Streams aus dem Seed deriviert + deterministische Tick/Agent-Reihenfolge sind PFLICHT (Red-Team M2 Angriff 4), sonst bricht der gepaarte Bootstrap. Wall-Clock UNGEPRÜFT.

- **M3 (VERMUTET):** MAP-Elites-Archiv über Deskriptor `(mean_active_dims metrics.py:156, action-Klasse PRIMITIVE_ACTIONS invention.py:52)`, neues Modul `systems/qd_archive.py` (self-registered `registry.py:125`). Greift NUR in M1s Auswahl-Score ein, KEINE Registry-Mutation (C3-Persistenz intakt). Residual-Risiko (BELEGT, Red-Team M3 Angriff 4): Deskriptor ist statisch + niedrigdimensional → Novelty-Bonus versiegt bei abgedecktem Gitter. Kosten pro Tick UNGEPRÜFT.

- **M4 (VERMUTET, höchstes Feasibility-Risiko):** Pro-Tick-Frontier-Berechnung (Lesen von `DISCOVERY_REGISTRY` + `culture.population_sequences`) MUSS amortisiert/gecacht sein — NICHT ~30k Registry-Entries pro Tick re-clustern (Red-Team M4 Angriff 6). Ob die offene Task-Basis-Erweiterung (`systems/coevolution.py`, neues self-registered Modul) bei 1500 × 12 in vertretbarer Zeit läuft, ist UNGEPRÜFT.

- **M5 (TEILS BELEGT):** Floor-Test + DV2-Tiefenauflösung + `discovered_by`-Durchreichung sind OFFLINE auf bestehenden Exports validierbar (kein Re-Run). NUR der record_use-via-Adoption-Teil erfordert einen Re-Run.

- **M6 (VERMUTET):** FIDELITY_BASE-Sweep über {0.45, 0.60, 0.72, 0.85, 0.95} × M1-ON × 12 Seeds × 1500 Ticks = 5× die Lauf-Last + Kontrollzelle (M1-ON / Fidelity-fest-0.72). Reiner Parameter-Sweep (`social_learning.py:26`, `simulation.py:34`), kein Erfindungs-Budget-Effekt, aber 5–6× Gesamt-Compute. Wall-Clock UNGEPRÜFT.

**OQ-5 — Vier-Tupel (Gesamt-Compute-Budget):**
1. *Mechanismus/Annahme:* Die volle Mechanismus-Matrix (M1–M6 mit allen Kontrollzellen) passt in das verfügbare GPU-PC-Zeitfenster.
2. *Baseline:* Wall-Clock eines einzelnen 1500-Tick-Laufs (gemessen, nicht geschätzt) × Zahl der Zellen × 12 Seeds.
3. *Metrik:* gemessene Sekunden/Tick pro Mechanismus-Zelle (kleiner Mikro-Benchmark vor dem Voll-Lauf).
4. *Kill:* Übersteigt die hochgerechnete Gesamtzeit das Fenster, müssen Mechanismen seriell/priorisiert werden (M5 → M1 → M2 zuerst; M3/M4/M6 nachgelagert).

**Status:** Erfordert einen kurzen Mikro-Benchmark (Sekunden/Tick) je neuer Code-Zelle, BEVOR der Voll-Lauf gestartet wird. Smokes (250 Ticks) zählen für die *wissenschaftliche* Bewertung NICHT, sind aber für die reine *Wall-Clock-Messung* zulässig.

---

## 6. Als VERMUTET geflaggte Code-Anker, die re-verifiziert werden müssen

Folgende Anker werden im Design als BELEGT geführt, sind aber entweder (a) nur aus Diagnose-Text/gonogo-Doc übernommen, (b) zeilengenau noch nicht in der aktuellen `feat/infra-research-stage0a`-HEAD gegengelesen, oder (c) in einem Red-Team-Angriff als kritisch identifiziert. Sie sind VOR der Implementierung zeilengenau zu bestätigen:

| Anker | Datei:Zeile (behauptet) | Warum zu re-verifizieren | Status |
|---|---|---|---|
| `combine_vectors` RNG-Konsum (`rub`/ignite) | `materials.py:452-455` | M1-RNG-Leck (Red-Team M1 Angriff 6) hängt exakt hieran; trägt den gesamten Determinismus-Vertrag | **VERMUTET — kritisch** |
| `get_vector` inkrementiert `uses` | `materials.py:294-298` | M1/M5 uses-Pollution (Red-Team M1 Angriff 4); DV1-Instrumentierung hängt daran | **VERMUTET — kritisch** |
| einziger `mat_XXXX`-Dekrement = `sharp_stone` | `agent.py:752` | M5 Angriff 4 (record_use ins Leere); entscheidet, ob DV1 überhaupt rettbar ist | **VERMUTET — kritisch** |
| `accumulated_useful_depth` Floor-Logik | `metrics.py:289-343` | DV2-Floor-Saturation (OQ-1) hängt an der exakten Margin-/Layer-Logik | **VERMUTET — kritisch** |
| `TASK_BASIS` 6 Ziele + Clamp-Decke | `metrics.py:261-268`, `_u_*` `metrics.py:237,249` | statische Fitness-Decke (M4/M5 Angriff 2); `utility<=1.0` ist load-bearing | **VERMUTET — kritisch** |
| `population_sequences` schrumpft bei Tod | `culture.py:120` | Disengagement-Check (M4 Angriff 4) hängt daran, dass dies NICHT append-only ist | **VERMUTET** |
| `sample_for_transmission` successes-Gewichtung | `culture.py:64-76` | statisches Selektionsziel (M6 Angriff 4) — Geltungsgrenze des Fidelity-Sweeps | **VERMUTET** |
| `receive_transmitted` Korruptions-Mechanik | `culture.py:80` (`if random.random() > fidelity`) | Adoptions-Proxy für record_use (M5) + Fidelity-Sweep (M6) | **VERMUTET** |
| `imitate_from` (Adoptions-Event) | `brain.py:186` (Design zitierte teils `social_learning.py:88-99`) | **Zeilen-Diskrepanz**: `imitate_from` ist in `brain.py:186` verankert, das Design referenziert an anderer Stelle `social_learning.py:88-99` — beide Anker zeilengenau gegenprüfen | **VERMUTET — Anker-Diskrepanz** |
| `need_driven_invention` WANN/WAS-Trennung | Design zitierte `:351`, `:305-310`, `:241-310`, `:318` (`agent_invent_from_need`) uneinheitlich | Zeilennummern im Design inkonsistent; M1 hängt an der exakten Stelle, wo WAS gewählt wird | **VERMUTET — Anker-Diskrepanz** |
| `INHERIT_FIDELITY` / `DEATH_BROADCAST_FIDELITY` | `simulation.py:34`, `:35` (Diagnose-Text sagte teils `:33-36`) | Zeilenbereich uneinheitlich zitiert | **VERMUTET — geringfügig** |
| `_reset_accumulating_singletons` Aufrufstelle | `simulation.py:122` (Diagnose-Text sagte `~73-80`) | **Zeilen-Diskrepanz** zwischen Diagnose-Text (~73-80) und Anker (122) | **VERMUTET — Anker-Diskrepanz** |
| while-Schleife Persistenz | `simulation.py:493-513` (Diagnose-Text sagte `~290-310`) | **Zeilen-Diskrepanz** zwischen Diagnose-Text und Anker | **VERMUTET — Anker-Diskrepanz** |
| `spawn_child_from_parent` Vererbung | `simulation.py:182` (Diagnose-Text sagte `~430-470`) | **Zeilen-Diskrepanz**; C3-Persistenz-Beleg | **VERMUTET — Anker-Diskrepanz** |

**OQ-6 — Auflösungskriterium (eine Aktion):** Ein einziger `grep`/Read-Durchlauf gegen die aktuelle `feat/infra-research-stage0a`-HEAD, der jede obige Zeile bestätigt oder korrigiert, VOR Beginn der M1/M5-Implementierung. Die als **Anker-Diskrepanz** markierten Einträge (mehrere Zeilenangaben im Diagnose-Text vs. Code-Anker-Liste) sind dabei verbindlich auf die HEAD-Realität zu vereinheitlichen; bis dahin gilt die `file:line`-Anker-Liste, nicht der Diagnose-Fließtext.

---

## 7. Querliegende, noch nicht entschiedene Design-Fragen

- **OQ-7a (VERMUTET — Tautologie-Risiko M1):** Der `RATCHET_GAIN`-Term koppelt M1s Generator an die DV2-Scoring-Funktion (Red-Team M1 Angriff 1). Offen: Entweder `RATCHET_GAIN=0` setzen ODER die primäre Metrik auf eine **held-out Task-Basis** legen, disjunkt von der, gegen die der Generator optimiert. Welche der beiden Auflösungen gewählt wird, ist noch zu entscheiden; ohne Entscheidung ist „Metrik durch den Mechanismus selbst hackbar" live.

- **OQ-7b (VERMUTET — Headline-Tauglichkeit M3):** `n_functional_clusters` (`metrics.py:144`) begünstigt den disembodied Monopolisten by construction (Red-Team M3 Angriff 1). Bestätigte Einschränkung: M3 ist NUR ein Within-Arm-A/B-Lever (QD-ON vs. value-VoI-ohne-Archiv), NICHT eine Open-Endedness-Headline gegen einen compute/embodiment-matched Null. Diese Geltungsgrenze MUSS in DIAGNOSE.md stehen.

- **OQ-7c (VERMUTET — M4 misst statisches Objekt):** Koppelt M4 nur den In-Run-Reward, nicht die offline-MESS-Frontier, misst die Metrik ein statisches Objekt, während nur der Reward variiert (Red-Team M4 Angriff 1/7). Offen: Wie die offene Task-Basis-Erweiterung *gleichzeitig* in Reward UND in die Mess-Frontier eingespeist wird, ohne `_base_frontier` (`metrics.py:280-289`, base-material-Decke) zu umgehen — noch nicht spezifiziert.

- **OQ-7d (VERMUTET — Positiv-Kontrolle als Pflicht für jede Null-Interpretation):** Für M1, M4 und M6 gilt: bevor ein *flaches/Null-Resultat* interpretiert wird, muss die jeweilige Metrik (graded für M1/M4; DV3 für M6) unter IRGENDeiner Intervention nachweislich beweglich sein. Ohne diese Positiv-Kontrolle ist ein Null-Resultat uninterpretierbar (verletzt „jede Hypothese braucht eine Messung, die sie widerlegt"). Ob eine solche Positiv-Kontrolle pro Mechanismus existiert, ist noch zu konstruieren.

- **OQ-7e (VERMUTET — DV1-Endgültigkeit):** Bleibt `record_use`-via-Adoption `total_weight` des learned arm über 1500 Ticks ~0 (Hook ins Leere, Red-Team M5 Angriff 4), ist DV1 als nicht-rettbar zu markieren und aus allen Gates zu entfernen. Diese Entscheidung ist datenabhängig und steht noch aus.

---

## 8. Zusammenfassung der nächsten verifizierenden Schritte (priorisiert)

1. **OQ-1 (offline, sofort):** graded-Metrik-Floor-Test auf bestehenden Pilot-Exports — billigster, blockierender Test. *Bis hier kein Re-Run.*
2. **OQ-6 (offline, sofort):** `grep`/Read-Pass gegen HEAD, alle als kritisch/Anker-Diskrepanz markierten `file:line` bestätigen oder korrigieren.
3. **OQ-4a/4b (offline, parallel):** Soros & Stanley 2014 PDF beschaffen; „Cook et al. 2024"-Identität beim Lead klären.
4. **OQ-5 (Mikro-Benchmark, GPU-PC):** Sekunden/Tick je neuer Zelle messen, BEVOR Voll-Läufe starten.
5. **OQ-2 (Re-Run, GPU-PC, GATED auf OQ-1):** embodiment-matched attribuierter Null.
6. **OQ-3 (gestaffelte GPU-Läufe):** Tick-Budget-Adäquatheit (1500 vs. längere Horizonte).

Alle Re-Run-Schritte: paired seeds 1001–1012 (`run_pilot.py:27`), 1500 Ticks (oder gestaffelt), grid 30×20, pop 24, CPU-Arm/`CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`, GPU-PC via SSH. 250-Tick-Smokes zählen für die wissenschaftliche Bewertung NICHT (Transient lügt); für reine Wall-Clock-Messung sind sie zulässig.
