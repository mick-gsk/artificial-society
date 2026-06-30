# METRICS-BASELINE.md — Funktionale Metriken, Null-Kalibrierung und A/B-Reproduzierbarkeit

> Status-Konvention: **[belegt]** = code-verifiziert mit `Datei:Zeile`-Anker. **[vermutet]** = plausible Annahme, im Experiment zu prüfen. Verbindliche Schreibregeln gelten: keine Garantie-/Unvermeidlichkeitssprache; jede Metrik trägt ihre Kill-/Akzeptanz-Bedingung; jede Mechanismus-Kopplung ist als *notwendig-nicht-hinreichend* gelabelt.

---

## 0. Zweck und Geltungsbereich

Dieses Dokument definiert **vor jeder Mechanismus-Bewertung** (M1–M6) die Messobjekte: pro funktionaler Metrik die Definition, den **compute-matched UND embodiment-matched** Zufalls-/Rekombinations-Baseline, das Anti-Hacking-Argument (gebunden an den `-9.4`-Befund des Piloten), die Null-Kalibrierungs-Prozedur, sowie die Ratchet-/Fidelity-Schwellen-Messung und die Diversitäts-Definition. Es schließt mit dem reproduzierbaren A/B-Protokoll (paired seeds 1001–1012, N Replikate, präregistrierter gepaarter Bootstrap, Gate-Regel).

**Kern-Begründung [belegt]:** Die Messung ist bereits als Anti-Hacking-Design angelegt — `accumulated_useful_depth` ist explizit *churn-immun und arm-symmetrisch* dokumentiert (`artificial_society/research/metrics.py:297-303`). Trotzdem zeigte das Red-Team mehrere Mess-Pathologien (DV2-Floor, DV3-non-computability, DV1-Instrumentierungsleere), die ALLE Mechanismen uninterpretierbar machen. **Daher ist M5 (Messreparatur) vorgezogener harter Blocker; M1–M4/M6 sind auf M5 gegated.** Dieses Dokument ist die formale Spezifikation dieses Blockers.

Das Pilot-Ergebnis (Gate `PATH_B_OR_RETROFIT`; learned max functional depth ~22.9 vs. random-recombiner ~32.3; gepaarte Differenz **−9.4 [−11.3, −7.2]**) ist der **Referenzbefund**, gegen den jede Metrik-Reform ihre Anti-Hacking-Eigenschaft beweisen muss: Das Defizit überlebt die Messreform; eine reformierte Metrik, die das Vorzeichen allein durch Aggregationswahl dreht, ist **disqualifiziert**.

---

## 1. Metriken-Katalog

Für jede Metrik: **(D) Definition mit Anker**, **(B) compute- + embodiment-matched Baseline**, **(H) Anti-Hacking-Argument**, **(K) Kill-/Akzeptanz-Kriterium**.

### 1.1 `max_functional_depth` — der bestehende Gate-DV [belegt]

**(D) Definition.** Tiefste funktionale/irreduzible Kette: `max(functional_depths(entries, func_tau).values())` (`analyze_gate.py:_max_functional_depth`, `metrics.py:115` `functional_depths`). `functional_depths` clustert Artefakte innerhalb einer `FUNC_TAU=0.15`-Kugel (`metrics.py:45`) und vererbt die **minimale** Strukturtiefe je Cluster, um redundanten Churn zu kollabieren. `structural_depths` (`metrics.py:85`): `sd(seed)=0`, `sd=1+max(inputs)`, zyklen-gesichert. Aktueller PRIMARY_DV des Gates (`analyze_gate.py:PRIMARY_DV = "max_functional_depth"`).

**(B) Baseline.** Compute-matched: der bestehende `recombiner.py` läuft dieselbe `n_attempts`-Zahl real gezählter `combine_vectors` (`recombiner.py:69` Schleife). Embodiment-matched: **fehlt** beim bestehenden recombiner — er ist disembodied perfect-memory-Monopolist (globaler monoton wachsender Pool `recombiner.py:104`, `discovered_by=-1` `recombiner.py:99`, `uses=0` `recombiner.py:101`). M2 liefert den embodiment-matched Null nach.

**(H) Anti-Hacking — gebunden an `-9.4`.** Diese Metrik ist die **schwächste** gegen die Embodiment-Konfundierung: Der Monopolist sättigt den 12-dim-Property-Raum mit ~10⁵ Attempts und erreicht Tiefe ~32 durch reine Akkumulation, nicht durch funktionale Neuheit. Der `-9.4`-Befund ist genau deshalb **NICHT** als Lerndefizit interpretierbar, solange kein embodiment-matched Null existiert (Diagnose-Befund C, Embodiment-Confound). `max_functional_depth` wird daher **degradiert** zu einer orientierenden Sekundär-Diagnostik; Entscheidungs-DV ist die graded-Metrik (§1.3).

**(K) Kill.** Kein eigenständiges Kill-Kriterium — als reiner Counting-Proxy ist diese Metrik laut Diagnose-Befund 1 ein „counting artifact". Sie wird **nur** als Floor-/Orientierungs-Diagnostik mitgeführt, nicht als Entscheidungs-DV (Korrektur gegenüber dem Pilot-Gate).

---

### 1.2 DV2 `accumulated_useful_depth` — churn-immun, gefloort [belegt]

**(D) Definition.** Pro Task aus `TASK_BASIS` (6 vorab-registrierte Utility-Funktionen, `metrics.py:261-267`) zählt ein Artefakt nur dann Tiefe, wenn es die Task-Frontier auf einer **strikt tieferen** Strukturschicht um `ADVANCE_MARGIN=0.02` (`metrics.py:47`) schlägt, die kein flacheres Artefakt erreichte. Score = `sum(per_task_maxdepth.values())` (`metrics.py:289-343`). Frontier startet aus den 24 Basis-`MATERIALS` (`_base_frontier`, `metrics.py:279`). **Churn-immun** (redundante tiefe Artefakte ohne Frontier-Schlag zählen 0) und **arm-symmetrisch** (reine Funktion von Vektoren+Rezepten+Basis) [belegt: `metrics.py:297-303`].

**(B) Baseline.** Offline auf bestehenden Pilot-Exports berechenbar für BEIDE Arme — kein Re-Run nötig. Compute-matched via `recombiner.py:69`; embodiment-matched via M2.

**(H) Anti-Hacking — gebunden an `-9.4`.** Stärkstes Argument: Volumen-/Churn-Hacking ist by design ausgeschlossen — Masse redundanter Artefakte zählt 0 [belegt]. Der Monopolist erreicht hohe DV2 **nicht** durch Hack, sondern durch unbegrenzte Attempts + Monopol-Pool. **Befund (A) der Diagnose:** Das Defizit überlebt diese Reform (learned DV2~2 vs. recombiner ~32) → „Metrik ist hackbar" ist für DV2 **FALSCH**.

**(K) Kill / dokumentierte Pathologie [belegt].** DV2 ist im Piloten am **Floor (=2)** für den learned arm in JEDEM arm/seed gepinnt. Ursache **[vermutet]:** Jede Task-Utility ist ein Produkt geclampter Property-Dims (`_u_edible_safe=edibility*(1-toxicity)` `metrics.py:237`; `_u_cutting_tool=sharpness*hardness` `metrics.py:249`) → Sättigung bei ~1.0; sind 6 Tasks 1× getroffen, deckelt der Score. **Konsequenz:** Ein gefloortes Null-Resultat **widerlegt nichts** (verletzt „jede Hypothese braucht eine widerlegende Messung"). DV2 wird daher **nur als Floor-Diagnostik** mitgeführt; Entscheidungen laufen über die graded-Metrik (§1.3). Zusätzlich [belegt]: `useful_depth_max` ist über die append-only `DISCOVERY_REGISTRY` (`materials.py:325`) **monoton nicht-fallend** — als Disengagement-Wächter daher *vacuous* (siehe §3.2).

---

### 1.3 `graded_useful_advance` — die non-gefloorte, tiefenauflösende Reparatur (M5a) [vermutet, zu implementieren]

**(D) Definition.** Erweiterung von `accumulated_useful_depth` (`metrics.py:289-343`): statt pro Task NUR die maximale advancing-Tiefe zu zählen, wird **die margin-normierte Summe der Frontier-Vorsprünge über ALLE Strukturschichten** akkumuliert. Formal pro Task `t` und Schicht `d`: addiere `max(0, U[rows_d, t] - (frontier_t + margin)) / margin`, dann fold die Schicht in die Frontier (gleiche Fold-Reihenfolge wie `metrics.py:335-337`). So deckelt der Score **nicht** bei 2, sobald 6 Tasks 1× getroffen sind, sondern löst den kumulierten Vorsprungs-**Betrag** je Schicht auf.

> Arm-Symmetrie ist **bindend**: dieselbe Funktion auf learned-Entries UND recombiner-Entries, **keine** per-Agent-Aggregation (Red-Team-Angriff 3: per-Agent-Mittel ist arm-asymmetrisch, da der Recombiner als 1 „Agent" gilt). Die Aggregationsregel wird VOR dem unblinded Re-Pilot präregistriert (analog `metrics.py`-Konventionen für `func_tau`/`k`/`basis`).

**(B) Baseline.** Compute-matched UND embodiment-matched: M2-Arm (3) (gedrosselter, attribuierter Recombiner) liefert dieselbe Gesamtzahl real gezählter `combine_vectors`. Offline auf bestehenden Pilot-Daten **vorvalidierbar** (Floor-Test, §2.1), der Arm-Vergleich erfordert Re-Run unter Standard-Regime (§4).

**(H) Anti-Hacking — gebunden an `-9.4`.** Die graded-Metrik erbt die Churn-Immunität von DV2 (nur Frontier-Schläge zählen). Sie löst die **Floor-Mehrdeutigkeit**: ein learned-vs-Null-Gleichstand kann jetzt zwischen „Lerndefizit" und „Metrik blind" unterscheiden (vorausgesetzt der Floor-Test §2.1 ist bestanden). **Tautologie-Wächter (Red-Team-Angriff 1):** Wird ein Generator (M1) gegen eine Frontier-Größe optimiert, MUSS die primäre Bewertung gegen eine **HELD-OUT Task-Basis** laufen, disjunkt vom Generator-Ziel (oder `RATCHET_GAIN=0`), sonst ist ein DV-Gewinn teil-tautologisch.

**(K) Akzeptanz + Kill.** **Akzeptanz (M5a):** `graded_useful_advance` gibt unter kontrolliert positiv-synthetischem Tiefen-Input einen Wert **strikt > Floor=2** aus UND ist arm-symmetrisch. **Kill:** Kann die Metrik unter positiv-synthetischem Input strukturell **nicht** > 2 ausgeben, ist sie kein Fortschritt gegenüber DV2 → Floor-Reparatur **FEHLGESCHLAGEN**, M1–M4 bleiben uninterpretierbar.

---

### 1.4 DV1 `population_functional_value` — provisional, instrumentierungs-abhängig [belegt]

**(D) Definition.** Σ über funktionale Cluster von (Adoptions-Gewicht · beste Task-Utility) (`metrics.py:346-412`, DV1-Docstring `metrics.py:352`). `provisional=True`, weil das Adoptions-Gewicht aus `uses` stammt — und `uses` ist **polluted**: `get_vector` (`materials.py:294-298`) inkrementiert `uses` bei JEDEM Lookup, nicht nur bei echter Adoption. Für den Recombiner strukturell 0 (`uses=0` `recombiner.py:101`) → der „fairness proof"-Befund (Diagnose-Befund 1): DV1=0 beim Null aus **Instrumentierungs-**, nicht Verhaltensgrund.

**(B) Baseline.** Compute-matched via `recombiner.py:69`; embodiment-matched + **attribuiert + konsumierend** via M2 (sonst eingebaut-asymmetrisch).

**(H) Anti-Hacking — gebunden an `-9.4`.** Solange DV1 aus `uses`-Lookups gespeist wird, ist jeder learned-vs-Null-Kontrast eine **Instrumentierungs-Asymmetrie**, kein Signal — derselbe Fehlertyp wie der `-9.4`-Confound, nur auf DV1 verschoben. **Reparatur (M5b):** `record_use` feuert auf **Adoptions-Events** (`culture.receive_transmitted` `culture.py:80`; `brain.imitate_from` `brain.py:186`), NICHT auf Inventar-Konsum.

> **Red-Team-Angriff 4 [code-verifiziert]:** Die EINZIGE echte Inventar-Dekrement-Stelle ist `sharp_stone` (`agents/agent.py:752`, ein NAMED material) — KEIN `mat_XXXX` wird je homeostatisch verbraucht. Ein `record_use`-Hook auf Konsum feuerte ins Leere → DV1 würde „sauber aber strukturell leer". Daher zwingend Adoptions-Proxy statt Konsum-Proxy, ohne Verhaltens-Edit der Golden-Trajektorie.

**(K) Kill.** **(b)** Bleibt `record_use`-via-Adoption `total_weight` des LEARNED arm über 1500 ticks ~0 (Hook ins Leere), ist DV1 weiter strukturell leer → DV1-Reparatur FEHLGESCHLAGEN, DV1 **endgültig aus allen Gates entfernt**. **(c)** Bleibt der embodiment-matched Null (M2) trotz Adoptions-Events DV1=0, ist die Instrumentierung nicht arm-symmetrisch → M2-DV1-Vergleich verworfen.

---

### 1.5 DV3 `transmitted_frontier_advances` — non-computable bis M5 [belegt]

**(D) Definition.** Frontier-vorrückende Artefakte, die **auch** adoptiert/transmittiert wurden: ein Advance zählt, wenn `discovered_by>=0` (echter Agent) UND `uses>=k` mit `k=2` (`metrics.py:415-447`, Docstring `metrics.py:423`). `computable=False`, wenn KEIN Entry `discovered_by>=0` trägt (`metrics.py:434-440`) — aktueller Export hat alle `discovered_by=-1`.

**(B) Baseline.** Compute-matched via `recombiner.py:69`; embodiment-matched + attribuiert via M2. **Per-Arm-Computability VOR jeder Auswertung asserten** (Red-Team M6-Angriff 3): ein agentenloser Recombiner wäre arm-asymmetrisch non-computable.

**(H) Anti-Hacking — gebunden an `-9.4`.** DV3 ist fidelity-sensitiv by design (`k>=2`-Adoption `metrics.py:445`) und damit der einzige Metrik, die die Transmissions-Achse (M6) auflösen kann. Reparatur (M5c): `discovered_by` ist im Registry vorhanden (`materials.py`-Export), wird aber im JSON gedroppt → Durchreichung im Export macht DV3 computable.

**(K) Akzeptanz + Kill.** **Akzeptanz:** `computable=True` (`metrics.py`) weil `discovered_by>=0`. **Kill (d):** Wird DV3 für KEINEN seed computable, ist der Export-Fix unvollständig.

---

### 1.6 `n_functional_clusters` — Anti-Greedy-Ko-Kriterium [belegt]

**(D) Definition.** Anzahl funktionaler Cluster über `DISCOVERY_REGISTRY.entries` bei `FUNC_TAU=0.15` (`metrics.py:144`); Deskriptor-Hilfe `mean_active_dims` (`metrics.py:156`). Reiner greedy Vektor-Cluster-Count, **ohne** interne Compute-/Embodiment-Normalisierung.

**(B) Baseline.** **NUR Within-Arm** (Red-Team M3-Angriff 1+5): QD-ON vs. value-VoI-OHNE-Archiv, gleicher embodied Learner, identische imaginierte+reale `combine_vectors` → compute- UND embodiment-matched. Der recombiner (M2) ist **NICHT** Headline-Null für diese Metrik: der Monopol-Pool out-clustert jede fragmentierte Population by construction.

**(H) Anti-Hacking — gebunden an `-9.4`.** Genau weil der disembodied Monopolist diese Metrik trivial dominiert (derselbe Mechanismus, der `-9.4` produziert), ist `n_functional_clusters` **kein** Open-Endedness-Headline-DV. Gültig nur als Within-Arm-Anti-Greedy-Co-Kriterium, gekoppelt an die graded-Metrik: gültiger Erfolg verlangt **mehr Cluster OHNE graded-Verlust** (sonst „nutzlose Vielfalt").

**(K) Kill.** Als M1-Ko-Kriterium: selbst bei graded-Anstieg widerlegt, wenn `n_functional_clusters(ON) < n_functional_clusters(OFF)` über gepaarten CI (Greedy-Kollaps). Als M3-Headline: **disqualifiziert** (Red-Team M3-Angriff 1, „does-not-survive als Headline-DV").

---

### 1.7 `cultural_diversity` — die Populations-Diversitätsmetrik [belegt]

**(D) Definition [belegt, Korrektur des Pilot-Missverständnisses].** `cultural_diversity()` = `len(population_sequences)` = Anzahl **distinkter kausaler Sequenzen, die von lebenden Agenten gewusst werden** (`culture.py:134`; Speicher `population_sequences` `culture.py:120`). Trait-Raum = `(action, mat_a, mat_b)`-Tupel, **NICHT** genotypisch/behavioral. Diese Größe **schrumpft bei Tod** (lebende Population).

> **Abgrenzung [belegt]:** Die im Pilot zitierte Zahl `0.127 → 0.029` ist **NICHT** diese Diversitätsmetrik. Sie ist **Invention-Effizienz** = `n_discoveries / executed combine_vectors` (`analyze_learning.py:71`), und die Zahlen sind OFF-arm (0.127) vs. ON/Retrofit-arm (0.029) — d.h. der greedy means-ends-Retrofit machte die **Rate schlechter** (Herding auf wenige high-value combos). Es ist KEINE temporale Kollaps- und KEINE Diversitätsmetrik. Dieses Dokument trennt beide explizit.

**(B) Baseline.** Within-Arm (OFF vs. ON), compute-matched; als Disengagement-Wächter (§3.2) über die **lebende** Population, nicht die unsterbliche Registry.

**(H) Anti-Hacking.** Da über lebende Agenten gemessen, ist sie gegen den append-only-Registry-Trick immun (anders als `useful_depth_max`). Volumen im Registry hebt sie nicht.

**(K) Kill (Disengagement).** Fällt `cultural_diversity` im letzten Drittel des Laufs gegenüber dem mittleren Drittel (gepaart über seeds), hat die Anti-Disengagement-Garde (Hall-of-Fame, M4) ihr Ziel verfehlt.

---

### 1.8 Invention-Effizienz — Diagnostik, nicht Entscheidungs-DV [belegt]

**(D) Definition.** `n_discoveries / executed combine_vectors`, abgeleitet aus `np.diff(cum)/np.diff(grid)` (`analyze_learning.py:71`). Diagnostisch für Herding/Erschöpfung.

**(B/H/K).** **Baseline:** Within-Arm OFF vs. ON, compute-matched. **Anti-Hacking:** Ein blinder Recombiner mit Monopol-Pool hat hohe Roh-Effizienz durch Volumen; daher **nur** Within-Arm-Diagnostik, kein Cross-Arm-DV. **Kill:** keine eigenständige Hypothese; sie ist die **Frühwarnung** für den C+D-Herding-Modus (`0.127→0.029`). Wird sie als Entscheidungs-DV missbraucht, ist das Resultat uninterpretierbar (Volumen-Confound).

---

## 2. Null-Kalibrierungs-Prozedur

### 2.1 Floor-Mobilitäts-Positiv-Kontrolle (PFLICHT vor jeder Null-Interpretation)

[vermutet, zu implementieren als Teil M5] Bevor irgendein Null-Resultat (gepaarter CI nicht > 0) interpretiert wird, muss bewiesen sein, dass die Entscheidungs-Metrik **überhaupt beweglich** ist:

1. **Synthetischer Tiefen-Input:** konstruiere eine künstliche Entry-Liste mit kontrolliert wachsenden, frontier-schlagenden Artefakten über mehrere Schichten.
2. **Assert:** `graded_useful_advance` gibt strikt > 2 aus (sonst Floor-Reparatur fehlgeschlagen, §1.3-K).
3. **Knock-out-Validierung** gegen die bestehende `metrics.py:27-30`-Konvention: entferne den Tiefen-Input → Metrik fällt zurück.
4. **In-Run-Beweglichkeit:** unter IRGENDeiner Intervention muss die graded-Metrik nachweislich > Floor beweglich sein, sonst ist ein Null **uninterpretierbar** (ein gefloortes Null widerlegt nichts).

Ohne bestandene Positiv-Kontrolle ist jedes Kill-Kriterium, das auf einem Null-Resultat beruht, **gesperrt**.

### 2.2 Compute-Match-Integrität (Unit-getestet)

[vermutet, zu implementieren] Für jeden Arm mit imaginiertem Zwilling (M1) und für die M2-Nullen:

- **(1)** Imaginierte `combine_vectors` real gezählt == 0.
- **(2)** ON-Arm reale `combine_vectors` <= OFF-Arm.
- **(3) [Red-Team M1-Angriff 6]** Globaler `random`-State **byte-identisch** vor/nach dem imaginierten Sweep. Begründung [belegt]: `combine_vectors` konsumiert den globalen Stream (`rub`/ignite `materials.py:452-455` `random.random()`/`random.uniform`). Der Zwilling MUSS `random.getstate()` vor und `random.setstate()` nach dem K-Kandidaten-Sweep (Muster `recombiner.py:54/113`), sodass die imaginierte Suche NETTO 0 RNG-Draws verbraucht. Test prüft den State, nicht nur „reale Aufrufe==0".
- **(4) [Red-Team M1-Angriff 4]** Der Zwilling liest Vektoren **ohne** `uses`-Inkrement (nebenwirkungsfreier Getter / gecachte Vektoren), da `get_vector` (`materials.py:294-298`) sonst `uses` für bloß *erwogene* Materialien aufbläht und DV1 korrumpiert.

### 2.3 Pro-Agent-RNG-Determinismus (M2)

[vermutet] Der embodiment-matched Recombiner mit K interleavten Agenten + sozialem Austauschkanal: pro-Agent-RNG-Streams aus dem Seed deriviert, **deterministische** tick/agent-Reihenfolge, unit-getestet (analog dem `cc.n==0`-Test der Phase D). Sonst bricht der gepaarte Bootstrap.

### 2.4 Embodiment-Match-Definition (M2, drei Arme)

Präregistriert, **vor** jeder Lernbewertung:

| Arm | Beschreibung | Anker |
|---|---|---|
| (1) learned | embodied Learner mit M1 (policy-gekoppelte selektive Generierung) | `need_driven_invention.py:318` |
| (2) recombiner-frei | bestehender disembodied Monopolist, orientierender externer Null | `recombiner.py:69` |
| (3) recombiner-matched | K=pop disjunkte lokale Pools, per-tick-Attempt-Budget, Teilinfo, Pool-Austausch nur via sozialen Kanal `FIDELITY_BASE=0.72` (`social_learning.py:26`); **attribuiert** (`discovered_by>=0`) **+ konsumierend** (`record_use`-Events); einziger Unterschied zu (1): fehlende Policy | neue `run_recombiner_embodied` |

**[Red-Team M2-Angriff 6]** Nutzt M1 einen ungezählten `_combine_pure`-Imaginationssuchlauf, bekommt Arm (3) denselben ungezählten Bewertungs-Twin, aber mit **uniformer** statt policy-gewichteter Auswahl → „nur die Auswahlregel unterscheidet sich" bleibt ehrlich.

---

## 3. Ratchet- und Fidelity-Schwellen-Messung

### 3.1 Fidelity-Sweep (M6, herabgestuft, bestätigend) [belegt]

Fidelity ist bereits HOCH und laut Diagnose-Befund (B) **nicht** bindend: `FIDELITY_BASE=0.72` (+`0.18*trust`→0.90, `social_learning.py:26-27`), `INHERIT_FIDELITY=0.70` (`simulation.py:34`), `DEATH_BROADCAST_FIDELITY=0.45` (`simulation.py:35`), Korruption `culture.receive_transmitted` (`culture.py:80` `if random.random() > fidelity`), erfolgs-gewichtetes `sample_for_transmission` (`culture.py:68`).

**Mechanismus:** reiner `FIDELITY_BASE`-Sweep über `{0.45, 0.60, 0.72, 0.85, 0.95}` zur Kartierung der Eigen-Error-Threshold-Kurve (Eigen 1971), gekreuzt mit M1-ON.

**Metrik (primär):** **DV3** `transmitted_frontier_advances` als Funktion von `FIDELITY_BASE` (fidelity-sensitiv by design `metrics.py:445`), **NICHT** das gefloorte DV2 (Red-Team M6-Angriff 2: DV2 ist durch Floor-Sättigung überdeterminiert flach → bestätigt „B" aus dem FALSCHEN Grund). Sekundär die graded-Metrik (§1.3).

**Kontroll-Zelle [Red-Team M6-Angriff 6]:** zusätzlich M1-ON / Fidelity-fest-0.72, damit ein DV3-Anstieg der **Fidelity-Achse** und NICHT M1 zugeschrieben werden kann (crossed-only kann beide nicht trennen).

**Positiv-Kontrolle PFLICHT:** DV3 muss bei IRGENDeinem Sweep-Wert nachweislich auf Fidelity reagieren, bevor ein flaches Resultat interpretiert wird.

**Geltungsgrenze [belegt, Red-Team M6-Angriff 4]:** `sample_for_transmission` gewichtet nach `successes` (`culture.py:68`) = **statische** Selektions-Pressung unabhängig von `FIDELITY_BASE`. Der Sweep testet daher Copy-Noise gegen ein **stationäres** Transmissions-Ziel, nicht die Ratchet-Kopplung (C1 = M4s Aufgabe). Diese Geltungsgrenze MUSS in DIAGNOSE.md dokumentiert werden.

**Kill / Akzeptanz (gated auf M5 für DV3-computability + auf M1-ON-Kontrollzelle):**
- Hypothese „Fidelity NICHT bindend" (Diagnose B) **BESTÄTIGT** (M6 als Lever verworfen), wenn DV3 über `{0.45..0.95}` flach bleibt (kein Wert hebt den gepaarten DV3-Median über 0.72, alle CIs überlappen) UND die Positiv-Kontrolle DV3-Beweglichkeit zeigte.
- **Umgekehrt** (überraschend): hebt ein Nicht-0.72-Wert DV3 über gepaarten `d_lo > 0` UND die M1-Kontrollzelle schließt M1 als Ursache aus, ist (B) teilweise falsch und Fidelity wird aufgewertet.

### 3.2 Ratchet-/Anti-Disengagement-Messung [belegt]

Der Ratchet existiert mechanisch [belegt]: erfolgs-gewichtetes `sample_for_transmission` (`culture.py:64-76`), campfire-Pool + `brain.imitate_from` (`brain.py:186`), Offspring-Vererbung (`spawn_child_from_parent`, `INHERIT_SEQUENCES=10` @ `INHERIT_FIDELITY=0.70`, `simulation.py:33-34,182`). C3-Persistenz ist **PASSED** [belegt]: kontinuierliche while-Schleife (`simulation.py:493-513`), `_reset_accumulating_singletons()` läuft einmal in `__init__` (`simulation.py:122`), globales `DISCOVERY_REGISTRY`-Singleton (`materials.py:325`).

**Disengagement-Wächter — KORREKT instrumentiert [Red-Team M4-Angriff 4, code-verifiziert]:** Messe über die **lebende Population** `cultural_diversity()` / `population_sequences` (`culture.py:120,134`, schrumpft bei Tod), **NICHT** über `useful_depth_max` der append-only-Registry — letzteres ist monoton nicht-fallend by construction (`metrics.py:289-343` über `DISCOVERY_REGISTRY` `materials.py:325`) und damit als Wächter *vacuous*.

---

## 4. A/B-Reproduzierbarkeits-Protokoll

### 4.1 Standard-Compute-Regime [belegt]

| Parameter | Wert | Anker / Quelle |
|---|---|---|
| Paired seeds | 1001–1012 (12) | `run_pilot.py:27` `DEFAULT_SEEDS = list(range(1001, 1013))` |
| Ticks | **1500** (Default) | Standard-Regime; Pilot lief 8000 |
| Grid | 30 × 20 | Standard-Regime |
| Pop | 24 | Standard-Regime |
| Compute | CPU | `CUDA_VISIBLE_DEVICES=-1` |
| Determinismus | `PYTHONHASHSEED=0` | Standard-Regime |
| Ausführung | GPU-PC via SSH | stehende Regel |

**Warnung [belegt]:** 250-tick Smokes **LÜGEN** (Transient). Der C+D-Retrofit lag bei 250 ticks +35% vorne und kehrte sich bis 1500 um. **Alle Kill-Kriterien werten ausschließlich terminale 1500-tick-Endpunkte (bzw. Slope über den vollen Horizont).** Smoke-Resultate zählen NICHT.

### 4.2 N Replikate und Paarung

- **N = 12 gepaarte seeds** pro Arm (paired design `run_pilot.py:46`). Jeder seed liefert exakt ein learned- und ein recombiner-Export (`analyze_gate._collect` mappt `seed -> {learned, recombiner}`).
- Paarung ist **bindend**: der gepaarte Bootstrap operiert auf der seed-weisen Differenz `learned_vals - recomb_vals`, nicht auf unabhängigen Mitteln.
- Für M2 drei Arme: zusätzlich learned-vs-(3) als embodiment-matched Kern-Kontrast.

### 4.3 Präregistrierter gepaarter Bootstrap-Test [belegt]

`analyze_gate._bootstrap_mean_ci` (`analyze_gate.py:41`), `N_BOOTSTRAP=10000` (`analyze_gate.py:38`), Perzentil-CI `[2.5, 97.5]`. Berechnet werden [belegt aus `_verdict`]:

- `l_mean, l_lo, l_hi` = Bootstrap-CI des learned arm (seed=1),
- `r_mean, r_lo, r_hi` = Bootstrap-CI des recombiner arm (seed=2),
- `d_mean, d_lo, d_hi` = Bootstrap-CI der **gepaarten Differenz** `learned - recombiner` (seed=3),
- `separated = l_lo > r_hi`, `paired_positive = d_lo > 0`.

> **Entscheidungs-DV-Korrektur:** Im Pilot war `PRIMARY_DV = "max_functional_depth"` (`analyze_gate.py`). Post-Red-Team ist der Entscheidungs-DV die **graded-Metrik** (§1.3, M5-gated) als **Ko-Primär** mit `n_functional_clusters` (Anti-Greedy). `max_functional_depth`/DV2 laufen nur noch als Floor-Diagnostik mit.

### 4.4 Gate-Regel [belegt]

Aus `analyze_gate._verdict`:

```
separated AND paired_positive   -> PATH_A                (analyze_gate.py:91)
paired_positive (nicht separated) -> BORDERLINE_LEAN_A   (analyze_gate.py:93)
sonst                            -> PATH_B_OR_RETROFIT    (analyze_gate.py:95)
```

**Pilot-Referenz [belegt]:** Verdict war `PATH_B_OR_RETROFIT`; gepaarte Differenz **−9.4 [−11.3, −7.2]** (learned UNTER dem Null). Jede Metrik-/Mechanismus-Reform wird gegen diesen Referenzbefund geprüft: ein Vorzeichenwechsel allein durch Aggregationswahl ist **disqualifiziert** (§1.3-H).

### 4.5 Gating-Reihenfolge (harte Vorbedingungen)

```
M5 (Messreparatur)  — HARTER BLOCKER, zuerst
  ├─ §2.1 Floor-Positiv-Kontrolle bestanden (graded > 2)
  ├─ DV1 record_use-via-Adoption total_weight > 0 (sonst DV1 aus Gates entfernt)
  └─ DV3 computable=True (per-Arm asserted)
        |
        v
M2 (embodiment-matched Null)  — gated auf M5-Floor-Test
M1 (selektive Generierung)    — gated auf M5; Entscheidung auf graded + n_functional_clusters
M3 (QD)                       — gated auf M5; NUR Within-Arm-Lever, kein Headline
M4 (Koevolution)              — gated auf M5; Disengagement über lebende Population
M6 (Fidelity-Sweep)           — gated auf M5 (DV3) + M1-ON-Kontrollzelle
```

Jeder nachgelagerte Mechanismus ist **notwendig-nicht-hinreichend** und auf M5 gegated: ohne reparierte, non-gefloorte, arm-symmetrische Messung sind M1–M4/M6 uninterpretierbar (Floor-Confound, DV3-non-computability, DV1-Leere). M5 entfernt den **Mess**-Blocker und **dreht das Vorzeichen NICHT** — Diagnose-Befund (A) belegt, dass das Defizit die DV2-Reform überlebt.

---

## 5. Zusammenfassung der Metrik-Rollen

| Metrik | Rolle | Entscheidung? | Floor-/Confound-Risiko | Gate |
|---|---|---|---|---|
| `max_functional_depth` | Orientierung | nein | Embodiment-Confound (`-9.4`) | — |
| DV2 `accumulated_useful_depth` | Floor-Diagnostik | nein | gefloort=2 [belegt] | — |
| `graded_useful_advance` (M5a) | **Entscheidung (ko-primär)** | **ja** | repariert Floor (zu validieren) | nach §2.1 |
| DV1 `population_functional_value` | Adoptions-Wert | bedingt | `uses`-Pollution / Leere-Risiko | nach M5b |
| DV3 `transmitted_frontier_advances` | Fidelity-Achse (M6) | bedingt | non-computable bis M5c | nach M5c |
| `n_functional_clusters` | Anti-Greedy (ko-primär M1) | bedingt | Monopolist-Dominanz → nur Within-Arm | Within-Arm |
| `cultural_diversity` | Disengagement-Wächter | nein | lebende Pop (immun gg. Registry-Trick) | Within-Arm |
| Invention-Effizienz | Herding-Frühwarnung | nein | Volumen-Confound → nur Within-Arm | Within-Arm |

**Geltungsgrenze gesamt [vermutet]:** Alle value-basierten DVs erben `TASK_BASIS` (`metrics.py:261-267`) = 6 fixe, exogen vorregistrierte Ziele. Sie können nur „Akkumulation in Richtung dieser 6 Ziele" belegen/widerlegen, nicht offene Akkumulation generell. Diese statische Basis ist für den **Zwischen-Arm-Kontrast** fair (symmetrisch angewandt), aber als Sättigungs-Quelle des DV2-Floors plausibel mitverantwortlich (Diagnose Caveat A) — M4 (offene Basis-Erweiterung) adressiert dies, ist aber selbst M5-gated und FRAGIL.
