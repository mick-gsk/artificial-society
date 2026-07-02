# RISKS-KILL.md — Risiken, Fruehindikatoren und harte Kill-Kriterien

> **Geltungsbereich.** Dieses Dokument operationalisiert pro Design-Entscheidung (M1–M6) jeweils einen **Fruehindikator** (ein billiges, frueh sichtbares Warnsignal) und ein **pre-quantifiziertes hartes Kill-Kriterium** (die Messung, die die Hypothese *widerlegt*). Es schliesst mit den **generischen Fallen** und dem **dokumentierten Red-Team-Log**. Bindende Regeln: keine Garantie-/Unvermeidlichkeitssprache; jedes Kill-Kriterium ist gegen eine **compute- UND embodiment-gematchte** Null kalibriert; jede Entscheidung ist als **notwendig-nicht-hinreichend** etikettiert. Belegtes (code-verifiziert mit `datei:zeile`) ist von Vermutetem getrennt.
>
> **Compute-Regime (verbindlich fuer JEDES Kill-Kriterium).** 12 gepaarte Seeds 1001–1012 (`run_pilot.py:27`), 1500 Ticks, Grid 30×20, Pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`. Gepaarter Bootstrap-CI via `analyze_gate._bootstrap_mean_ci`, `N_BOOTSTRAP=10000` (`analyze_gate.py:38,41`). **250-Tick-Smokes zaehlen NIE** als Entscheidungsgrundlage (siehe Falle F5). Notation: `d_lo` = untere Schranke des gepaarten Bootstrap-CI der Differenz; "schlaegt" = `d_lo > 0`.

---

## 0. Globale Vorbedingung: das Floor-Gate

**Belegt.** Der learned arm liegt in der bestehenden Schritt-A-Metrik DV2 (`accumulated_useful_depth`, `metrics.py:290`) am **Floor = 2** in JEDEM Arm und Seed (gonogo-Doc; Diagnose-Caveat A). Jede Utility in `TASK_BASIS` (`metrics.py:261`) ist ein Produkt geclampter Property-Dims (z. B. `_u_edible_safe = edibility·(1-toxicity)`, `metrics.py:237`) und saettigt bei ~1.0. Eine am Floor klebende Metrik kann weder steigen noch fallen — ein Null-Resultat darauf **widerlegt nichts** und verletzt damit die Regel "jede Hypothese braucht eine widerlegende Messung".

**Konsequenz (hartes Gating, code-belegt durch das Red-Team an M1/M2/M3/M4/M6).** Die Kill-Kriterien von **M1, M2, M3, M4 und M6 sind GATED auf M5**: sie duerfen erst ausgewertet werden, nachdem M5s `graded_useful_advance` den **Floor-Test (strikt > 2 unter positiv-synthetischem Input)** bestanden hat. Wird ein M1–M4/M6-Kill am gefloorten DV2 ausgewertet, ist es durch die bekannte Floor-Saettigung konfundiert und **ungueltig**. M5 ist deshalb vorgezogen und ist ein **harter Blocker**.

---

## M5 — Messreparatur (vorgezogen, harter Blocker)

**Entscheidung.** Non-gefloorte arm-symmetrische funktionale Metrik (`graded_useful_advance`) + adoptions-basierte Instrumentierung (`record_use` ueber Transmission/`imitate_from`, NICHT Inventar-Konsum) + `discovered_by`-Durchreichung im Export.
**Label.** notwendig-nicht-hinreichend; **harter Blocker** fuer M1–M4+M6. *Begruendung:* bessere Messung erzeugt keine Komplexitaet; das Defizit ueberlebt laut Diagnose (A) die DV2-Reform (learned ~2 vs Recombiner ~32) — M5 **dreht das Vorzeichen nicht**.

### M5 — Vier-Tupel
- **(1) Mechanismus (codeverankert).** (a) `graded_useful_advance` = pro Task die **Summe** der margin-normierten Frontier-Vorspruenge ueber **alle** Strukturschichten (Erweiterung von `accumulated_useful_depth` `metrics.py:289–343`), deckelt nicht bei Floor=2. (b) `record_use` auf **Adoptions-Events** (`culture.receive_transmitted` `culture.py:80`; `brain.imitate_from` `brain.py:186`), nicht auf Inventar-Konsum. (c) `discovered_by`-Durchreichung im Export (`materials.py:325`-Registry → JSON).
- **(2) Baseline (VOR dem Mechanismus).** Bestehende provisional/non-computable DVs auf **denselben** Pilot-Exports (DV1 provisional, DV3 `computable=False`, DV2 floored). (a)+(c) sind **offline** auf existierenden Pilot-Daten validierbar (kein Re-Run); (b) erfordert Re-Run im Standard-Regime.
- **(3) Metrik (selbstreferentiell, Metrik IST das Artefakt).** Knock-out-validiert (`metrics.py:27–30): (a) `graded_useful_advance` gibt unter positiv-synthetischem Tiefen-Input einen Wert **strikt > 2** aus UND ist **arm-symmetrisch** (gleiche Funktion auf learned- und Recombiner-Entries, KEINE per-Agent-Aggregation, die nur einem Arm zusteht); (b) DV1-Adoptions-Gewicht (`weight_source='adoption'`) > 0 fuer BEIDE Arme ODER explizit als arm-asymmetrisch deklariert; (c) DV3 `computable=True` (`metrics.py:415`) wegen `discovered_by>=0`.
- **(4) Kill-Kriterium (pre-quantifiziert, vierteilig).**
  - **(a) Floor-Reparatur fehlgeschlagen:** wenn `graded_useful_advance` unter positiv-synthetischem Input strukturell NICHT > 2 ausgeben kann → kein Fortschritt ggue. DV2 → **M1–M4 bleiben uninterpretierbar**.
  - **(b) DV1 nicht rettbar:** wenn `record_use`-`total_weight` des LEARNED arm ueber 1500 Ticks ~0 bleibt (Hook ins Leere) → DV1 wird endgueltig als nicht-rettbar markiert und **aus allen Gates entfernt**.
  - **(c) Instrumentierung arm-asymmetrisch:** wenn der embodiment-matched Null (M2) trotz Adoptions-Events DV1=0 bleibt → M2-DV1-Vergleich **verworfen**.
  - **(d) Export-Fix unvollstaendig:** wenn DV3 fuer KEINEN Seed `computable` wird.

### M5 — Fruehindikator
Schon der **Offline-Smoke-Test** (a)+(c) auf bestehenden Pilot-Exports zeigt binnen Minuten: liefert `graded_useful_advance` auf realen learned-Daten weiterhin 2, ist (a)-Kill nahezu praeget riggert — **stop, bevor irgendein Re-Run finanziert wird**. Zweiter Fruehindikator: `record_use`-Counter im **ersten** Re-Run-Seed; bleibt `total_weight≈0` nach ~300 Ticks, ist (b) auf Kurs.

### M5 — Red-Team-Einwaende (dokumentiert)
- **Angriff 4 (am staerksten, code-belegt, NICHT ueberlebt in Originalform):** einziger echter Inventar-Dekrement ist `sharp_stone` (`agent.py:752`), KEIN `mat_XXXX` wird je homeostatisch verbraucht → `record_use` auf Konsum feuert ins Leere → DV1 "sauber aber strukturell leer". **Korrektur:** `record_use` auf **Adoptions-Events** statt Inventar-Konsum; neues Kill (b).
- **Angriff 3 (Aggregations-Fabrikation):** per-Agent-Mittel macht den Effekt durch Aggregationswahl statt durch Messung; ausserdem arm-asymmetrisch (Recombiner = 1 "Agent"). **Korrektur:** echte tiefenaufloesende, **arm-symmetrische** Metrik; Praeregistrierung der Aggregation.
- **Angriff 1 (DV3-Asymmetrie):** disembodied Recombiner ist `uses=0`/`discovered_by=-1` (`recombiner.py:99,101`) → DV3 trivial 0 → eingebaute Asymmetrie. **Korrektur:** an M2 gekoppelt (attribuierter, konsumierender Null); Kill (c).
- **Angriff 2 (statische Fitness):** `TASK_BASIS` bleibt eine fixe 6-Ziel-Annaeherung → M5 macht die Metrik arm-symmetrischer, **nicht offen** (Teil-Survival, dokumentierte Geltungsgrenze).
- **Verdikt:** *fragil* — survives nur als Instrumentierung in der korrigierten Form; der reine Inventar-Konsum-Hook **faellt**.

---

## M2 — Embodiment-matched Baseline (gedrosselter, attribuierter, konsumierender Null)

**Entscheidung.** `run_recombiner_embodied`: K=Pop disjunkte lokale Pools (Fragmentierung), per-Tick-Attempt-Budget, Teilinformation, Pool-Austausch nur ueber sozialen Kanal mit `FIDELITY_BASE=0.72` (`social_learning.py:26`); per-Agent `discovered_by>=0` + `record_use`-Adoptions-Events. Einziger Unterschied zum learned arm bleibt die **fehlende Policy**.
**Label.** notwendig-nicht-hinreichend. *Begruendung:* Diagnose nennt explizit den EMBODIMENT-CONFOUND; ohne diese Kontrolle ist jeder learned-vs-Recombiner-Vergleich uninterpretierbar. M2 erzeugt keine Komplexitaet (reine Falsifikation).

### M2 — Vier-Tupel
- **(1) Mechanismus (belegt + korrigiert).** Bestehender `run_recombiner` ist disembodied Monopolist: globaler monotoner Pool (`recombiner.py:104`), `discovered_by=-1` (`:99`), `uses=0` (`:101`), Seed = alle 24 MATERIALS, RNG save/restore (`:54-55,113`). Neuer embodied Null beruehrt **keine** Hot-Files; pro-Agent-RNG-Streams aus Seed deriviert, deterministische Tick/Agent-Reihenfolge, unit-getestet.
- **(2) Baseline = M2 selbst (VOR jeder Lernbewertung definiert).** Drei-Arm pre-registriert: (1) learned (mit M1), (2) embodiment-FREIER Recombiner (bestehend), (3) embodiment-MATCHED+attribuierter Recombiner (neu). Compute-Match: Arm (3) faehrt dieselbe Gesamtzahl REAL gezaehlter `combine_vectors` (`materials.py:430`) wie der learned arm; bei M1s ungezaehltem `_combine_pure`-Twin bekommt Arm (3) denselben ungezaehlten Twin, aber **uniforme** statt policy-gewichtete Auswahl ("nur die Auswahlregel unterscheidet sich").
- **(3) Metrik.** PRIMAER: `graded_useful_advance` (M5), **nicht** das gefloorte DV2. SEKUNDAER: DV3 (nach M5 + attribuiertem Null). Gepaarter Bootstrap-CI fuer learned-vs-(2) und learned-vs-(3).
- **(4) Kill-Kriterium (pre-quantifiziert, GATED auf M5).** Nur ausfuehrbar, nachdem M5s graded-Metrik den Floor-Test (>2) bestand. **Dann:** erreicht der embodiment-matched Null (3) `graded_useful_advance >= learned` (gepaarter CI learned−(3) NICHT >0, `d_lo<=0`), ist das "Lerndefizit" ein **EMBODIMENT-ARTEFAKT** → Lernhypothese fuer diese Metrik **widerlegt**, Befund als Konfundierung berichtet (Path B). Echtes Lernsignal nur, wenn learned **> BEIDE** Nullen (2) UND (3) ueber gepaarte CI auf der non-gefloorten Metrik.

### M2 — Fruehindikator
Im laufenden gepaarten Re-Run: konvergiert `graded_useful_advance` von Arm (3) bereits im **mittleren Drittel** auf das learned-Niveau, ist der Embodiment-Artefakt-Befund auf Kurs. Zusatz-Fruehwarnung: bleibt DV3 fuer Arm (3) `computable=False` trotz Attribution, ist M5-Kill (c) gefaehrdet → M2-DV3-Zweig vorab als nicht auswertbar markieren.

### M2 — Red-Team-Einwaende (dokumentiert)
- **Angriff 5 (Killer, NICHT ueberlebt ohne M5):** Gleichstand AM FLOOR belegt KEIN Embodiment-Artefakt, nur fehlende Metrik-Aufloesung. → Kill-Kriterium **MUSS** auf M5 gated sein (jetzt umgesetzt).
- **Angriff 6 (Compute-Match unter M1):** matcht M2 nur gezaehlte Combines, bekommt der Null **null** Suchbudget; gibt M2 ihm einen Twin, ist "uniform" nicht mehr trivial. **Korrektur:** ungezaehlter Twin mit **uniformer** Auswahl, pre-registriert.
- **Angriff 1 (DV1/DV3-Asymmetrie):** Null muss attribuiert+konsumierend sein, sonst eingebauter Vorteil fuer learned. **Korrektur:** per-Agent `discovered_by>=0` + `record_use`.
- **Angriff 4 (RNG-Ordering):** pro-Agent-Streams + deterministische Reihenfolge, unit-getestet (sonst bricht der gepaarte Bootstrap).
- **Angriff 2 (statische Fitness):** symmetrisch fuer den Zwischen-Arm-Kontrast; Geltungsgrenze "nur Akkumulation Richtung der 6 Ziele" dokumentieren.
- **Verdikt:** *fragil* — billigste+haerteste Falsifikation, aber nur valide nach M5.

---

## M1 — Policy-gekoppelte selektive Generierung (seeded-softmax + Value-of-Information)

**Entscheidung.** Ungezaehlter imaginierter Zwilling `_combine_pure` bewertet ein kleines Kandidatenset nach `value(_need_score) + voi_weight·novelty (+ RATCHET_GAIN·marginal)`; Auswahl per **seeded-softmax** (NICHT argmax); dann EINE real gezaehlte `combine_vectors`. Gegated per env-Flag, OFF byte-identisch zur Golden.
**Label.** notwendig-nicht-hinreichend. *Begruendung:* Diagnose (C) identifiziert die fehlende WAS-Kopplung als bindenden Engpass. M1 adressiert nur die Generator-Seite, nicht Transmission/Akkumulation ueber die Population — das Floor-Kill auf der graded-Metrik widerlegt Suffizienz korrekt.

### M1 — Vier-Tupel
- **(1) Mechanismus (belegt + korrigiert).** Policy entscheidet WANN (`need_driven_invention.py:351`), das WAS war quasi-zufaellig (`need_driven_invention.py:305–310`). Korrekturen, code-belegt: (i) **RNG-Snapshot** `random.getstate()`/`setstate()` um den K-Kandidaten-Sweep (Muster `recombiner.py:54/113`), weil `combine_vectors` den globalen Stream konsumiert (`materials.py:452–455`); der imaginierte Sweep verbraucht NETTO **0** RNG-Draws. (ii) **Nebenwirkungsfreier Vektor-Getter** statt `get_vector` (`materials.py:294–298` inkrementiert `uses` bei jedem Lookup → DV1-Pollution). (iii) **Held-out-Task-Basis** disjunkt vom Generator-Ziel ODER `RATCHET_GAIN=0` (Tautologie-Schutz).
- **(2) Baseline (VOR dem Mechanismus).** PRIMAERE Entscheidungs-Baseline = **embodiment-matched INTERNER Null** = derselbe embodied Pfad mit `voi_weight=0` und `RATCHET_GAIN=0` (uniformes Kandidaten-Sampling), gleiche Anzahl imaginierter+realer Aufrufe. Der Recombiner (M2 Arm 2/3) ist nur **orientierender EXTERNER** Null. Compute-Match-Unit-Tests: (1) imaginierte real-gezaehlte Aufrufe ==0; (2) ON-real-Aufrufe <= OFF; (3) **globaler RNG-State byte-identisch vor/nach** dem imaginierten Sweep.
- **(3) Metrik (CO-PRIMAER).** `graded_useful_advance` (M5, held-out) UND als Anti-Greedy-Co-Kriterium `n_functional_clusters` (`metrics.py:144`) **gemeinsam**. Gefloortes Original-DV2 nur noch als Floor-Diagnostik. Gepaarter Bootstrap-CI.
- **(4) Kill-Kriterium (pre-quantifiziert, GATED auf M5).** Nur auswertbar nach M5-Floor-Test (>2). **Widerlegt**, wenn ueber 12 Seeds @1500 der gepaarte CI von `graded(VoI-ON) − graded(VoI-OFF intern)` NICHT >0 (`d_lo<=0`). **Zusatz-Kill (Anti-Greedy):** selbst bei Anstieg widerlegt, wenn `n_functional_clusters(ON) < n_functional_clusters(OFF)` ueber gepaarten CI. **Positiv-Kontrolle PFLICHT:** vor jeder Null-Interpretation muss die graded-Metrik unter IRGENDeiner Intervention nachweislich > Floor beweglich sein.

### M1 — Fruehindikator
Drei Unit-Tests sind das **billigste Fruehwarnsystem** und laufen vor jedem Pilot: schlaegt einer fehl — reale imaginierte Aufrufe >0, ON-real > OFF, oder RNG-State divergiert — ist der Compute-/RNG-Match gebrochen und der Lauf wird **abgebrochen**, da das Ergebnis weder compute- noch RNG-gematcht waere. Inhaltlicher Fruehindikator: `n_functional_clusters(ON)` faellt schon im mittleren Drittel unter OFF → Greedy-Herding kuendigt sich an (vgl. C+D-Kollaps).

### M1 — Red-Team-Einwaende (dokumentiert)
- **Angriff 2 (am schwersten, NICHT ueberlebt ohne M5):** am gefloorten DV2 ist ON−OFF ~0 nicht weil Generierung scheiterte, sondern weil der embodied Arm den Floor nicht verlassen KANN — statisches Mess-Deckenlimit als Null getarnt. **Korrektur:** M5-Gating + Positiv-Kontrolle + non-gefloorte Co-Primaer-Metrik.
- **Angriff 6 (RNG-Leck, code-belegt):** `combine_vectors` konsumiert den globalen Stream; der naive Compute-Match-Test ist blind dafuer. **Korrektur:** Snapshot/Restore + RNG-State-Unit-Test.
- **Angriff 1 (Metrik-Tautologie):** `RATCHET_GAIN` koppelt den Generator an die DV2-Scoring-Funktion. **Korrektur:** held-out-Task-Basis ODER `RATCHET_GAIN=0`.
- **Angriff 4 (uses-Pollution):** imaginierter Sweep inflationiert `uses`. **Korrektur:** nebenwirkungsfreier Getter; Registry bleibt unberuehrt → C3 intakt.
- **Angriff 6b (falsche Primaer-Baseline):** Recombiner als "Primaer" konfundiert Lernen mit Embodiment. **Korrektur:** interner embodiment-matched Null ist die Entscheidungs-Baseline.
- **Angriff 5 (250-Tick-Transient):** gut abgewehrt — Gate nur @1500, Smokes ausgeschlossen.
- **Angriff 3 (notwendig≠hinreichend):** sauber gelabelt; Floor-Kill widerlegt Suffizienz korrekt.
- **Verdikt:** *fragil* — ueberlebt nur mit M5-Gating, RNG-Fix und held-out-Metrik.

---

## M3 — Diversitaets-erhaltende Generierung (QD / MAP-Elites / Novelty) — FRAGIL, nur Within-Arm

**Entscheidung.** MAP-Elites-Archiv ueber Deskriptor=(dominante aktive Property-Dim `metrics.py:156`, Action-Klasse aus `PRIMITIVE_ACTIONS` `invention.py:52`); in M1s imaginierter Suche wird seeded-softmax um Novelty/Archiv-Bonus erweitert (Lehman&Stanley 2011, VERIFIZIERT; Mouret&Clune 2015, VERIFIZIERT). Neues `systems/qd_archive.py`, self-registered (`registry.py:125`), gespeist aus `DISCOVERY_REGISTRY.entries`, greift NUR in M1s Auswahl-Score ein (KEINE Registry-Mutation → C3 intakt).
**Label.** notwendig-nicht-hinreichend, **EXPLIZIT FRAGIL**. *Begruendung:* schuetzt M1 vor dem in C+D dokumentierten Greedy-Kollaps; kann **KEINE Open-Endedness-Headline** liefern.

### M3 — Vier-Tupel
- **(1) Mechanismus (belegt + eingeschraenkt).** RESIDUAL-RISIKO (code-belegt): Deskriptor ist statisch+niedrigdimensional (1 Property-Dim + 8 fixe Actions); echte Ausdruckskraft liegt in **offener** Material-Rekombination (Diagnose Befund 7). Ist das fixe Gitter abgedeckt, geht der Novelty-Bonus auf 0 → QD degeneriert zum value-VoI-Sampler (**endliche** Diversitaets-Pressung).
- **(2) Baseline (VOR dem Mechanismus, WITHIN-ARM).** M1 mit reinem value-VoI-Sampler OHNE Archiv (`novelty_archive_weight=0`), compute-matched auf identische imaginierte+reale `combine_vectors`. Selber embodied Learner → compute- UND embodiment-matched. Der Recombiner (M2) wird hier **NICHT** als Headline-Null benutzt (er out-clustert jede fragmentierte Population by construction).
- **(3) Metrik (Within-Arm, hack-resistent).** `n_functional_clusters(QD-ON)` vs `(value-VoI-OHNE-Archiv)` ueber gepaarten CI, **gekoppelt** an die M5 graded-Metrik. Gueltiger Erfolg verlangt BEIDE: mehr funktionale Cluster OHNE graded-useful-advance-Verlust.
- **(4) Kill-Kriterium (pre-quantifiziert, GATED auf M5).** **Widerlegt**, wenn QD-ON gegen value-VoI-OHNE-Archiv KEINEN gepaarten CI-Anstieg von `n_functional_clusters` >0 zeigt (`d_lo<=0`); ODER wenn der Cluster-Zuwachs mit `graded_useful_advance`-RUECKGANG einhergeht (gepaarter CI <0 → nutzlose Vielfalt); ODER wenn die M5-Metrik trotz QD am Floor bleibt → QD irrelevant fuer den bindenden Engpass.

### M3 — Fruehindikator
`n_functional_clusters` ist eine **monoton kumulative** Registry-Zaehlung — ein frueher Cluster-Bloom kann ein Vor-Kollaps-Transient sein. Fruehwarnung: steigt der Cluster-Count im **ersten Drittel** stark, faellt aber die **per-Tick-Entdeckungsrate** danach, ist der Endwert ein eingefrorener Transient (kein lebendiger Prozess) → Endpunkt skeptisch lesen, graded-Metrik als Schiedsrichter heranziehen.

### M3 — Red-Team-Einwaende (dokumentiert)
- **Angriff 1 (NICHT ueberlebt als Headline):** `n_functional_clusters` ist ein reiner Cluster-Count; der Monopolist-Recombiner dominiert ihn trivial → M3 darf **nicht** als "learned schlaegt die Welt" verkauft werden, nur als Within-Arm-Lever.
- **Angriff 2 (NICHT ueberlebt als durable-Beweis):** monotone Kumulation → ein frueher Lead bleibt eingefroren; DV2 ist am Floor und kann nicht adjudizieren. → Within-Arm-Paarung + graded-Kopplung noetig.
- **Angriff 3 (NICHT ueberlebt ohne M5):** "kein DV2-Verlust" ist am Floor trivial erfuellt → Anti-Churn-Cap **inert** → M3 ist auf M5 contingent.
- **Angriff 4 (statische Fitness im Deskriptor):** fixes Gitter liefert endliche Pressung; weakens, kills nicht die Notwendigkeit.
- **Angriff 5 (ueberlebt):** Within-Arm-A/B ist der EINZIGE sauber embodiment-matched Kontrast.
- **Angriff 6 (ueberlebt):** keine Registry-Mutation → C3-Persistenz intakt.
- **Verdikt:** *fragil* — nur enge Within-Arm-Frage, M5-abhaengig, endliche Pressung.

---

## M4 — Koevolutionaere Nicht-Stationaritaet (C1) — FRAGIL, M5-abhaengig

**Entscheidung.** Nicht-Stationaritaet gekoppelt an eine **OFFENE Erweiterung der Task-Basis** (NICHT blosse Margin-Anhebung): neue Task-Dimensionen erscheinen, wenn die LEBENDE Population die bestehenden saettigt (Komposit-Tasks aus erreichten Frontier-Artefakten, gespeist aus `DISCOVERY_REGISTRY` + `culture.population_sequences` `culture.py:120`). Neues `systems/coevolution.py`, self-registered; Hall-of-Fame der besten je-erreichten Frontier-Artefakte. Reward UND offline-Mess-Frontier werden gemeinsam gekoppelt.
**Label.** notwendig-nicht-hinreichend, **FRAGIL**. *Begruendung:* statische 6-Ziel-Basis saettigt bei utility<=1.0 (plausible DV2-Floor-Ursache in BEIDEN arms); offene Nichtstationaritaet entfernt diesen Saettigungs-Blocker, erzeugt aber ohne erreichbare Trittsteine (M1/M3) nur unerreichbare Ziele. MLS/Gruppenselektion wird NICHT als Haupt-Lever verkauft (C4-Regel).

### M4 — Vier-Tupel
- **(1) Mechanismus (belegt + neukonstruiert).** **Kernkorrektur:** blosse Anhebung von `ADVANCE_MARGIN`/Bedrohung hebt nur die Latte, NICHT das erreichbare Maximum (utility<=1.0 by construction, `metrics.py:237,261`) → loest den Floor strukturell nicht. Daher offene Basis-Erweiterung. **Wichtig:** der Druck muss SOWOHL In-Run-Reward ALS AUCH die offline-Mess-Frontier koppeln, sonst misst die Metrik ein statisches Objekt waehrend nur der Reward variiert.
- **(2) Baseline (VOR dem Mechanismus).** Identisches System mit STATISCHER Basis (Flag OFF, byte-identisch zur Golden). Compute-matched: KEINE zusaetzlichen `combine_vectors`; die per-Tick-Frontier-Berechnung MUSS amortisiert/gecacht sein (NICHT ~30k Registry-Entries pro Tick re-clustern). Regime ist FIXE Tick-Zahl (1500), nicht wall-clock → ON verliert keine Ticks. Embodiment-matched Recombiner (M2) faehrt unter DERSELBEN offenen Basis.
- **(3) Metrik.** `graded_useful_advance` (M5, NICHT gefloortes DV2) ueber die ZEIT (**Akkumulationssteigung**). Anti-Disengagement-Check korrigiert: gemessen ueber die **LEBENDE Population** (`culture.population_sequences` `culture.py:120`, schrumpft bei Tod), NICHT ueber die unsterbliche append-only Registry.
- **(4) Kill-Kriterium (pre-quantifiziert, GATED auf M5).** **Widerlegt**, wenn die graded-Akkumulationssteigung unter offener nicht-stationaerer Basis NICHT groesser ist als unter statischer Basis (gepaarter CI der End-Differenz <=0 ueber 12 Seeds); ODER wenn **Disengagement** auftritt, gemessen ueber die LEBENDE Population (`culture.population_sequences` faellt im letzten Drittel ggue. mittlerem Drittel, gepaart) → Hall-of-Fame verfehlt sein Ziel. Smokes luegen; nur 1500-Tick. Die Steigung+Spaet-Fenster-Instrumentierung faengt explizit den C+D-Reversal-Modus.

### M4 — Fruehindikator
Drei Fruehwarnungen: (1) wall-clock pro Tick explodiert → Frontier-Berechnung ist nicht amortisiert (Feasibility-Stop, sonst 12×1500 nicht finanzierbar); (2) `culture.population_sequences` faellt bereits ab dem mittleren Drittel → Disengagement-Reversal kuendigt sich an; (3) die graded-Steigung ist im ersten Drittel hoch und flacht ab → Vor-Kollaps-Transient (vgl. C+D +35%@250 → Reversal@1500).

### M4 — Red-Team-Einwaende (dokumentiert)
- **Angriff 7 (Killer, fatal coupling gap):** DV2-Frontier wird offline aus dem Export rekonstruiert; koppelt der Druck nur den Reward, misst die Metrik ein statisches Objekt → echter Effekt maskiert ODER false-KILL durch fehlendes Headroom. **Korrektur:** Reward+Mess-Frontier gemeinsam koppeln; M5-Gating.
- **Angriff 1 (Recombiner spuert keine Nicht-Stationaritaet):** ein unbedingt addierender Null hat keinen Reward-Kanal → nur ueber die Mess-Frontier erreichbar. **Korrektur:** Mess-Frontier koppeln.
- **Angriff 2 (Ceiling):** Margin-Anhebung hebt die Latte, nicht das Maximum. **Korrektur:** offene Basis-Erweiterung.
- **Angriff 4 (vacuous Disengagement-Check):** `useful_depth_max` ueber append-only Registry ist monoton nicht-fallend by construction. **Korrektur:** Check ueber lebende `population_sequences`.
- **Angriff 6 (Feasibility):** per-Tick-Frontier ist teuer. **Korrektur:** amortisiert/gecacht; Tick-gematcht (nicht wall-clock).
- **Angriff 5 (ueberlebt):** Steigung + Spaet-Fenster faengt den Reversal.
- **Angriff 3 (ueberlebt als notwendig-nicht-hinreichend):** ohne M1/M3 nur unerreichbare Ziele — korrekt gelabelt.
- **Verdikt:** *fragil* — erst nach M5 interpretierbar; offene-Basis-Erweiterung neu und ungetestet.

---

## M6 — Error-Threshold / Fidelity-Sweep (bestaetigend, herabgestuft) — FRAGIL, M1+M5-gated

**Entscheidung.** `FIDELITY_BASE`-Sweep ueber {0.45, 0.60, 0.72, 0.85, 0.95} (`social_learning.py:26` + `simulation.py:34`), gekreuzt mit M1-ON, um die Eigen-Error-Threshold-Kurve (Eigen 1971) zu kartieren. **KEIN neuer Lever.**
**Label.** notwendig-nicht-hinreichend, **herabgestuft zu bestaetigend, FRAGIL**. *Begruendung:* Diagnose (B) belegt Fidelity ist hoch (`FIDELITY_BASE=0.72`+`0.18·trust→0.90`, `social_learning.py:26–27`; `INHERIT_FIDELITY=0.70` `simulation.py:34`) und nicht bindend. M6 testet nur, dass die Fidelity-Achse kein Sub-Optimum ist.

### M6 — Vier-Tupel
- **(1) Mechanismus (belegt + korrigiert).** Reiner Parameter-Sweep. **Geltungsgrenze (code-belegt):** `sample_for_transmission` gewichtet nach `successes` (`culture.py:64–76`) = STATISCHE Selektions-Pressung unabhaengig von `FIDELITY_BASE` → der Sweep testet Copy-Noise gegen ein STATIONAERES Transmissions-Ziel, nicht die Ratchet-Kopplung (C1 = M4s Aufgabe). Diese Grenze MUSS in DIAGNOSE.md dokumentiert werden.
- **(2) Baseline (VOR dem Mechanismus).** live-Wert `FIDELITY_BASE=0.72` als Within-Arm-Referenz; compute-matched (Sweep aendert kein Erfindungs-Budget). **Korrektur:** zusaetzliche **Kontroll-Zelle M1-ON / Fidelity-fest-0.72**, damit ein graded-Anstieg der FIDELITY-Achse zugeschrieben werden kann und NICHT M1 selbst (crossed-only kann beide nicht trennen).
- **(3) Metrik (PRIMAER auf DV3 verschoben).** DV3 `transmitted_frontier_advances` (`metrics.py:415`, fidelity-sensitiv by design, `k>=2`-Adoption `metrics.py:445`) als Funktion von `FIDELITY_BASE`, PLUS M5 graded-Metrik als Sekundaer. Das gefloorte DV2 ist durch Floor-Saettigung ueberdeterminiert flach (bestaetigt "B" aus dem FALSCHEN Grund) → nicht primaer. **Positiv-Kontrolle PFLICHT:** DV3 muss bei IRGENDeinem Sweep-Wert nachweislich auf Fidelity reagieren, bevor ein flaches Resultat interpretiert wird.
- **(4) Kill-Kriterium (pre-quantifiziert, GATED auf M5 UND M1-Kontrollzelle).** Hypothese "Fidelity NICHT bindend" (Diagnose B) **bestaetigt** (M6 als Lever VERWORFEN), wenn DV3 ueber den Sweep flach bleibt (kein Wert hebt den gepaarten DV3-Median ueber 0.72, alle CIs ueberlappen) UND die Positiv-Kontrolle zeigte, dass DV3 ueberhaupt fidelity-beweglich ist. **Umgekehrt (ueberraschend):** hebt ein Nicht-0.72-Wert DV3 ueber gepaarten `d_lo>0` UND schliesst die M1-Kontrollzelle M1 als Ursache aus, ist (B) teilweise falsch und Fidelity wird aufgewertet. Per-Arm-Computability von DV3 vor jeder Auswertung asserten.

### M6 — Fruehindikator
Vor dem Voll-Sweep: ein **Zwei-Punkt-Vorlauf** {0.45, 0.95} @ wenigen Seeds. Reagiert DV3 zwischen den Extremen ueberhaupt nicht (Positiv-Kontrolle scheitert), ist jedes flache Voll-Sweep-Resultat uninterpretierbar → **stop**, bevor 5×12 Zellen gefahren werden. Zweite Fruehwarnung: bleibt DV3 fuer einen Arm `computable=False`, ist M5 unvollstaendig → M6 nicht ausfuehrbar.

### M6 — Red-Team-Einwaende (dokumentiert)
- **Angriff 2 (NICHT ueberlebt am DV2):** DV2 ist durch Floor ueberdeterminiert flach → Kill feuert aus dem falschen Grund. **Korrektur:** Kill auf **DV3** verschoben + Positiv-Kontrolle.
- **Angriff 3 (NICHT ueberlebt ohne M5):** DV3 ist `computable=False`, solange `discovered_by=-1`. **Korrektur:** harte M5-Vorbedingung + per-Arm-Computability-Assert.
- **Angriff 6 (M1-Confound):** crossed-only kann Fidelity-Achse nicht von M1 trennen. **Korrektur:** M1-ON/0.72-Kontrollzelle.
- **Angriff 4 (statisches Ziel):** `successes`-Gewichtung ist fidelity-unabhaengig → Copy-Noise gegen ein statisches Ziel; Geltungsgrenze dokumentieren.
- **Angriff 1 (kein echter Null):** Within-Arm-Referenz, kein null-kalibrierter Recombiner-Arm (Recombiner ist fidelity-invariant) — akzeptiert als interne Sensitivitaets-Aussage.
- **Angriff 5 (ueberlebt):** 1500-Tick-Regime, Over-Fidelity-Freeze ist die kartierte Hypothese.
- **Verdikt:** *fragil* — ohne M5 nicht ausfuehrbar; nur bestaetigend.

---

## Generische Fallen (gelten quer ueber M1–M6)

### F1 — Disengagement (POET / Hall-of-Fame)
**Risiko (belegt-mechanisch).** Koevolutionaerer/non-stationaerer Druck (M4) kann die Population von unerreichbaren Zielen "abkoppeln" → Kollaps statt Akkumulation.
**Fruehindikator.** `culture.population_sequences` (`culture.py:120`, schrumpft bei Tod) faellt im mittleren Drittel.
**Hartes Kill.** Disengagement-Klausel in M4: `population_sequences` im letzten Drittel < mittleres Drittel (gepaart) → Hall-of-Fame-Garde gescheitert. **Achtung (Red-Team M4 Angriff 4):** der Check darf NICHT ueber die append-only `DISCOVERY_REGISTRY` laufen (monoton nicht-fallend → vacuous), sondern ueber die lebende Population.

### F2 — Diversitaets-Kollaps (Greedy-Herding)
**Risiko (belegt).** Greedy means-ends herdet auf wenige high-value Combos (Diagnose Befund 4; C+D-Effizienz 0.127→0.029, OFF-vs-ON, `analyze_learning.py:71`).
**Fruehindikator.** `n_functional_clusters(ON)` (`metrics.py:144`) faellt unter OFF im mittleren Drittel.
**Hartes Kill.** Anti-Greedy-Co-Kriterium in M1/M3: `n_functional_clusters(ON) < (OFF)` ueber gepaarten CI → widerlegt, selbst bei graded-Anstieg. seeded-softmax statt argmax ist die Praeventiv-Massnahme.

### F3 — Metrik-Hacking
**Risiko (belegt).** (a) Volumen-Saettigung: ein blinder Recombiner erreicht DAG-Tiefe ~22–32 durch Masse; DV2 ist dagegen churn-immun via `ADVANCE_MARGIN=0.02` (`metrics.py:47,290–343`). (b) Generator-Tautologie: ein an `RATCHET_GAIN` gekoppelter Generator optimiert genau die DV2-Scoring-Groesse.
**Fruehindikator.** graded-Gewinn faellt mit der Generator-Zielfunktion zusammen; `n_functional_clusters` steigt ohne graded-Gewinn (reine Vielfalt).
**Hartes Kill.** (a) Gepaarter ON-vs-internem-Null neutralisiert Volumen-Hacks (inflationiert beide Arme gleich). (b) **Held-out-Task-Basis** disjunkt vom Generator-Ziel ODER `RATCHET_GAIN=0` (M1-Korrektur). (c) `n_functional_clusters` darf NIE Headline-DV gegen den Monopolisten sein (M3 Angriff 1).

### F4 — Versteckte statische Fitness
**Risiko (belegt).** `TASK_BASIS` ist ein fixes 6-Ziel-Set (`metrics.py:261`), jede Utility saettigt bei ~1.0 (`metrics.py:237`). Reward UND Metrik beide gegen statische Ziele → "Open-Endedness" reduziert sich auf Annaeherung an 6 feste Vektoren; Frontier wird offline aus statischen `_base_frontier` rekonstruiert.
**Fruehindikator.** graded-Score deckelt bei einer kleinen Zahl, obwohl neue Materialien entstehen.
**Hartes Kill.** M4s offene Basis-Erweiterung MUSS Reward+Mess-Frontier gemeinsam koppeln (Angriff 7); bleibt graded trotzdem gedeckelt → statische-Fitness-Decke bestaetigt, Befund als Geltungsgrenze berichtet. Pre-registrierte Klausel: M5/M4 drehen das Vorzeichen NICHT, falls das Defizit die Reform ueberlebt (Diagnose A).

### F5 — Der 250→1500-Transient-Trap
**Risiko (belegt).** C+D lag bei 250 Ticks +35% vorne und kehrte sich bis 1500 um (gonogo-Doc). Smokes LUEGEN.
**Fruehindikator.** Steigung im ersten Drittel hoch, dann Abflachung/Reversal; bei monoton-kumulativen Metriken (`n_functional_clusters`) friert ein frueher Lead ein, obwohl die per-Tick-Rate kollabiert.
**Hartes Kill / Regel.** Kein Verdikt aus <1500 Ticks. Alle Kill-Kriterien werten **terminale** 1500-Tick-Werte ueber 12 gepaarte Seeds; M4 misst zusaetzlich die **Steigung + Spaet-Fenster** explizit, um den Reversal abzufangen.

### F6 — Determinismus / RNG-Bruch (Invariante, CLAUDE.md "Determinism is sacred")
**Risiko (belegt).** `combine_vectors` konsumiert den globalen `random`-Stream (`materials.py:452–455`); jeder ungezaehlte Imaginations-Sweep (M1/M2-Twin) oder per-Agent-Stream (M2) kann die Golden-Trajektorie verschieben → Headless-Digest- und Golden-Tests RED.
**Fruehindikator.** Golden-/Digest-Test wird RED; oder der RNG-State-Unit-Test (M1) schlaegt fehl.
**Hartes Kill / Regel.** OFF-Pfad byte-identisch zur Golden (env-gated). Pflicht-Unit-Tests: (1) imaginierte real-gezaehlte `combine_vectors`==0; (2) ON-real <= OFF; (3) **globaler RNG-State byte-identisch vor/nach** dem imaginierten Sweep (M1 Angriff 6); pro-Agent-RNG-Streams deterministisch geordnet (M2 Angriff 4). Schlaegt einer fehl → Lauf verworfen, kein wissenschaftliches Verdikt.

### F7 — Instrumentierungs-Asymmetrie (eingebauter Lern-Vorteil)
**Risiko (belegt).** Der disembodied Recombiner ist `uses=0`/`discovered_by=-1` (`recombiner.py:99,101`) → DV1/DV3 fuer ihn strukturell 0; jede positive learned-Zahl "gewinnt" trivial = der "fairness proof"-Fehler, nur auf DV3 verschoben.
**Fruehindikator.** DV1/DV3 fuer den Null bleibt 0, obwohl er Materialien nutzt/teilt.
**Hartes Kill.** M5-Kill (c) + M2-Attribution: bleibt der embodiment-matched Null trotz Adoptions-Events DV1=0 → Vergleich verworfen. Per-Arm-Computability vor jeder DV3-Auswertung asserten (M6 Angriff 3).

---

## Ausfuehrungs-Reihenfolge (aus den Gates abgeleitet)

1. **M5** zuerst (harter Blocker). Floor-Test offline auf Bestandsdaten; faellt (a) → STOP, M1–M4/M6 bleiben uninterpretierbar.
2. **M2** (billigste+haerteste Falsifikation), nur nach bestandenem M5-Floor-Test.
3. **M1** (primaerer Lever), CO-PRIMAER graded + `n_functional_clusters`, gegen internen embodiment-matched Null.
4. **M3** als Within-Arm-Schutz fuer M1; **M4** als Anti-Disengagement/Non-Stationaritaet — beide M5-gated, beide FRAGIL.
5. **M6** zuletzt, bestaetigend, M1+M5-gated.

**Querschnittsregel (Diagnose A).** Ueberlebt das Defizit die Messreform (learned weiterhin << embodiment-matched Null auf der non-gefloorten Metrik), ist der ehrliche Befund **Path B** (Methodik-/Konfundierungs-Paper), NICHT ein "ueberlegenes Lern-System". M5 dreht das Vorzeichen nicht; kein Mechanismus wird als hinreichend verkauft.
