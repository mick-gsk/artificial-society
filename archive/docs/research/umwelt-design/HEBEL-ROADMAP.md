# HEBEL-ROADMAP.md

> **Status:** Arbeitsdokument zur Stage-0a-Forschung (`feat/infra-research-stage0a`). Definiert die sieben Hebel in **diagnose-korrigierter Prioritaet** als testbare Hypothesen mit Vier-Tupel (Mechanismus → Codebase-Mapping; compute-gematchte Null-Baseline; hack-resistente funktionale Metrik; pre-quantifiziertes Kill-Kriterium). Jeder Hebel ist explizit als **notwendig-nicht-hinreichend** gelabelt.
>
> **Operatives Ziel (kein "ueberlegene Spezies"):** Bedingungen identifizieren und testen, unter denen das **gekoppelte** System (Lerner + Umwelt) demonstrierbare offene, kumulative Akkumulation **funktionaler** Komplexitaet zeigt, gemessen gegen eine **compute-gematchte** Zufalls-/Rekombinations-Baseline.
>
> **Bindende Schreibregeln:** Keine Garantie-/Unvermeidbarkeits-Sprache. Trennung von **belegt** (code-verifiziert, mit `file:line`) und **vermutet**. Jeder Mechanismus = Hypothese mit Vier-Tupel und Kill-Kriterium; ohne Kill-Kriterium keine Hypothese.

---

## 0. Diagnose-Kontext (belegt) und Prioritaets-Logik

Die Reihenfolge der Hebel folgt **Diagnose-Relevanz × Aufwand × Compute**, nicht der intuitiven Erwartung. Drei code-verifizierte Befunde bestimmen sie:

- **(A) Mess-Artefakt ist antizipiert und verteidigt.** Das Defizit (`learned` DV2 ≈ 2 vs. Recombiner ≈ 32) **ueberlebt** die Mess-Reform. *Aber*: DV2 sitzt fuer den embodied Arm in **beiden** Armen (OFF und ON) am **Floor = 2** — eine **andere** moegliche Mess-Pathologie (Floor-Saettigung embodied Arme). Offene Frage.
- **(B) Fidelity ist NICHT der bindende Engpass.** `FIDELITY_BASE = 0.72` (`artificial_society/systems/social_learning.py:26`), `INHERIT_FIDELITY = 0.70` (`artificial_society/simulation.py:34`), `DEATH_BROADCAST_FIDELITY = 0.45` (`artificial_society/simulation.py:35`), erfolgs-gewichtete Transmission (`artificial_society/systems/culture.py:68`). Fidelity ist bereits hoch → **herabgestuft** zu bestaetigendem Sweep.
- **(C) DER bindende Engpass:** Generierung ist **nicht selektiv/policy-gekoppelt**. Die PPO-Policy entscheidet das **WANN** der Erfindung, aber nicht das **WAS** der Kombination (`artificial_society/systems/need_driven_invention.py:318`); Erfindung triggert quasi-zufaellig → der `learned` Arm ist ein compute-gedrosselter, fragmentierter Zufalls-Rekombinierer. **Plus Embodiment-Confound:** `recombiner.py` ist ein disembodied Perfect-Memory-Monopolist (`discovered_by=-1` `artificial_society/research/recombiner.py:99`; `uses=0` `:101`; unbedingtes Anhaengen `:104`), compute-gematcht auf `n_attempts`, aber **nicht** embodiment-gematcht.

**Code-verifizierte Vorbedingungs-Kette (aus dem Red-Team):** M5 (Messreparatur) ist ein **harter Blocker** — M1 scheitert am Floor-Confound, M2 wird am Floor uninterpretierbar, M3s Anti-Churn-Cap ist am Floor inert, M4 hat ohne entfloorten Headroom keinen messbaren Spielraum, M6 braucht DV3-Computability. Daher ist M5 **vorgezogen**.

**Compute-Regime (verbindlich, belegt):** 12 gepaarte Seeds `1001–1012` (`artificial_society/research/run_pilot.py:27`), 1500 Ticks, Grid 30×20, Pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`, GPU-PC via SSH. Gepaarter Bootstrap-CI: `N_BOOTSTRAP = 10000` (`artificial_society/research/analyze_gate.py:38`), `_bootstrap_mean_ci` (`:41`). **250-Tick-Smokes luegen** (Pre-Kollaps-Transient: C+D lag bei 250 Ticks +35 % vorne, kehrte sich bis 1500 um). Kein Kill-Urteil aus Smokes.

---

## Prioritaets-Tabelle (diagnose-korrigiert)

| # | Hebel | Adressiert | Status / Fragilitaet | Gate |
|---|-------|-----------|----------------------|------|
| **0 (vorgezogen)** | **M5 Messreparatur** | Befund 1 (DV1-uses-Pollution), Caveat A (DV2-Floor), DV3-Computability | harter Blocker; fragil (3 Teile, 2 umgeschrieben) | — |
| **1** | **M1 Policy-gekoppelte selektive Generierung** | Befund (C) bindender Engpass | fragil | gated auf M5 |
| **2** | **M2 Embodiment-gematchte Baseline** | Embodiment-Confound | fragil | gated auf M5 |
| **3** | **M3 Diversitaets-erhaltende Generierung (QD)** | Befund 4 (greedy collapse) | fragil; nur Within-Arm-Lever | gated auf M5 + M1 |
| **4** | **M4 Koevolutionaere Nicht-Stationaritaet** | C1, DV2-Floor-Hypothese | fragil; offene Basis ungetestet | gated auf M5 (+ M1/M3 fuer Erreichbarkeit) |
| **5** | **M6 Error-Threshold / Evolvabilitaets-Balance (Fidelity-Sweep)** | Befund (B) bestaetigend | herabgestuft, fragil | gated auf M5 (DV3) + M1-Kontrollzelle |
| **6** | **Compute-Budget** | Querschnitt (Tick-Horizont, world-update-Bottleneck) | belegt | — |
| **7** | **Erreichbarkeit / Stepping-Stones** | C2 (Erreichbarkeit ≠ Ausdruckskraft) | sekundaer | nachgelagert |

> **Fidelity (M6) ist herabgestuft** auf einen bestaetigenden Sweep — nicht als Haupthebel verkauft (Diagnose B). **Metriken sind weitgehend erledigt** ausser DV2-Floor + Instrumentierung (= M5).

---

## Hebel 0 (vorgezogen) — M5: Messreparatur (harter Blocker)

**Non-gefloorte, arm-symmetrische funktionale Metrik + adoptions-basierte Instrumentierung + per-Agent `discovered_by` im Null.**

### Mechanismus → Codebase (belegt + red-team-korrigiert)
Drei Teile, zwei umgeschrieben, weil das Red-Team die Original-Form code-belegt widerlegte:

- **(a) Non-gefloorte Metrik statt per-Agent-Mittel.** Die DV2-Familie misst gegen `TASK_BASIS` (`artificial_society/research/metrics.py:261`) = 6 Utility-Funktionen, jede ein Produkt geclampter Property-Dims (`_u_edible_safe = edibility·(1−toxicity)`, `_u_cutting_tool = sharpness·hardness`) → **Saettigung bei ≈ 1.0 (belegt)**. **Korrektur (Red-Team Angriff 3):** per-Agent-Mittel ist **arm-asymmetrisch** (Recombiner = 1 "Agent") → verworfen. Stattdessen eine **echte tiefen-aufloesende** Metrik `graded_useful_advance`: pro Task NICHT binaer (`> frontier + margin`), sondern die **Summe der margin-normierten Frontier-Vorspruenge ueber alle Strukturschichten** (`accumulated_useful_depth` `artificial_society/research/metrics.py:290` erweitert), sodass der Score **nicht** bei Floor = 2 deckelt, sobald 6 Tasks 1× getroffen sind.
- **(b) `record_use` auf Adoptions-Events, NICHT auf Inventar-Konsum.** **Korrektur (Red-Team Angriff 4, belegt):** der einzige echte Inventar-Dekrement betrifft `sharp_stone` (ein NAMED material), **kein** entdecktes `mat_XXXX` wird je homeostatisch verbraucht → ein Konsum-Hook fuerte ins Leere → DV1 waere "sauber, aber strukturell leer". Stattdessen Hook auf **Adoptions-Events**: `culture.receive_transmitted` (`artificial_society/systems/culture.py:80`) + `brain.imitate_from` (`artificial_society/agents/brain.py:186`) als Adoptions-Proxy, **ohne Verhaltens-Edit der Golden-Trajektorie**.
- **(c) `discovered_by`-Durchreichung im Export** (vorhanden in der Registry, im JSON gedroppt) → DV3 (`artificial_society/research/metrics.py:415`) computable.

### Baseline (vor dem Mechanismus definiert)
Bestehende provisional/non-computable DVs auf **denselben Pilot-Exports** (DV1 provisional, DV3 `computable=False`, DV2 floored). Teile (a) und (c) sind **offline auf existierenden Pilot-Daten validierbar** (kein Re-Run noetig). Teil (b) `record_use`-via-Adoption erfordert Re-Run im Standard-Regime (12 Seeds, 1500 Ticks).

### Metrik (hack-resistent — die Metrik IST das Artefakt)
Selbst-referentiell, drei knock-out-validierte Akzeptanzkriterien:
- **(a)** `graded_useful_advance` gibt unter kontrolliert positiv-synthetischem Tiefen-Input einen Wert **strikt > Floor = 2** UND ist **arm-symmetrisch** (gleiche Funktion auf `learned`- und Recombiner-Entries, keine arm-exklusive Aggregation).
- **(b)** DV1-Adoptions-Gewicht (`weight_source='adoption'`) > 0 fuer **beide** Arme — ODER der Befund wird explizit als arm-asymmetrisch deklariert.
- **(c)** DV3 `computable=True`, weil `discovered_by ≥ 0`.

### Kill-Kriterium (pre-quantifiziert, vier Teile)
- **(a)** Wenn `graded_useful_advance` unter positiv-synthetischem Input strukturell **nicht > 2** ausgeben kann → **Floor-Reparatur fehlgeschlagen**, M1–M4 bleiben uninterpretierbar.
- **(b)** Wenn `record_use`-via-Adoption `total_weight` des `learned` Arm ueber 1500 Ticks ≈ 0 bleibt (Hook ins Leere) → **DV1-Reparatur fehlgeschlagen**; DV1 endgueltig als nicht-rettbar markiert und aus allen Gates entfernt.
- **(c)** Wenn der embodiment-gematchte Null (M2) trotz Adoptions-Events DV1 = 0 bleibt → Instrumentierung nicht arm-symmetrisch → M2-DV1-Vergleich verworfen.
- **(d)** Wenn DV3 fuer **keinen** Seed computable wird → Export-Fix unvollstaendig.

**C-Mapping:** Querschnitt (Messhygiene fuer C5-Bewertung); erlaubt erst die saubere Trennung learner-vs-environment-Effekt und behebt die "fairness proof"-Frage (DV1 floored den Recombiner aus **Instrumentierungs-**, nicht Verhaltensgrund). Adressiert direkt das DV2-Floor (Diagnose-Caveat A).

**notwendig-nicht-hinreichend:** **Harter Blocker** — M1 (Floor-Confound), M2 (Floor-uninterpretierbares Kill), M3 (DV2-Cap inert), M4 (kein Headroom), M6 (DV3 non-computable) scheitern **alle** auf der ungerepairten Messung. M5 entfernt den Mess-Blocker und ist deshalb vorgezogen. **Nicht hinreichend:** bessere Messung erzeugt keine Komplexitaet; das Defizit ueberlebt die DV2-Reform (Befund A) → **M5 dreht das Vorzeichen nicht**.

---

## Hebel 1 — M1: Policy-gekoppelte selektive Generierung (bindender Engpass)

**Seeded-Softmax + Value-of-Information (NICHT greedy-argmax, das bereits scheiterte); RNG-isolierter, ungezaehlter imaginierter Zwilling; gemessen auf der M5-Metrik als Co-Primaer.**

### Mechanismus → Codebase (belegt + red-team-korrigiert)
Die PPO-Policy entscheidet das **WANN** (`artificial_society/systems/need_driven_invention.py:318`), das **WAS** ist quasi-zufaellig. **Mechanismus:** ein ungezaehlter imaginierter Zwilling `_combine_pure` (reine Funktion ueber `combine_vectors` `artificial_society/environment/materials.py:430`, **ohne** `DISCOVERY_REGISTRY.register` und ohne `uses`-Inkrement), der ein kleines Kandidatenset nach `value(_need_score) + voi_weight·novelty + RATCHET_GAIN·marginal(Frontier-Vorsprung)` bewertet; Auswahl per **seeded-Softmax** (nicht argmax); dann **eine** real gezaehlte `combine_vectors`+`register`. Gegated per Env-Flag (`AS_PATHA_CD`-Muster), OFF byte-identisch zur Golden.

**Red-Team-Korrekturen (alle code-verifiziert):**
- **(i) RNG-Leck (Angriff 6):** `combine_vectors` konsumiert den globalen `random`-Stream (`rub`/ignite). Der Zwilling **muss** `random.getstate()` vor dem K-Kandidaten-Sweep snapshotten und `random.setstate()` danach (Muster `recombiner.py:54/113`), sodass die imaginierte Suche **netto 0** RNG-Draws verbraucht und die eine reale `combine_vectors` aus demselben Stream-Punkt wie OFF zieht. **Unit-Test: globaler RNG-State byte-identisch vor/nach dem imaginierten Sweep** (nicht nur "reale Aufrufe == 0").
- **(ii) `uses`-Pollution (Angriff 4):** der Zwilling muss Vektoren **ohne** `uses`-Inkrement lesen (`get_vector` inkrementiert `uses` bei jedem Lookup) → nebenwirkungsfreier Getter / gecachte Vektoren.
- **(iii) Metrik-Kopplung / Tautologie (Angriff 1):** der `RATCHET_GAIN`-Term koppelt den Generator an die DV2-Scoring-Funktion. **Korrektur:** die **primaere** Metrik ist eine **held-out** Task-Basis, disjunkt von der, gegen die der Generator optimiert — ODER `RATCHET_GAIN = 0`.

### Baseline (vor dem Mechanismus)
**Primaere Entscheidungs-Baseline (Korrektur Angriff 6b):** der **embodiment-gematchte interne Null** = derselbe embodied Pfad mit `voi_weight = 0` und `RATCHET_GAIN = 0` (uniformes Kandidaten-Sampling), gleiche Anzahl imaginierter + realer Aufrufe. Der Recombiner (M2 Arm 2/3) ist nur orientierender **externer** Null. Compute-Match-Integritaet per Unit-Test: (1) imaginierte `combine_vectors` real gezaehlt == 0; (2) ON-Arm reale Aufrufe ≤ OFF; (3) globaler RNG-State identisch vor/nach dem imaginierten Sweep.

### Metrik (hack-resistent, Co-Primaer)
**Co-Primaer** (Angriff 2: ein gefloortes Null-Resultat widerlegt nichts): M5 `graded_useful_advance` (non-gefloort, tiefen-aufloesend) **UND** als Anti-Greedy-Co-Kriterium `n_functional_clusters` (`artificial_society/research/metrics.py:144`) gemeinsam. Held-out-Task-Basis fuer die graded-Metrik. Das gefloorte Original-DV2 wird nur noch als **Floor-Diagnostik** mitgefuehrt. Gepaarter Bootstrap-CI (`analyze_gate._bootstrap_mean_ci`, `N_BOOTSTRAP = 10000`).

### Kill-Kriterium (pre-quantifiziert, GATED auf M5)
Nur auswertbar, **nachdem** M5s graded-Metrik den Floor-Test (> 2) bestand. Dann **widerlegt**, wenn ueber 12 gepaarte Seeds bei 1500 Ticks der gepaarte CI von `graded_useful_advance(VoI-ON) − graded(VoI-OFF intern)` **nicht > 0** ist (`d_lo ≤ 0`). **Zusatz-Kill (Anti-Greedy):** selbst bei Anstieg widerlegt, wenn `n_functional_clusters(ON) < n_functional_clusters(OFF)` ueber gepaarten CI. **Positiv-Kontrolle Pflicht:** vor jeder Null-Interpretation muss die graded-Metrik unter **irgendeiner** Intervention nachweislich > Floor beweglich sein. **Smoke-Tests (250 Ticks) zaehlen nicht** (C+D war +35 % bei 250, kollabierte bei 1500).

**C-Mapping:** C5 — Generierung ist konstitutiv Teil des gekoppelten learner+environment-Systems (Hughes 2024, **verifiziert**). C2 — VoI/Novelty nutzt die offene Material-Rekombination (`materials.py:430`) als Ausdrucksraum (nicht rohe Kettenlaenge); seeded-Softmax statt argmax respektiert, dass **Erreichbarkeit ueber Stepping-Stones** der Engpass ist, nicht die Raumgroesse.

**notwendig-nicht-hinreichend:** Diagnose (C) identifiziert die fehlende WAS-Kopplung als bindenden Engpass; M1 entfernt diesen Blocker. **Nicht hinreichend:** der plan-getreue Lever (seeded-Softmax+VoI) ist laut gonogo-Doc **ungetestet**; greedy-argmax (getestet) verschlechterte die Effizienz 0.127 → 0.029. M1 adressiert nur die Generator-Seite, **nicht** Transmission/Akkumulation ueber die Population (Angriff 3) → das Floor-Kill auf der graded-Metrik widerlegt Suffizienz korrekt. C3-Persistenz ist unabhaengig PASSED (kontinuierliche While-Schleife `artificial_society/simulation.py:493`; `_reset_accumulating_singletons` einmalig `:122`; Offspring erben Sequenzen `:182`).

---

## Hebel 2 — M2: Embodiment-gematchte Baseline (Confound-Kontrolle)

**Gedrosselter Recombiner unter Lokalitaet / Teilinformation / per-Tick-Limit mit ATTRIBUIERTEM, KONSUMIERENDEM Null; GATED auf M5.**

### Mechanismus → Codebase (belegt + red-team-korrigiert)
`recombiner.py:run_recombiner` ist ein disembodied Perfect-Memory-Monopolist: globaler monoton wachsender Pool (`artificial_society/research/recombiner.py:104`), `discovered_by=-1` (`:99`), `uses=0` (`:101`), RNG save/restore (belegt). **Mechanismus:** `run_recombiner_embodied` (neue Funktion, beruehrt keine Hot-Files): K = Pop disjunkte lokale Pools (Fragmentierung), per-Tick-Attempt-Budget, Teilinformation (nur Materialien in Zelle/Inventar), Pool-Austausch nur ueber sozialen Kanal mit `FIDELITY_BASE = 0.72` (`social_learning.py:26`).

**Red-Team-Korrekturen:**
- **(Angriff 1)** Der Null **muss attribuiert + konsumierend** sein, sonst ist der DV1/DV3-Vergleich eingebaut-asymmetrisch: per-Agent `discovered_by ≥ 0` setzen UND `record_use`-via-Adoption-Events erzeugen (analog M5), sodass DV1/DV3 fuer den Null strukturell ≠ 0 sein **kann**.
- **(Angriff 4)** RNG: pro-Agent-Streams aus dem Seed deriviert, deterministische Tick/Agent-Reihenfolge, unit-getestet.
- Uniform action/material wie das Original (`recombiner.py:71–78`); **einziger** Unterschied zum `learned` Arm bleibt die fehlende Policy.

### Baseline (M2 IST die Baseline)
Drei-Arm pre-registriert: (1) `learned` (mit M1), (2) embodiment-**freier** Recombiner (bestehend), (3) embodiment-**gematchter + attribuierter** Recombiner (neu). Compute-Match: Arm (3) faehrt dieselbe Gesamtzahl **real gezaehlter** `combine_vectors` wie der `learned` Arm. **Korrektur (Angriff 6):** nutzt M1 einen ungezaehlten `_combine_pure`-Imaginationssuchlauf, bekommt Arm (3) denselben ungezaehlten Bewertungs-Twin, **aber mit uniformer statt policy-gewichteter Auswahl** → "nur die Auswahlregel unterscheidet sich" bleibt ehrlich.

### Metrik (hack-resistent)
**Primaer:** `graded_useful_advance` (M5, non-gefloort) je Arm — **nicht** das gefloorte DV2 (Angriff 5: am Floor ist learned-vs-(3) ≈ 0 wegen Mess-Saettigung, nicht wegen Embodiment). **Sekundaer:** DV3 (nach M5 computable + attribuiertem Null). Gepaarter Bootstrap-CI fuer learned-vs-(2) und learned-vs-(3).

### Kill-Kriterium (pre-quantifiziert, billigste + haerteste Falsifikation, GATED auf M5)
Nur ausfuehrbar, **nachdem** M5s graded-Metrik den Floor-Test (> 2) bestand (sonst durch Floor-Saettigung konfundiert, Angriff 5 belegt). Dann: erreicht der embodiment-gematchte Null (3) `graded_useful_advance ≥ learned` (gepaarter CI learned−(3) **nicht > 0**, `d_lo ≤ 0`), ist das "Lerndefizit" ein **Embodiment-Artefakt** → Lernhypothese fuer diese Metrik **widerlegt**, Befund als Konfundierung berichtet (Path B). **Echtes Lernsignal nur, wenn `learned` > BEIDE Nullen (2) UND (3)** ueber gepaarte CI auf der non-gefloorten Metrik.

**C-Mapping:** C5/C1 — trennt die Eigenschaft des gekoppelten Systems (embodied, fragmentiert, nicht-stationaer) vom disembodied Monopolisten; klaert "Umwelt vs. Agent" als falsche Dichotomie (Hughes 2024). Direkt an der Fairness-/Confound-Frage (Diagnose C).

**notwendig-nicht-hinreichend:** Diagnose nennt explizit den Embodiment-Confound. Ohne diese Kontrolle ist jeder learned-vs-recombiner-Vergleich uninterpretierbar. **Nicht hinreichend:** reine Messhygiene/Falsifikation — erzeugt keine Komplexitaet, macht die Lernhypothese falsifizierbar, ohne sie zu garantieren.

---

## Hebel 3 — M3: Diversitaets-erhaltende Generierung (QD/MAP-Elites/Novelty)

**NUR als Within-Arm-A/B-Lever, NICHT als Open-Endedness-Headline. FRAGIL.**

### Mechanismus → Codebase (belegt + red-team-eingeschraenkt)
Das C+D-NO-GO zeigte: greedy means-ends herdet auf wenige high-value combos (Befund 4). **Mechanismus:** MAP-Elites-Archiv (Mouret & Clune 2015, **verifiziert**) ueber Deskriptor = (dominante aktive Property-Dim via `mean_active_dims` `artificial_society/research/metrics.py:156`, action-Klasse aus `PRIMITIVE_ACTIONS` `artificial_society/systems/invention.py:52`). In M1s imaginierter Suche wird der seeded-Softmax um einen Novelty-/Archiv-Bonus erweitert (Lehman & Stanley 2011, **verifiziert**); Elite-Ersetzung pro Zelle nach best Task-Utility. Neues Modul `systems/qd_archive.py` (self-registered via `registry.tick_systems` `artificial_society/systems/registry.py:125`), gespeist aus `DISCOVERY_REGISTRY.entries`; greift **nur** in M1s Auswahl-Score ein (**kein** Hot-File-Edit, **keine** Registry-Mutation → C3-Persistenz intakt, Angriff 6 sauber).

**Residual-Risiko (Angriff 4, belegt):** der Deskriptor ist statisch + niedrigdimensional (1 Property-Dim + 8 fixe Actions); die echte Ausdruckskraft liegt in **offener** Material-Rekombination (Befund 7). Ist das fixe Gitter abgedeckt, geht der Novelty-Bonus auf 0 und QD degeneriert zum value-VoI-Sampler → nur **endliche** Diversitaets-Pressung.

### Baseline (vor dem Mechanismus)
M1 mit reinem value-VoI-Sampler **ohne** Archiv/Novelty-Bonus (`voi_weight > 0`, `novelty_archive_weight = 0`), compute-gematcht auf identische imaginierte + reale `combine_vectors`. Das ist eine **Within-Arm-A/B** (gleicher embodied Learner) → compute- UND embodiment-gematcht (Angriff 5: der **einzige** Kontrast, den M3 sauber beantwortet). Der Recombiner (M2) wird fuer M3 **nicht** als Headline-Null benutzt (Angriff 1: auf `n_functional_clusters` out-clustert der Monopolist jede fragmentierte Population by construction).

### Metrik (hack-resistent, Within-Arm)
**Primaer:** `n_functional_clusters(QD-ON)` vs. `n_functional_clusters(value-VoI-ohne-Archiv)` ueber gepaarten CI, **gekoppelt an die M5 graded-Metrik** (nicht das gefloorte DV2 — Angriff 3 belegt: am Floor ist "kein DV2-Verlust" trivial, Anti-Churn-Cap inert). Gueltiger Erfolg verlangt **beides**: mehr funktionale Cluster **ohne** graded-useful-advance-Verlust.

### Kill-Kriterium (pre-quantifiziert, GATED auf M5)
**Widerlegt**, wenn QD-ON gegen value-VoI-ohne-Archiv **keinen** gepaarten CI-Anstieg von `n_functional_clusters > 0` zeigt (`d_lo ≤ 0`); ODER wenn der Cluster-Zuwachs mit einem `graded_useful_advance`-**Rueckgang** einhergeht (gepaarter CI < 0 → QD erkauft nutzlose Vielfalt); ODER wenn die M5-Metrik **trotz QD am Floor bleibt** → QD irrelevant fuer den bindenden Engpass.

**C-Mapping:** C2 — Stepping-Stones/Erreichbarkeit ueber QD-Deskriptoren; Novelty Search adressiert, dass das **Finden von Trittsteinen**, nicht die Raumgroesse, der Engpass ist. C1 — Diversity-Pressure als Anti-Disengagement-Garde fuer M1/M4. **Einschraenkung (Angriff 4):** ueber die offene Material-Achse liefert das fixe Gitter **keine** dauerhafte Diversitaets-Pressung.

**notwendig-nicht-hinreichend, FRAGIL:** Befund 4 zeigt, greedy Generierung kollabiert ohne Diversitaetsschutz; QD entfernt diesen Kollaps-Blocker **innerhalb** des `learned` Arm. **Nicht hinreichend und explizit fragil:** (1) M3 kann **keine** Open-Endedness-Headline gegen einen compute/embodiment-gematchten Null liefern, weil `n_functional_clusters` den Monopolisten beguenstigt (Angriff 1) → beschraenkt auf die enge Within-Arm-Frage; (2) Anti-Churn-Anspruch nur valide nach M5; (3) Diversitaets-Pressung versiegt bei abgedecktem fixem Gitter.

---

## Hebel 4 — M4: Koevolutionaere Nicht-Stationaritaet (C1)

**Mit OFFENER Task-Basis-Erweiterung (NICHT nur Margin-Anhebung) + korrekt instrumentierter Anti-Disengagement-Garde. FRAGIL, M5-abhaengig.**

### Mechanismus → Codebase (belegt + red-team-neukonstruiert)
`TASK_BASIS` ist ein **statisches** 6-Ziel-Set (`artificial_society/research/metrics.py:261`), jede Utility ein Produkt geclampter Dims → Saettigung bei ≈ 1.0 (belegt, Angriff 2). **Kernkorrektur:** blosse Anhebung von `ADVANCE_MARGIN`/Bedrohung hebt **nur die Latte**, nicht das erreichbare Maximum (`utility ≤ 1.0` by construction) → kann den Floor strukturell **nicht** loesen. Daher koppelt M4 die Nicht-Stationaritaet an eine **offene Erweiterung der Task-Basis selbst**: neue Task-Dimensionen erscheinen, wenn die **lebende** Population die bestehenden saettigt (z. B. Komposit-Tasks aus erreichten Frontier-Artefakten, gespeist aus `DISCOVERY_REGISTRY` + `culture.population_sequences` `artificial_society/systems/culture.py:120`) → populations-gekoppelt (C1; Red Queen Van Valen 1973 **verifiziert**; POET Wang 2019 **verifiziert**). Neues `systems/coevolution.py` (self-registered via `registry.py:125`). Hall-of-Fame: Archiv bester je-erreichter Frontier-Artefakte.

**Wichtig (Angriff 7+1):** der nicht-stationaere Druck muss **sowohl** den In-Run-Reward **als auch** die offline-**Mess**-Frontier koppeln, sonst misst die Metrik ein statisches Objekt, waehrend nur der Reward variiert.

### Baseline (vor dem Mechanismus)
Identisches System mit **statischer** Basis (`coevolution`-Flag OFF, byte-identisch zur Golden). Compute-gematcht: **keine** zusaetzlichen `combine_vectors` (gleiches Erfindungs-Budget); die per-Tick-Frontier-Berechnung **muss amortisiert/gecacht** sein (nicht ~30k Registry-Entries pro Tick re-clustern, Angriff 6 Feasibility) — Regime ist **fixe Tick-Zahl** (1500), nicht wall-clock → ON verliert keine Ticks. Embodiment-gematchter Recombiner (M2) faehrt unter **derselben** offenen Basis.

### Metrik (hack-resistent)
M5 `graded_useful_advance` (nicht gefloortes DV2 — Angriff 7: am Floor maskiert fehlendes Headroom einen echten Effekt als false-KILL) ueber die **Zeit** (Akkumulationssteigung). **Anti-Disengagement-Check korrigiert (Angriff 4, belegt):** `useful_depth_max` ueber die append-only `DISCOVERY_REGISTRY` ist monoton nicht-fallend by construction → **vacuous**. Der Disengagement-Check wird ueber die **lebende** Population gemessen (`culture.population_sequences` `:120`, die bei Tod **schrumpft**), nicht ueber die unsterbliche Registry.

### Kill-Kriterium (pre-quantifiziert, GATED auf M5)
**Widerlegt**, wenn die graded-Akkumulationssteigung unter offener nicht-stationaerer Basis **nicht groesser** ist als unter statischer Basis (gepaarter CI der End-Differenz ≤ 0 ueber 12 Seeds); ODER wenn **Disengagement** auftritt, gemessen ueber die lebende Population (`culture.population_sequences` faellt im letzten Drittel ggue. mittlerem Drittel, gepaart) → Hall-of-Fame hat sein Ziel verfehlt. Smoke-Tests luegen; nur 1500-Tick. Die Steigung + Spaet-Fenster-Instrumentierung faengt explizit den C+D-Reversal-Modus (Angriff 5 sauber).

**C-Mapping:** C1 (Kernachse, korrigiert) — **statisch vs. nicht-stationaer** (an eigene Population gekoppelt), mit Hall-of-Fame + Diversity-Pressure (M3) als Disengagement-Garde. C5 — Koevolution macht Open-Endedness zur Eigenschaft des gekoppelten Systems. C2 — offene Basis-Erweiterung statt blosser Margin-Anhebung respektiert, dass das **attainable maximum**, nicht die Latte, der Engpass ist.

**notwendig-nicht-hinreichend, FRAGIL:** Eine statische 6-Ziel-Basis saettigt bei `utility ≤ 1.0` → kein Druck fuer weitere Tiefe (plausible DV2-Floor-Ursache in **beiden** Armen). Offene Nicht-Stationaritaet entfernt diesen Saettigungs-Blocker. **Nicht hinreichend:** ohne erreichbare Trittsteine (M1/M3) erzeugt Nicht-Stationaritaet nur unerreichbare Ziele (Angriff 3). **Fragil:** (1) die offene-Basis-Erweiterung ist neu und ungetestet; (2) der Test ist erst nach M5 (Headroom) interpretierbar; MLS/Gruppenselektion wird **nicht** als Haupthebel verkauft (C4-Regel: cultural-vs-genetic, group-vs-individual, coordination-vs-IQ nie verwechseln).

---

## Hebel 5 — M6: Error-Threshold / Evolvabilitaets-Balance der Transmissions-Fidelity

**Bestaetigender Sweep, herabgestuft. Kill-Kriterium auf DV3 verschoben; M1/M5-gated mit Positiv-Kontrolle. FRAGIL.**

### Mechanismus → Codebase (belegt + red-team-korrigiert)
Fidelity ist bereits hoch und laut Diagnose (B) **nicht bindend**: `FIDELITY_BASE = 0.72` (+`0.18·trust → 0.90`, `social_learning.py:26–27`), `INHERIT_FIDELITY = 0.70` (`simulation.py:34`), `DEATH_BROADCAST_FIDELITY = 0.45` (`:35`), Korruption (`culture.receive_transmitted` `culture.py:80` `if random.random() > fidelity`), erfolgs-gewichtetes `sample_for_transmission` (`culture.py:68`). **Mechanismus (kein neuer Lever):** `FIDELITY_BASE`-Sweep ueber {0.45, 0.60, 0.72, 0.85, 0.95}, um die Eigen-Error-Threshold-Kurve (Eigen 1971 **verifiziert**) zu kartieren. Reiner Parameter-Sweep an `social_learning.py:26` + `simulation.py:34`, gekreuzt mit M1-ON.

**Einschraenkung (Angriff 4, belegt):** `sample_for_transmission` gewichtet nach `successes` (`culture.py:68`) = **statische** Selektions-Pressung unabhaengig von `FIDELITY_BASE` → der Sweep testet Copy-Noise gegen ein **stationaeres** Transmissions-Ziel, nicht die Ratchet-Kopplung (C1 = M4s Aufgabe). Diese Geltungsgrenze **muss** in `DIAGNOSE.md` dokumentiert werden.

### Baseline (vor dem Mechanismus)
Live-Wert `FIDELITY_BASE = 0.72` (`social_learning.py:26`) als Within-Arm-Referenz; compute-gematcht (Sweep aendert kein Erfindungs-Budget). **Korrektur (Angriff 6):** zusaetzliche **Kontroll-Zelle** M1-ON / Fidelity-fest-0.72, damit ein graded-Anstieg der **Fidelity-Achse** zugeschrieben werden kann und **nicht** M1 selbst (crossed-only kann beide nicht trennen).

### Metrik (hack-resistent)
**Primaer auf DV3 verschoben** (Angriff 2 belegt: das gefloorte DV2 ist durch Floor-Saettigung ueberdeterminiert flach → bestaetigt B aus dem **falschen** Grund): DV3 `transmitted_frontier_advances` (`artificial_society/research/metrics.py:415`, fidelity-sensitiv by design, `k ≥ 2` Adoption) als Funktion von `FIDELITY_BASE`, **plus** die M5 graded-Metrik als Sekundaer. **Positiv-Kontrolle Pflicht:** DV3 muss bei **irgendeinem** Sweep-Wert nachweislich auf Fidelity reagieren, bevor ein flaches Resultat interpretiert wird.

### Kill-Kriterium (pre-quantifiziert, GATED auf M5 + M1-Kontrollzelle)
Hypothese "Fidelity **nicht** bindend" (Diagnose B) **bestaetigt** (M6 als Lever verworfen), wenn DV3 ueber den Sweep {0.45..0.95} flach bleibt (kein Wert hebt den gepaarten DV3-Median ueber 0.72, alle CIs ueberlappen) **UND** die Positiv-Kontrolle zeigte, dass DV3 ueberhaupt fidelity-beweglich ist. **Umgekehrt (ueberraschend):** hebt ein Nicht-0.72-Wert DV3 ueber gepaarten `d_lo > 0` **UND** schliesst die M1-Kontrollzelle M1 als Ursache aus → (B) teilweise falsch, Fidelity wird aufgewertet. Relevanter Effekt erst ab gepaartem `d_lo > 0`. **Per-Arm-Computability von DV3 vor jeder Auswertung asserten** (Angriff 3: Recombiner ohne Agenten waere arm-asymmetrisch).

**C-Mapping:** C4 — kumulative Kultur via hochfideler Transmission (unbestrittener Kern) mit Eigen-Error-Threshold-Balance; ohne die drei Paare zu verwechseln. MLS **nicht** als Haupthebel.

**notwendig-nicht-hinreichend, HERABGESTUFT zu bestaetigend, FRAGIL:** Diagnose (B) belegt, Fidelity ist hoch und nicht bindend. M6 testet nur, dass die Fidelity-Achse kein Sub-Optimum ist; entfernt keinen primaeren Blocker. **Fragil:** (1) ohne M5 (DV3 computable + non-gefloortes DV2) **nicht ausfuehrbar**; (2) die statische Selektions-Pressung bedeutet, M6 misst Copy-Noise gegen ein statisches Ziel (Geltungsgrenze). Hinreichendkeit nicht beansprucht.

---

## Hebel 6 — Compute-Budget (Querschnitt)

### Mechanismus → Codebase (belegt)
Zwei verbindliche Compute-Tatsachen rahmen alle obigen Tests:

- **Tick-Horizont ist load-bearing.** 250-Tick-Smokes zeigen Pre-Kollaps-Transienten (C+D +35 % bei 250, Reversal bis 1500). **Alle Kill-Urteile** erfordern 1500-Tick-Endpunkte ueber 12 gepaarte Seeds.
- **world-update ist der Bottleneck.** `tick_growth` (`artificial_society/environment/growth.py:113`) ist eine pure-Python `O(Zellen × Materialien × Dispersal)`-Doppelschleife; CPU ist ~7–11× schneller als GPU (sparse/unvektorisiert). Konsequenz: **compute-intensive Laeufe auf der GPU-PC via SSH, CPU-Modus** (`CUDA_VISIBLE_DEVICES=-1`), nicht GPU.

### Baseline / Metrik / Kill
Kein eigener Lever, sondern Rahmen-Constraint. **Compute-Match** ist die Achse `n_attempts` = real gezaehlte `combine_vectors` (`artificial_society/research/recombiner.py:69`). **Kill-relevant:** wird ein Effekt nur unter erhoehtem Compute-Budget (mehr Ticks/Seeds) sichtbar, ist das ein Skalierungs-Befund, kein Mechanismus-Befund — explizit als solcher zu berichten. Bei M4 zusaetzlich Feasibility-Pflicht: per-Tick-Frontier amortisiert/gecacht, sonst ist 1500 × 12 Seeds nicht durchfuehrbar.

**notwendig-nicht-hinreichend:** Hinreichend Tick-Horizont ist **notwendig**, um Transienten von durablem Fortschritt zu trennen; er erzeugt selbst keine Komplexitaet. Mehr Raum/Compute ist laut Diagnose **sekundaer** — nicht der bindende Engpass.

---

## Hebel 7 — Erreichbarkeit / Stepping-Stones (C2, sekundaer)

### Mechanismus → Codebase (belegt + vermutet)
**Belegt:** Der Action-Raum ist gemischt — primitive Actions **fix** (8-Enum `PRIMITIVE_ACTIONS` `artificial_society/systems/invention.py:52`), Strukturen fix (5), aber der **Material**-Raum ist **offen/rekursiv** (`mat_A + mat_B → mat_C` via `combine_vectors` `artificial_society/environment/materials.py:430`; adjacent-possible). Die Ausdruckskraft lebt in der Material-Rekombination — exakt, was der blinde Recombiner ausbeutet. **C2-Kern (korrigiert):** Ausdruckskraft ≠ Erreichbarkeit; mehr Raum ist **nicht** der Engpass (sekundaer); diskrete/Bottleneck-Kanaele **treiben** Kompositionalitaet (Resnick 2020 **verifiziert**) und sind ein **Feature**.

**Vermutet:** Der bindende Erreichbarkeits-Aspekt ist das **Finden von Trittsteinen**, nicht die Raumgroesse — daher ist dieser Hebel den generierungs-koppelnden Hebeln (M1 seeded-Softmax, M3 Novelty) **nachgelagert** und wird primaer **durch** sie adressiert, nicht eigenstaendig.

### Baseline / Metrik / Kill
Kein eigenstaendiger Re-Run vor M1/M3. **Metrik:** Anteil der via M1/M3 imaginierten Kandidaten, die einen **erreichbaren** Frontier-Vorsprung (graded-Metrik) liefern, gegen die Zahl der **unerreichbaren** (Score 0 trotz Novelty). **Kill (pre-quantifiziert):** bleibt unter M1+M3 der Anteil erreichbarer Frontier-Advances ueber 12 Seeds bei 1500 Ticks **statistisch ununterscheidbar** vom value-VoI-ohne-Archiv-Baseline (gepaarter CI `d_lo ≤ 0`), ist Erreichbarkeit **kein** separat hebelbarer Engpass auf dem fixen Action-Gitter → Befund: Erreichbarkeit ist an die offene Material-Achse (M4-Basis-Erweiterung) gebunden, nicht an Stepping-Stone-Heuristiken.

**notwendig-nicht-hinreichend:** Erreichbare Trittsteine sind **notwendig**, damit M1/M3/M4 ueberhaupt Tiefe erzeugen koennen (sonst nur unerreichbare Ziele, Angriff 3 auf M4). **Nicht hinreichend** und **sekundaer**: die Raumgroesse ist nicht der Engpass; dieser Hebel hat keinen eigenstaendigen Mechanismus ueber M1/M3/M4 hinaus.

---

## Naechste 3 konkrete Schritte gegen die Codebase (E0 / E1 / E2)

### E0 — M5 offline-validierbarer Teil (harter Blocker, KEIN Re-Run)
**Was:** In `artificial_society/research/metrics.py` (a) `graded_useful_advance` als Erweiterung von `accumulated_useful_depth` (`:290`) implementieren (Summe margin-normierter Frontier-Vorspruenge ueber **alle** Schichten, arm-symmetrisch, keine per-Agent-Aggregation); (c) `discovered_by`-Durchreichung im Export-Pfad reparieren, damit DV3 (`:415`) `computable=True` wird.
**Gegen welche Daten:** die **bestehenden** Pilot-Exports (kein neuer Sim-Lauf).
**Akzeptanz/Kill:** Floor-Test — `graded_useful_advance` gibt unter positiv-synthetischem Tiefen-Input **strikt > 2** aus (Knock-out-validiert gegen `metrics.py:27–30`); DV3 wird fuer ≥ 1 Seed computable. Schlaegt der Floor-Test fehl → M1–M4 bleiben gesperrt (Kill-Kriterium M5(a)/(d)).

### E1 — M5 record_use-via-Adoption + Re-Run (Blocker abschliessen)
**Was:** `record_use`-Hook auf **Adoptions-Events** (`culture.receive_transmitted` `culture.py:80`, `brain.imitate_from` `brain.py:186`), **nicht** Inventar-Konsum (Angriff 4); ohne Verhaltens-Edit der Golden. Re-Run im Standard-Regime (12 Seeds `1001–1012`, 1500 Ticks, CPU, `PYTHONHASHSEED=0`, GPU-PC via SSH).
**Akzeptanz/Kill:** DV1-`total_weight(weight_source='adoption') > 0` fuer den `learned` Arm. Bleibt es ≈ 0 (Hook ins Leere) → DV1 endgueltig als nicht-rettbar markieren und aus allen Gates entfernen (Kill M5(b)). Erst nach E0+E1 sind M1/M2 ausfuehrbar.

### E2 — M1 imaginierter Zwilling mit RNG-/uses-Isolation (Generator-Hebel)
**Was:** `_combine_pure` als reine, **RNG-snapshot/restore-isolierte** Funktion ueber `combine_vectors` (`materials.py:430`) implementieren (Muster `recombiner.py:54/113`); nebenwirkungsfreier Vektor-Getter (kein `uses`-Inkrement); seeded-Softmax + VoI (NICHT argmax); Env-Flag-gegated (OFF byte-identisch zur Golden); interner embodiment-gematchter Null (`voi_weight=0`, `RATCHET_GAIN=0`) als Entscheidungs-Baseline.
**Unit-Tests (Pflicht, vor jedem Sim-Lauf):** (1) imaginierte `combine_vectors` real gezaehlt == 0; (2) ON reale Aufrufe ≤ OFF; (3) **globaler RNG-State byte-identisch vor/nach dem imaginierten Sweep**; (4) `uses` unveraendert nach imaginiertem Sweep.
**Akzeptanz/Kill (GATED auf E0):** Positiv-Kontrolle (graded-Metrik unter irgendeiner Intervention > Floor beweglich) zuerst; dann gepaarter CI `graded(VoI-ON) − graded(VoI-OFF intern) > 0` UND `n_functional_clusters(ON) ≥ n_functional_clusters(OFF)` ueber 12 Seeds bei 1500 Ticks. Smokes zaehlen nicht.

---

*Erstellt 2026-06-30. Belegt-Markierungen mit `file:line` referenzieren `artificial-society` @ `feat/infra-research-stage0a`. Zitate getaggt: Resnick 2020, Hughes 2024, Baker 2019/2020, POET Wang 2019 / Enhanced POET 2020, Lehman & Stanley 2011, Mouret & Clune 2015, Eigen 1971, Van Valen 1973 = VERIFIZIERT. Soros & Stanley 2014 (Chromaria) = nur als Paraphrase + [UNVERIFIZIERT]; "Cook et al. 2024" = NICHT GEFUNDEN, in OPEN-QUESTIONS zu klaeren. Korrigierte Misattribution: Niche Construction / Evolution of Complexity ALIFE-IX 2004 = Tim Taylor (nicht Chli & De Wilde).*
