# CODEBASE-MAP.md — Modul- und Anker-Karte fuer die Mechanismen M1–M6

Dieses Dokument kartiert pro priorisiertem Lever die beruehrten Module/Dateien/Funktionen mit den Anker-Zeilennummern. Es trennt **belegt** (code-verifiziert, mit `file:line`) von **vermutet** (Design-Annahme, noch nicht im Code). Alle Mechanismen sind als **notwendig-nicht-hinreichend** gelabelt; jeder Eintrag verweist auf das Vier-Tupel (Mechanismus / null-kalibrierte compute-matched Baseline / hack-resistente funktionale Metrik / pre-quantifiziertes Kill-Kriterium) im zugehoerigen DESIGN-Dokument.

> **Bindende Schreibregeln eingehalten.** Keine Garantie-/Unvermeidlichkeitssprache. Formulierungen ausschliesslich der Form „entfernt einen bekannten Blocker“, „macht X moeglich, ohne es zu garantieren“, „ist plausibel notwendig fuer X“. Ob ein Mechanismus *hinreichend* ist, entscheidet das Experiment.

> **Determinismus-Kontrakt (belegt, CLAUDE.md „Determinismus ist heilig“).** Jeder neue Pfad ist env-gegated (`AS_PATHA_CD`-Muster) und im OFF-Zustand byte-identisch zur Golden-Trajektorie. Jede neue Zufallsquelle MUSS ueber `artificial_society.rng.seed_all` laufen oder den globalen `random`-Stream per `getstate`/`setstate` snapshotten (Muster `research/recombiner.py:54` / `:113`).

---

## 0. Querschnitt: Persistenz-Site (C3-Check) und World-Update-Engpass

Diese zwei Sites sind fuer JEDEN Mechanismus relevant und werden hier zentral verankert, statt pro Lever wiederholt.

### 0.1 Episode-Reset / Persistenz-Site — C3-Check: PASSED (belegt)

| Aspekt | Datei:Zeile | Befund |
|---|---|---|
| Kontinuierliche Tick-Schleife (kein Episode-Reset) | `artificial_society/simulation.py:493` (`while self.running:`) … `:513` (`break`) | **belegt** — eine durchgaengige `while`-Schleife, kein per-Episode-Reset im Loop. |
| Reset der akkumulierenden Singletons NUR im `__init__` | `artificial_society/simulation.py:122` (`_reset_accumulating_singletons()`) | **belegt** — laeuft genau einmal bei Konstruktion, nicht im Tick-Loop. |
| Globaler Discovery-Registry-Singleton (append-only) | `artificial_society/environment/materials.py:325` (`DISCOVERY_REGISTRY = DiscoveryRegistry()`) | **belegt** — monoton wachsend, importiert in `simulation.py`. |
| Vererbung kausaler Sequenzen an Nachkommen | `artificial_society/simulation.py:182` (`for seq in list(parent_mem.sequences.keys())[:INHERIT_SEQUENCES]`), Konstanten `:33` `INHERIT_SEQUENCES = 10`, `:34` `INHERIT_FIDELITY = 0.70`, `:35` `DEATH_BROADCAST_FIDELITY = 0.45` | **belegt** — Offspring erben Sequenzen mit Fidelity 0.70; Death-Broadcast 0.45. |
| Lebende-Population-Sequenzen (schrumpft bei Tod) | `artificial_society/systems/culture.py:120` (`self.population_sequences: dict[tuple,int] = defaultdict(int)`) | **belegt** — DIESE Struktur, nicht die append-only Registry, ist der korrekte Disengagement-Indikator (siehe M4). |

**Schlussfolgerung (belegt):** Niche-Construction-Persistenz + Heritabilitaet sind mechanisch vorhanden; **C3 ist KEIN offener Engpass.** Konsequenz fuer das Design: Jeder Mechanismus, der den Registry-Singleton mutiert, gefaehrdet C3 — M1/M3 sind deshalb so spezifiziert, dass sie NUR lesen (kein `register`, keine Elite-Loeschung in der Registry).

### 0.2 World-Update-Engpass (belegt)

| Aspekt | Datei:Zeile | Befund |
|---|---|---|
| Wachstums-Tick (reiner Python-Doppelloop) | `artificial_society/environment/growth.py:113` (`def tick_growth(world, season: str = "summer")`) | **belegt** — O(cells × materials × dispersal); CPU ~7–11× schneller als GPU (sparse/unvektorisiert). |
| Orchestrierung der Systeme | `artificial_society/systems/registry.py:125` (`def tick_systems(sim, tick: int)`) | **belegt** — zentraler Tick-Bus; neue Systeme (M3/M4) registrieren sich hier, statt `simulation.py` zu editieren. |
| World-Zellen-Mutatoren | `artificial_society/world.py:54` (`def set_cell`), `:58` (`def adjust_cell`) | **belegt** — Schreibpfade in die World-Fassade. |

**Konsequenz fuer das Design (vermutet, Feasibility-Caveat):** M4s nicht-stationaere Frontier-Berechnung pro Tick darf NICHT 30k Registry-Eintraege re-clustern; sie MUSS amortisiert/gecacht laufen, da das Regime eine FIXE Tickzahl (1500) und nicht Wall-Clock ist (Redteam M4-Angriff 6). Der `tick_growth`-Engpass setzt die praktische Obergrenze fuer zusaetzliche Pro-Tick-Arbeit.

---

## M5 — MESSREPARATUR (VORGEZOGEN, harter Blocker fuer M1–M4+M6)

**Lever-Prioritaet:** #0 (Vorbedingung). **Label:** notwendig-nicht-hinreichend, HARTER BLOCKER. **Vier-Tupel-Status:** Baseline = bestehende provisional/non-computable DVs auf denselben Pilot-Exports; Metrik IST das Artefakt (selbst-referentiell); Kill = vierteilig (siehe DESIGN). Redteam-Verdikt: **fragil** (Teile (b)/record_use und (a)/Asymmetrie wurden umgeschrieben).

### Beruehrte Module/Funktionen

| Teil | Datei:Zeile | Funktion/Konstante | belegt/vermutet |
|---|---|---|---|
| (a) Non-gefloorte, tiefenaufloesende Metrik | `artificial_society/research/metrics.py:290` (`def accumulated_useful_depth(`), DV2-Docstring `:297` | erweitern: statt `per_task_maxdepth` den kumulierten margin-normierten Vorsprungs-Betrag pro Strukturschicht summieren (`graded_useful_advance`) | **belegt** (Funktion existiert); **vermutet** (graded-Erweiterung ist neu) |
| (a) Task-Basis (Saettigungs-Ursache) | `artificial_society/research/metrics.py:261` (`TASK_BASIS = {`) | 6 Utility-Funktionen, jede Produkt geclampter Property-Dims → Saettigung ~1.0 | **belegt** |
| (a) Advance-Margin / Func-Tau | `artificial_society/research/metrics.py:47` (`ADVANCE_MARGIN = 0.02`), `:45` (`FUNC_TAU = 0.15`), `:44` (`DEDUP_TAU = 0.08`) | Frontier-Schwelle + Cluster-Radius | **belegt** |
| (a) Struktur-/Funktionstiefe (Frontier-Geometrie) | `artificial_society/research/metrics.py:85` (`def structural_depths`), `:115` (`def functional_depths`) | Tiefen-Berechnung, in die graded_useful_advance eingreift | **belegt** |
| (b) record_use auf ADOPTIONS-Events (NICHT Inventar-Konsum) | `artificial_society/systems/culture.py:80` (`if random.random() > fidelity:` — `receive_transmitted`) + `artificial_society/agents/brain.py:186` (`def imitate_from`) | Adoptions-Proxy ohne Verhaltens-Edit der Golden-Trajektorie | **belegt** (Sites existieren); **vermutet** (Hook neu) |
| (b) Begruendung gegen Inventar-Konsum-Hook | (Redteam M5-Angriff 4) einziger echter Inventar-Dekrement ist `sharp_stone` in `agents/agent.py:752` (NAMED material, kein `mat_XXXX`) | belegt: Hook auf Inventar-Konsum feuert ins Leere | **belegt** (Begruendung) |
| (b) Begruendung gegen `get_vector`-Zaehler | `artificial_society/environment/materials.py` `get_vector` inkrementiert `uses` bei JEDEM Lookup (Polluition-Quelle) | belegt: DV1-`uses` ist instrumentierungsverschmutzt | **belegt** (Begruendung) |
| (c) discovered_by-Durchreichung im Export | `artificial_society/environment/materials.py` (Feld vorhanden, im JSON-Export gedroppt) → DV3 computable | im Export-Pfad ergaenzen | **belegt** (Feld existiert); **vermutet** (Export-Fix neu) |
| Ziel-DVs | `:346` (`population_functional_value`, DV1 `:352`), `:290` (DV2), `:415` (`transmitted_frontier_advances`, DV3 `:423`), Computability-Flag `:432` | DV1 provisional, DV2 floored, DV3 computable=False | **belegt** |
| Knock-out-Validierung der Metrik | `artificial_society/research/metrics.py:27`–`:30` | Akzeptanzkriterien (a)/(b)/(c) gegen synthetischen Input | **belegt** |
| Offline-Analyse-Einstieg | `artificial_society/research/metrics.py:450` (`def analyze_registry`) | konsumiert die DVs auf Export-Eintraegen | **belegt** |

**C-Mapping:** Querschnitt-Messhygiene fuer die C5-Bewertung (saubere Trennung learner-vs-environment); adressiert Diagnose-Befund 1 (DV1 uses-Pollution / „fairness proof“) und Caveat A (DV2-Floor-Saettigung). **Notwendig-nicht-hinreichend:** entfernt den Mess-Blocker, dreht aber das Defizit-Vorzeichen NICHT (Diagnose A: Defizit ueberlebt die DV2-Reform).

---

## M2 — EMBODIMENT-MATCHED Baseline (attribuierter, konsumierender Null), GATED auf M5

**Lever-Prioritaet:** #2 (billigste + haerteste Falsifikation). **Label:** notwendig-nicht-hinreichend (reine Messhygiene/Falsifikation). **Vier-Tupel-Status:** M2 IST die Baseline (Drei-Arm pre-registriert). Redteam-Verdikt: **fragil** (Kill-Kriterium ist ohne M5-Floor-Reparatur durch Saettigung konfundiert → harter M5-Gate).

### Beruehrte Module/Funktionen

| Aspekt | Datei:Zeile | belegt/vermutet |
|---|---|---|
| Bestehender disembodied Recombiner (externer Null) | `artificial_society/research/recombiner.py:69` (`for i in range(n_attempts):` — Pool-Grow-Loop / compute-match) | **belegt** |
| Uniforme Action-Auswahl | `artificial_society/research/recombiner.py:78` (`action = random.choice(PRIMITIVE_ACTIONS)`) | **belegt** |
| Material-Paar-Wahl | `artificial_society/research/recombiner.py:76` (`mat_a, va = pool_ids[ai], pool_vecs[ai]`) | **belegt** |
| Dedup-Guard (DEDUP_TAU) | `artificial_society/research/recombiner.py:84` (`is_new = not (...)`) | **belegt** |
| Unconditional add (Monopol-Pool) | `artificial_society/research/recombiner.py:104` (`pool_ids.append(mat_id)`) | **belegt** |
| `discovered_by=-1` (Attribution fehlt) | `artificial_society/research/recombiner.py:99` | **belegt** |
| `uses=0` (kein Konsum) | `artificial_society/research/recombiner.py:101` | **belegt** |
| **NEU:** `run_recombiner_embodied` (K=pop disjunkte lokale Pools, per-tick-Attempt-Budget, Teilinfo, sozialer Austausch mit Fidelity, attribuiert + konsumierend) | neues Symbol in `artificial_society/research/recombiner.py` (beruehrt keine Hot-Files) | **vermutet** |
| Fidelity fuer Pool-Austausch im Null | `artificial_society/systems/social_learning.py:26` (`FIDELITY_BASE = 0.72`) | **belegt** (Konstante); **vermutet** (Verwendung im Null) |
| Combine-Operator (gemeinsam mit learned arm) | `artificial_society/environment/materials.py:430` (`def combine_vectors(`) | **belegt** |
| RNG-Isolation pro Agent (Redteam M2-Angriff 4) | Muster `recombiner.py:54`/`:113` (save/restore) → pro-Agent-Streams aus Seed, deterministische tick/agent-Reihenfolge | **belegt** (Muster); **vermutet** (Erweiterung) |
| Paired-Design / Seeds | `artificial_society/research/run_pilot.py:27` (`DEFAULT_SEEDS = list(range(1001, 1013))`), `:46` (`def run_one_seed`) | **belegt** |
| Gepaarter Bootstrap-CI | `artificial_society/research/analyze_gate.py:38` (`N_BOOTSTRAP = 10000`), `:41` (`def _bootstrap_mean_ci`) | **belegt** |
| Gate-Verdikte | `artificial_society/research/analyze_gate.py:91` (`PATH_A`), `:93` (`BORDERLINE_LEAN_A`), `:95` (`PATH_B_OR_RETROFIT`) | **belegt** |

**Metrik:** PRIMAER `graded_useful_advance` (M5), NICHT das gefloorte DV2 (Redteam M2-Angriff 5). SEKUNDAER DV3 (nach M5 + attribuiertem Null). **C-Mapping:** C5/C1 — trennt das gekoppelte, embodied, fragmentierte System vom disembodied Monopolisten („Umwelt vs Agent“ als falsche Dichotomie, Hughes 2024). **Notwendig-nicht-hinreichend:** ohne diese Kontrolle ist jeder learned-vs-recombiner-Vergleich uninterpretierbar; M2 erzeugt keine Komplexitaet.

---

## M1 — Policy-gekoppelte SELEKTIVE Generierung (seeded-softmax + Value-of-Information)

**Lever-Prioritaet:** #1 (ACTUAL BINDING ENGPASS, Diagnose C). **Label:** notwendig-nicht-hinreichend. **Vier-Tupel-Status:** Entscheidungs-Baseline = embodiment-matched INTERNER Null (voi=0, RATCHET_GAIN=0); Metrik CO-PRIMAER (graded + n_functional_clusters) auf HELD-OUT-Task-Basis; Kill GATED auf M5 + Positiv-Kontrolle. Redteam-Verdikt: **fragil** (Floor-Confound = Angriff 2 „does-not-survive on its own terms“ → erst nach M5-Gate auswertbar).

### Beruehrte Module/Funktionen

| Aspekt | Datei:Zeile | belegt/vermutet |
|---|---|---|
| WANN-Kopplung (PPO existiert) + quasi-zufaelliges WAS | `artificial_society/systems/need_driven_invention.py:318` (`def agent_invent_from_need(`) | **belegt** |
| Primitive Actions (fixe 8-Enum) | `artificial_society/systems/invention.py:52` (`PRIMITIVE_ACTIONS = [...]`) | **belegt** |
| Combine-Operator (Basis fuer den imaginierten Zwilling) | `artificial_society/environment/materials.py:430` (`def combine_vectors(`) | **belegt** |
| Property-Dim-Index | `artificial_society/environment/materials.py:47` (`IDX = {...}`) | **belegt** |
| **NEU:** ungezaehlter, RNG-isolierter `_combine_pure`-Zwilling (reine Funktion ueber `combine_vectors`, ohne `register`, ohne `uses`-Inkrement) | neues Symbol; ruft `materials.py:430` ohne Seiteneffekte | **vermutet** |
| RNG-Snapshot/Restore um den K-Kandidaten-Sweep (Redteam M1-Angriff 6) | Muster `recombiner.py:54`/`:113` (`random.getstate()`/`setstate()`) — Netto-0 RNG-Draws; Unit-Test: globaler RNG-State byte-identisch vor/nach Sweep | **belegt** (Muster); **vermutet** (Anwendung im Zwilling) |
| Nebenwirkungsfreier Vektor-Getter (Redteam M1-Angriff 4) | `get_vector` in `materials.py` inkrementiert `uses` bei jedem Lookup → der Zwilling MUSS gecachte/seiteneffektfreie Vektoren lesen | **belegt** (Pollution-Quelle); **vermutet** (Getter neu) |
| Seeded-Softmax statt argmax (NICHT greedy) | konzeptuell ueber die Auswahl in `need_driven_invention.py` (Score = value + voi·novelty + RATCHET_GAIN·marginal) | **vermutet** |
| Held-out-Metrik gegen Tautologie (Redteam M1-Angriff 1) | zweiter, im Generator NICHT verwendeter Task-Satz disjunkt von `TASK_BASIS` (`metrics.py:261`), ODER RATCHET_GAIN=0 | **belegt** (TASK_BASIS); **vermutet** (Held-out-Satz) |
| Env-Gating (`AS_PATHA_CD`-Muster), OFF byte-identisch zur Golden | konzeptuell; Determinismus-Kontrakt CLAUDE.md | **belegt** (Kontrakt); **vermutet** (Flag neu) |
| Anti-Greedy-Co-Metrik | `artificial_society/research/metrics.py:144` (`n_functional_clusters`) | **belegt** |
| Entscheidungs-DV (CO-PRIMAER) | `graded_useful_advance` (M5, erweitert aus `metrics.py:290`); gefloortes DV2 nur als Floor-Diagnostik | **belegt** (Basis); **vermutet** (graded) |
| Gepaarter Bootstrap-CI | `artificial_society/research/analyze_gate.py:41` (`_bootstrap_mean_ci`), `:38` (`N_BOOTSTRAP`) | **belegt** |
| Seeds / Regime | `artificial_society/research/run_pilot.py:27`, `:46` | **belegt** |
| Frueherer Effizienz-Befund (greedy-argmax scheiterte) | `artificial_society/research/analyze_learning.py:71` (`eff = np.diff(cum, axis=1) / np.diff(grid)[None, :]` — INVENTION EFFICIENCY = discoveries/attempts, NICHT Diversitaet) | **belegt** |

**Kill-Kriterium (belegt + vermutet):** GATED auf M5 (Floor-Test >2 bestanden); WIDERLEGT, wenn gepaarter CI von `graded(VoI-ON) − graded(VoI-OFF intern)` NICHT >0 (d_lo≤0) ueber 12 Seeds @ 1500 Ticks; ZUSATZ-Kill, wenn `n_functional_clusters(ON) < (OFF)`. Positiv-Kontrolle PFLICHT; 250-Tick-Smokes zaehlen NICHT (transient luegt; C+D war +35% @ 250, kollabierte @ 1500). **C-Mapping:** C5 (Generierung konstitutiv Teil des gekoppelten Systems, Hughes 2024); C2 (VoI/novelty nutzt offene Material-Rekombination `materials.py:430` als Ausdrucksraum; seeded-softmax respektiert Erreichbarkeit ueber stepping stones). **Notwendig-nicht-hinreichend:** entfernt den WAS-Kopplungs-Blocker (Diagnose C); adressiert NUR die Generator-Seite, nicht Transmission/Akkumulation ueber die Population.

---

## M3 — DIVERSITAETS-ERHALTENDE Generierung (QD / MAP-Elites / Novelty)

**Lever-Prioritaet:** #3, NUR als WITHIN-ARM A/B-Lever, NICHT als Open-Endedness-Headline. **Label:** notwendig-nicht-hinreichend, FRAGIL. **Vier-Tupel-Status:** Baseline = M1 mit reinem value-VoI-Sampler OHNE Archiv (within-arm, compute- UND embodiment-matched). Redteam-Verdikt: **fragil** (Angriff 1/2/3 = does-not-survive als Headline; ueberlebt nur die enge Within-Arm-Frage, Angriff 5).

### Beruehrte Module/Funktionen

| Aspekt | Datei:Zeile | belegt/vermutet |
|---|---|---|
| **NEU:** `systems/qd_archive.py` (self-registered, NUR Lese-Eingriff in M1s Score, KEINE Registry-Mutation) | neues Modul, registriert via `artificial_society/systems/registry.py:125` (`tick_systems`) | **vermutet** |
| Deskriptor: dominante aktive Property-Dim | `artificial_society/research/metrics.py:156` (`mean_active_dims`) | **belegt** |
| Deskriptor: Action-Klasse | `artificial_society/systems/invention.py:52` (`PRIMITIVE_ACTIONS`) | **belegt** |
| Archiv-Quelle (nur lesen) | `artificial_society/environment/materials.py:325` (`DISCOVERY_REGISTRY`) | **belegt** |
| Eingriffspunkt: M1s imaginierter Auswahl-Score | siehe M1 (`need_driven_invention.py:318` + `_combine_pure`-Zwilling) | **belegt** (Site); **vermutet** (Novelty-Bonus) |
| PRIMAER-Metrik (Within-Arm) | `artificial_society/research/metrics.py:144` (`n_functional_clusters`) gekoppelt an `graded_useful_advance` (M5, aus `:290`) | **belegt** (Cluster); **vermutet** (graded) |
| Gepaarter CI | `artificial_society/research/analyze_gate.py:41`, `:38` | **belegt** |

**Kill-Kriterium (belegt + vermutet):** GATED auf M5 (Redteam M3-Angriff 3: ohne M5-Floor-Reparatur feuert die eigene Floor-Klausel sofort durch Bestandsdaten). WIDERLEGT, wenn QD-ON gegen value-VoI-OHNE-Archiv keinen gepaarten CI-Anstieg von `n_functional_clusters` >0 zeigt; ODER wenn Cluster-Zuwachs mit `graded`-Rueckgang einhergeht (nutzlose Vielfalt); ODER wenn die M5-Metrik trotz QD am Floor bleibt. **C-Mapping:** C2 (stepping-stones/Erreichbarkeit, Lehman&Stanley 2011 VERIFIZIERT; Mouret&Clune 2015 VERIFIZIERT); C1 (Diversity-Pressure als Anti-Disengagement-Garde). **Notwendig-nicht-hinreichend, FRAGIL:** RESIDUAL-RISIKO (Redteam M3-Angriff 4) — Deskriptor ist statisch + niedrigdimensional (1 Property-Dim + 8 fixe Actions); die echte Ausdruckskraft liegt in OFFENER Material-Rekombination (Diagnose Befund 7). Ist das fixe Gitter abgedeckt, geht der Novelty-Bonus auf 0 → nur ENDLICHE Diversitaets-Pressung. M3 kann KEINE Headline gegen einen Monopolisten-Null liefern (out-clustert by construction).

---

## M4 — KOEVOLUTIONAERE NICHT-STATIONARITAET (C1) mit OFFENER Task-Basis-Erweiterung

**Lever-Prioritaet:** #4. **Label:** notwendig-nicht-hinreichend, FRAGIL, M5-abhaengig. **Vier-Tupel-Status:** Baseline = identisches System mit STATISCHER Basis (Flag OFF, byte-identisch zur Golden), compute-matched (keine zusaetzlichen `combine_vectors`). Redteam-Verdikt: **fragil** (Angriff 1/2/4/7 = does-not-survive in Originalform → neukonstruiert: offene Basis-Erweiterung statt Margin-Anhebung; Disengagement-Check ueber LEBENDE Population).

### Beruehrte Module/Funktionen

| Aspekt | Datei:Zeile | belegt/vermutet |
|---|---|---|
| **NEU:** `systems/coevolution.py` (self-registered) | neues Modul, registriert via `artificial_society/systems/registry.py:125` (`tick_systems`) | **vermutet** |
| Statisches 6-Ziel-Set (Saettigungs-Ursache) | `artificial_society/research/metrics.py:261` (`TASK_BASIS`) | **belegt** |
| Frontier-Geometrie (graded-Mess-Frontier) | `artificial_society/research/metrics.py:290` (`accumulated_useful_depth`), `:85` (`structural_depths`), `:115` (`functional_depths`) | **belegt** |
| Quelle fuer Komposit-Tasks (Population-Kopplung) | `artificial_society/environment/materials.py:325` (`DISCOVERY_REGISTRY`) + `artificial_society/systems/culture.py:120` (`population_sequences`) | **belegt** |
| Hall-of-Fame (Archiv bester je-erreichter Frontier-Artefakte) | im neuen `coevolution.py` | **vermutet** |
| **Anti-Disengagement-Check ueber LEBENDE Population** (Redteam M4-Angriff 4: useful_depth_max ueber append-only Registry ist monoton → vacuous) | `artificial_society/systems/culture.py:120` (`population_sequences`, schrumpft bei Tod) — NICHT die Registry | **belegt** |
| Metrik (NICHT gefloortes DV2) | `graded_useful_advance` (M5, aus `metrics.py:290`) ueber die ZEIT (Akkumulationssteigung) | **belegt** (Basis); **vermutet** (graded + Steigung) |
| Gepaarter CI / Seeds | `analyze_gate.py:41`, `:38`; `run_pilot.py:27`, `:46` | **belegt** |
| Amortisierte Pro-Tick-Frontier (Feasibility, Redteam M4-Angriff 6) | siehe World-Update-Engpass 0.2; FIXE Tickzahl, kein Re-Clustern von 30k Eintraegen | **belegt** (Engpass); **vermutet** (Cache) |

**Kill-Kriterium (belegt + vermutet):** GATED auf M5 (Headroom-Confound, Angriff 7). WIDERLEGT, wenn die graded-Akkumulationssteigung unter offener nicht-stationaerer Basis NICHT groesser ist als unter statischer Basis (gepaarter CI der End-Differenz ≤0 ueber 12 Seeds); ODER wenn Disengagement auftritt, gemessen ueber `culture.population_sequences` (faellt im letzten Drittel ggue. mittlerem Drittel). Steigung + Spaet-Fenster faengt den C+D-Reversal-Modus. **C-Mapping:** C1 (Kernachse korrigiert: STATISCH vs NICHT-STATIONAER, an eigene Population gekoppelt; Hall-of-Fame + M3-Diversity als Garde; Red Queen Van Valen 1973 / POET Wang 2019, beide VERIFIZIERT); C5; C2 (offene Basis-Erweiterung respektiert: das attainable maximum, nicht die Latte, ist der Engpass). **Notwendig-nicht-hinreichend, FRAGIL:** blosse Margin-/Bedrohungs-Anhebung hebt nur die Latte, nicht das Maximum (utility ≤ 1.0 by construction); ohne erreichbare Trittsteine (M1/M3) erzeugt Nichtstationaritaet nur unerreichbare Ziele. MLS/Gruppenselektion wird NICHT als Haupt-Lever verkauft (C4-Regel).

---

## M6 — ERROR-THRESHOLD / Evolvabilitaets-Balance der Transmissions-Fidelity (bestaetigend, herabgestuft)

**Lever-Prioritaet:** #5 (bestaetigender Sweep, Diagnose B: Fidelity NICHT bindend). **Label:** notwendig-nicht-hinreichend, HERABGESTUFT, FRAGIL. **Vier-Tupel-Status:** Baseline = live-Wert 0.72 als Within-Arm-Referenz + KONTROLL-ZELLE M1-ON/Fidelity-fest-0.72; Kill auf DV3 verschoben + Positiv-Kontrolle. Redteam-Verdikt: **fragil** (Angriff 2/3 = does-not-survive ohne M5; deshalb DV3-Verschiebung + harter M5-Gate).

### Beruehrte Module/Funktionen

| Aspekt | Datei:Zeile | belegt/vermutet |
|---|---|---|
| Sweep-Parameter (Copy-Fidelity) | `artificial_society/systems/social_learning.py:26` (`FIDELITY_BASE = 0.72`), Trust-Bonus `:27` (`FIDELITY_TRUST_BONUS = 0.18` → 0.90) | **belegt** |
| Zweiter Sweep-Parameter (Vererbung) | `artificial_society/simulation.py:34` (`INHERIT_FIDELITY = 0.70`), Death-Broadcast `:35` (`DEATH_BROADCAST_FIDELITY = 0.45`) | **belegt** |
| Sweep-Werte {0.45, 0.60, 0.72, 0.85, 0.95} um die Eigen-Error-Threshold-Kurve | konzeptuell ueber `social_learning.py:26` + `simulation.py:34` | **belegt** (Sites); **vermutet** (Sweep) |
| Korruptions-Mechanik | `artificial_society/systems/culture.py:80` (`if random.random() > fidelity:` — `receive_transmitted`) | **belegt** |
| Erfolgs-gewichtete Selektion (statisches Transmissions-Ziel — Geltungsgrenze, Angriff 4) | `artificial_society/systems/culture.py:68` (`weights = [v["successes"] for _, v in good]`, `sample_for_transmission`) | **belegt** |
| Soziales Lernen / Observe-Teach-Raten | `artificial_society/systems/social_learning.py:39` (`def social_learning_step`), `:24` (`OBSERVE_PROB = 0.18`), `:25` (`TEACH_PROB = 0.12`) | **belegt** |
| Imitations-Pfad (Adoptions-Proxy) | `artificial_society/agents/brain.py:186` (`def imitate_from`) | **belegt** |
| PRIMAER-Metrik (auf DV3 verschoben, fidelity-sensitiv) | `artificial_society/research/metrics.py:415` (`transmitted_frontier_advances`, DV3 `:423`), Computability `:432`, k≥2-Adoption (DV3-Logik) | **belegt** |
| SEKUNDAER-Metrik | `graded_useful_advance` (M5, aus `metrics.py:290`) | **belegt** (Basis); **vermutet** (graded) |
| Gepaarter CI / Seeds | `analyze_gate.py:41`, `:38`; `run_pilot.py:27`, `:46` | **belegt** |

**Kill-Kriterium (belegt + vermutet):** GATED auf M5 (DV3 computable, Angriff 3) UND M1-ON-Kontrollzelle (Angriff 6). Hypothese „Fidelity NICHT bindend“ (Diagnose B) BESTAETIGT (M6 als Lever VERWORFEN), wenn DV3 ueber den Sweep flach bleibt (kein Wert hebt den gepaarten DV3-Median ueber 0.72, alle CIs ueberlappen) UND die Positiv-Kontrolle zeigte, dass DV3 ueberhaupt fidelity-beweglich ist. UMGEKEHRT (ueberraschend): hebt ein Nicht-0.72-Wert DV3 ueber gepaarten d_lo>0 UND die M1-Kontrollzelle schliesst M1 als Ursache aus → (B) teilweise falsch, Fidelity aufgewertet. Per-Arm-Computability von DV3 vor jeder Auswertung asserten (Recombiner ohne Agenten waere arm-asymmetrisch). **C-Mapping:** C4 (kumulative Kultur via hochfideler Transmission, unbestrittener Kern; Eigen-Error-Threshold-Balance 1971; OHNE die drei Paare zu verwechseln — cultural-vs-genetic, group-vs-individual, coordination-vs-IQ; MLS NICHT als Haupt-Lever). **Notwendig-nicht-hinreichend, HERABGESTUFT, FRAGIL:** Fidelity ist bereits HOCH (0.72→0.90 mit Trust) und laut Diagnose B nicht bindend; M6 testet nur, dass die Fidelity-Achse kein Sub-Optimum ist. Geltungsgrenze: `sample_for_transmission` gewichtet nach `successes` = STATISCHE Selektions-Pressung → M6 misst Copy-Noise gegen ein STATIONAERES Ziel, nicht die Ratchet-Kopplung (C1 = M4s Aufgabe). Diese Grenze MUSS in DIAGNOSE.md dokumentiert werden.

---

## Anhang: Abhaengigkeits- und Gating-Reihenfolge (belegt aus den Vier-Tupeln)

```
M5  (Messreparatur, VORGEZOGEN)  ─── harter Blocker fuer ───►  M1, M2, M3, M4, M6
M2  (embodiment-matched Null)    ─── GATED auf M5 (Floor) ──►  Falsifikation von M1
M1  (selektive Generierung)      ─── GATED auf M5 + Pos.-Kontrolle
M3  (QD/Novelty, within-arm)     ─── GATED auf M5; haengt an M1s Auswahl-Score
M4  (Koevolution / C1)           ─── GATED auf M5 (Headroom); braucht M1/M3 fuer Trittsteine
M6  (Fidelity-Sweep, bestaetigend) ─ GATED auf M5 (DV3 computable) + M1-ON-Kontrollzelle
```

**Compute-Regime (belegt, alle Laeufe):** 4–12 paired seeds (`run_pilot.py:27` → 1001–1012), 1500 Ticks, Grid 30×20, Pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`, GPU-PC via SSH. Gepaarter Bootstrap-CI via `analyze_gate.py:41` mit `N_BOOTSTRAP = 10000` (`:38`). 250-Tick-Smokes LUEGEN (transient; dokumentierter C+D-Reversal).

**Hot-Files-Kontrakt (belegt, CLAUDE.md):** `simulation.py`, `world.py`, `agents/agent.py`, `agents/brain.py`, `environment/materials.py`, `systems/registry.py` sind eingefrorene Vertraege. M3/M4 fuegen neue self-registered Module unter `systems/` hinzu (Eingriff ueber `registry.py:125`, NICHT durch Edit der Hot-Files); M2 erweitert die research-Lane (`recombiner.py`); M5/M1 lesen die Combine-/Vektor-Pfade seiteneffektfrei (RNG-Snapshot, kein `uses`-Inkrement, kein `register`), um Determinismus und C3-Persistenz intakt zu halten.
