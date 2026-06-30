# DIAGNOSE.md — Code-verifizierte Ursachenanalyse: Warum der blinde Recombiner gewinnt

> **Geltung & Sprachregel.** Dieses Dokument trennt strikt **BELEGT** (code-verifiziert, mit `datei:zeile`) von **VERMUTET** (Hypothese, offene Frage). Es enthält **keine** Garantie-/Unvermeidbarkeits-Sprache. Befunde, die einen Blocker *entfernen* oder *plausibel notwendig* sind, werden als solche markiert; ob etwas *hinreichend* ist, entscheidet ausschließlich das Experiment.

---

## 0. Kernbefund in einem Satz

Der Defizit des gelernten Arms gegenüber dem blinden Recombiner **überlebt die Mess-Reform** (das ist eine Eigenschaft der Metrik, kein Artefakt) — der *bindende* Engpass ist **nicht** schlechte Messung und **nicht** mangelnde Transmissions-Fidelity, sondern **(C)** die fehlende *Policy-Kopplung der Generierung* ("WANN" ist gekoppelt, "WAS" nicht) **plus** ein **Embodiment-Confound** im Baseline-Design.

---

## 1. Warum der blinde Recombiner "gewinnt": (A) vs. (B) vs. (C)

Die naive Lesart lautet: "Ein zielloser Zufalls-Recombiner erreicht DAG-Tiefe ~22-32, der gelernte Arm nur ~2 — also lernt das System nichts Nützliches." Diese Lesart ist in drei konkurrierende Erklärungen zu zerlegen, von denen genau **eine** der bindende Engpass ist.

### (A) Mess-Artefakt — ANTIZIPIERT UND VERTEIDIGT, NICHT die Ursache

**BELEGT.** Die rohe DAG-Tiefe ist *per Design* als "counting artifact" deklariert; der Recombiner erreicht ~22 ohne jede funktionale Bedeutung (Docstring `research/metrics.py` ~Z.3-7). Die wertbasierten DVs der Schritt-A-Familie wurden gerade gebaut, um diesen Zähl-Artefakt zu neutralisieren:

- `structural_depths` (`artificial_society/research/metrics.py:85`): `sd(seed)=0`, `sd=1+max(inputs)`, zyklus-geschützt — das ist die *naive* Tiefe, die der Recombiner trivial maximiert.
- `functional_depths` (`artificial_society/research/metrics.py:115`): clustert Artefakte innerhalb `FUNC_TAU = 0.15` (`metrics.py:45`) und erbt die **minimale** Strukturtiefe des Clusters, um redundanten Churn zu kollabieren.
- `accumulated_useful_depth` / **DV2** (`artificial_society/research/metrics.py:290`, Docstring `:297`): vergibt Tiefenkredit nur, wenn ein Artefakt die Task-Frontier um `ADVANCE_MARGIN = 0.02` (`metrics.py:47`) auf einer **tieferen** Strukturschicht schlägt.

**Konsequenz (BELEGT durch gonogo-Doc):** Auf der reform-Metrik DV2 steht der gelernte Arm bei ~2, der Recombiner bei ~32 — das Defizit **bleibt** nach der Mess-Reform bestehen. "Die Metrik ist hackbar" ist für die aktuellen Schritt-A-Metriken damit **FALSCH** (Volumen-/Churn-Hacking ist durch die Margin-Klausel ausgeschlossen).

**CAVEAT A (VERMUTET, offene Frage):** DV2 sitzt am **Floor (=2)** für den gelernten Arm in **beiden** Armen (OFF *und* ON) jedes Seeds. Das ist *eine andere* mögliche Mess-Pathologie als Hackbarkeit: **Floor-Saturation für embodied Arme**. Ein Nullresultat am Floor widerlegt nichts (es kann "Generierung unzureichend" nicht von "Metrik ist am Floor blind" trennen). Dies ist der harte Grund, warum **M5 (Messreparatur) allen anderen Mechanismen vorgezogen** wird (siehe §7).

### (B) Direktiven-/Fidelity-Defizit — NICHT der bindende Engpass

Hypothese: Soziale Transmission verliert Information → die Ratsche greift nicht → keine Akkumulation. **BELEGT, dass diese Erklärung NICHT bindet**, weil die Fidelity bereits *hoch* ist:

- `FIDELITY_BASE = 0.72` (`artificial_society/systems/social_learning.py:26`), plus Trust-Bonus `FIDELITY_TRUST_BONUS = 0.18` (`:27`) → bis 0.90.
- Genetische Vererbung `INHERIT_FIDELITY = 0.70` (`artificial_society/simulation.py:34`), Todes-Broadcast `DEATH_BROADCAST_FIDELITY = 0.45` (`simulation.py:35`), `INHERIT_SEQUENCES = 10` (`simulation.py:33`).
- Eine **Ratsche existiert**: erfolgsgewichtetes `sample_for_transmission` (`artificial_society/systems/culture.py:68`, `weights = [v["successes"] ...]`); Korruptionsmechanik in `receive_transmitted` (`culture.py:80`, `if random.random() > fidelity`).
- Zusätzliche Kanäle: `social_learning_step` (`social_learning.py:39`), `brain.imitate_from` (`artificial_society/agents/brain.py:186`), Offspring-Vererbung kausaler Sequenzen (`spawn_child_from_parent`, `simulation.py:182`).

**Konsequenz:** (B) ist auf **bestätigenden Sweep herabgestuft** (Mechanismus M6). **VERMUTET / Geltungsgrenze (BELEGT-flankiert):** `sample_for_transmission` gewichtet nach `successes` (`culture.py:68`) = eine **statische** Selektionspressung, unabhängig von `FIDELITY_BASE`. Ein Fidelity-Sweep testet daher Kopier-Rauschen gegen ein *stationäres* Transmissionsziel, **nicht** die Ratschen-Kopplung an die Population (das ist die C1-Frage, M4s Aufgabe). Diese Geltungsgrenze ist hier dokumentiert.

### (C) Der ACTUAL BINDING ENGPASS — Generierung nicht policy-gekoppelt + Embodiment-Confound

**BELEGT.** Die PPO-Policy entscheidet das **WANN** der Erfindung, aber **nicht das WAS**:

- Das WANN ist policy-gekoppelt: `if need_magnitude < eff_threshold` (`artificial_society/systems/need_driven_invention.py:351`), Einstiegspunkt `agent_invent_from_need` (`need_driven_invention.py:318`).
- Das WAS ist quasi-zufällig: `_select_materials_by_need` / `_select_action_by_need` ziehen via Softmax + `np.random.choice` (`need_driven_invention.py:~305-310`). Die Auswahl, *welche* Materialien/Aktion kombiniert werden, ist **nicht** an die Policy gekoppelt.

→ **Der gelernte Arm ist mechanisch ein compute-gedrosselter, fragmentierter Zufalls-Recombiner**: gleiche blinde Kombinationslogik, nur mit weniger Versuchen und über embodied Agenten verstreut.

**PLUS Embodiment-Confound (BELEGT):** Der Vergleichs-Recombiner ist ein *disembodied perfect-memory Monopolist*:

- globaler, monoton wachsender Pool, **unconditional add** (`artificial_society/research/recombiner.py:104`, `pool_ids.append(mat_id)`);
- `discovered_by = -1` (`recombiner.py:99`) — keine Agenten-Attribution;
- `uses = 0` (`recombiner.py:101`) — kein Konsum;
- Seed = alle 24 MATERIALS, uniforme Aktions-/Material-Wahl (`recombiner.py:78`, `:76`);
- Compute-Match nur auf **gezählte** `combine_vectors` (`recombiner.py:69`, Schleife `for i in range(n_attempts)`); **nicht** embodiment-gematcht (keine Lokalität, keine Teilinformation, keine Fragmentierung).

→ Jeder Vergleich `learned-DV2~2` vs. `recombiner-DV2~32` **konfundiert** "Lerndefizit" mit "Embodiment-Penalty". Ohne einen embodiment-gematchten Null (M2) ist der Vergleich uninterpretierbar.

**VERDIKT:** (A) verteidigt, (B) nicht bindend, **(C) ist der bindende Engpass** — *notwendig zu adressieren, nicht hinreichend* (s.u.).

---

## 2. Die Metrik ist bereits ein Anti-Hacking-Design (Anker)

**BELEGT.** Die Schritt-A-Metriken sind konstruktiv gegen die offensichtlichen Hacks gehärtet:

| Schutz | Anker | Wirkung |
|---|---|---|
| "Counting-Artifact" explizit benannt | `metrics.py` Docstring ~Z.3-7 | rohe DAG-Tiefe wird *nicht* als DV verwendet |
| Funktionale Clusterung | `functional_depths` `metrics.py:115`, `FUNC_TAU=0.15` `:45` | redundante Vektor-Vielfalt kollabiert zu einem Cluster (Min-Tiefe) |
| Dedup-Schwelle | `DEDUP_TAU=0.08` `metrics.py:44` | Near-Duplikate zählen nicht doppelt |
| Margin-Gate | `ADVANCE_MARGIN=0.02` `metrics.py:47` | Tiefenkredit nur bei echtem Frontier-Vorsprung; Masse zählt 0 |
| Churn-Immunität / Arm-Symmetrie | `accumulated_useful_depth` `metrics.py:290`, DV2-Docstring `:297` | offline auf Exporten validierbar, gleiche Funktion für beide Arme |

**DV-Familie (BELEGT, mit Statusflags):**

- **DV2** `accumulated_useful_depth` (`metrics.py:290`): churn-immun, arm-symmetrisch, **offline validierbar**. → primärer Tiefen-DV. *Caveat:* Floor-Saturation (§1-A).
- **DV1** `population_functional_value` (`metrics.py:346`, Docstring `:352`): `provisional=True`, weil `uses` polluted ist — `get_vector` inkrementiert `uses` bei **jedem** Lookup (`materials.py:~294-298`) — und für den Recombiner strukturell 0 (`uses=0`, `recombiner.py:101`). Das ist der "fairness proof"-Befund: DV1 floored den Recombiner aus **Instrumentierungs-, nicht Verhaltensgrund**.
- **DV3** `transmitted_frontier_advances` (`metrics.py:415`, Docstring `:423`): `computable=False`, weil der Export kein `discovered_by` durchreicht (alle `-1`); k≥2-Adoption gefordert.
- **TASK_BASIS** = 6 vorab-registrierte Ziele (`metrics.py:261`); `analyze_registry` (`metrics.py:450`).

**Aggregation/Gate (BELEGT):** gepaarter Bootstrap-CI `_bootstrap_mean_ci` (`analyze_gate.py:41`), `N_BOOTSTRAP=10000` (`analyze_gate.py:38`); Verdikt-Pfade `PATH_A` (`:91`), `BORDERLINE_LEAN_A` (`:93`), `PATH_B_OR_RETROFIT` (`:95`). Paired Seeds `DEFAULT_SEEDS=range(1001,1013)` (`run_pilot.py:27`).

---

## 3. Das Path-A C+D NO-GO — und was es ausschließt

**BELEGT (gonogo-Doc `docs/research/stage0a-cd-coupling-gonogo-2026-06-30.md`).** Ein erster Retrofit auf der Generator-Seite (Phase C = wert-gekoppelter Reward; Phase D = imaginierte Means-Ends-Generierung) wurde gefahren: 4 Seeds (1001-1004) × 1500 Ticks, Grid 30×20, Pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1`.

Ergebnis:

- **DV2 blieb am Floor=2** in *jedem* Arm/Seed.
- Die Invention-**Effizienz** fiel von 0.127 → 0.029 (0/4 Seeds verbessert).
- Ein 250-Tick-Smoke zeigte C+D **+35% VORNE** — ein *Vor-Kollaps-Transient*, der sich bis ~1500 Ticks umkehrt.

**Was das AUSSCHLIESST / einschränkt:**

1. **Greedy-argmax-Means-Ends ist getestet und schlechter** — der Retrofit verschlechterte die Effizienz (greedy herdet auf wenige high-value Combos → Modus-Kollaps). Dies motiviert in M1 **seeded-softmax + Value-of-Information** statt argmax und in M3 eine **Diversitäts-erhaltende** Generierung (QD).
2. **250-Tick-Smokes LÜGEN** (der +35%-Transient kehrt sich um). Alle Kill-Kriterien werten ausschließlich terminale **1500-Tick**-Endpunkte über 12 Seeds.

**HONEST LIMITS (BELEGT-flankiert):** D nutzte `argmax` + 512-Novelty-Cap (NICHT das geplante seeded-softmax+VoI); C+D waren konflatiert (keine C-only-Zelle); es war **keine** volle end-to-end gelernte Kopplung. → **Der plan-getreue Generierungs-Hebel (seeded-softmax+VoI) ist STILL UNGETESTET.** Das NO-GO widerlegt *greedy* Means-Ends, nicht selektive Generierung als solche.

---

## 4. Die Diversitäts-Metrik DEFINIEREN — und das "0.127→0.029" korrekt zuordnen

Hier liegt eine häufige Fehlbeschreibung, die DIAGNOSE.md ausdrücklich richtigstellt.

### 4a. Die echte Populations-Diversitäts-Metrik (BELEGT)

`cultural_diversity()` (`artificial_society/systems/culture.py:134`) = **Anzahl distinkter kausaler Sequenzen, die von lebenden Agenten gewusst werden**. Sie liest die **lebende** Population: `population_sequences: dict[tuple, int]` (`culture.py:120`), die bei **Tod schrumpft** (im Gegensatz zur unsterblichen, append-only `DISCOVERY_REGISTRY`). Der Merkmalsraum ist `(action, mat_a, mat_b)`-Tupel — **nicht** genotypisch/verhaltensbasiert.

> **Implikation für die Mechanismen:** Ein Anti-Disengagement-Check (M4) muss über `population_sequences` (`culture.py:120`, schrumpfend) gemessen werden, **nicht** über die append-only `DISCOVERY_REGISTRY` (`materials.py:325`) — letztere ist monoton nicht-fallend by construction, ein Disengagement-Check darauf wäre *vacuous*.

### 4b. Das "0.127→0.029" ist KEINE Diversitäts-Metrik (BELEGT)

Die in der gonogo-Doc zitierte Zahl ist **Invention-Effizienz**, nicht Diversität und **nicht** ein temporaler Kollaps:

- Effizienz = `n_discoveries / ausgeführte combine_vectors` (`analyze_learning.py:71`, `eff = np.diff(cum, axis=1) / np.diff(grid)[None, :]`; gonogo-Doc Z.~38).
- Die zwei Zahlen sind **OFF-Arm (0.127) vs. ON/Retrofit-Arm (0.029)** — also: der Retrofit machte die **Rate schlechter** (greedy Means-Ends herdet auf wenige high-value Combos), **nicht** ein zeitlicher Diversitäts-Verfall.

**Richtigstellung (verbindlich):** Wer "0.127→0.029" als "Diversität bricht über die Zeit zusammen" liest, beschreibt das falsche Objekt. Die korrekte Diversitäts-Metrik ist `cultural_diversity()` (§4a); die zitierte Zahl ist ein **Effizienz**-Vergleich zwischen zwei **Armen**.

---

## 5. C3-Persistenz-Check am Reset-Code — PASSED

**BELEGT.** Die Persistenz-/Vererbungs-Voraussetzungen für Niche Construction sind erfüllt:

- **Kein Episode-Reset:** kontinuierliche `while self.running:`-Schleife (`artificial_society/simulation.py:493` bis `break` `:513`).
- **Singletons nur einmal zurückgesetzt:** `_reset_accumulating_singletons()` läuft im `__init__` (Aufruf `simulation.py:122`), **nicht** pro Tick/Episode.
- **Globaler Discovery-Pool persistent:** `DISCOVERY_REGISTRY = DiscoveryRegistry()` (`materials.py:325`), append-only.
- **Strukturen persistieren mit Decay:** `environment/structures.py` (über `world.set_cell`/`adjust_cell`, `world.py:54`/`:58`).
- **Offspring erben:** kausale Sequenzen (10 @ `INHERIT_FIDELITY=0.70`), `spawn_child_from_parent` (`simulation.py:182`, `for seq in list(parent_mem.sequences.keys())[:INHERIT_SEQUENCES]`).

**VERDIKT:** Niche Construction ist **mechanisch tragfähig**; C3 ist **KEIN** offener Engpass. (Direktion-neutrale *enabling condition*, nicht zentraler Treiber.)

---

## 6. World-Update-Engpass + Aktionsraum

### 6a. World-Update-Bottleneck (BELEGT)

`tick_growth()` (`artificial_society/environment/growth.py:113`) ist eine reine Python-Doppelschleife, O(Zellen × Materialien × Dispersal). **CPU ist ~7-11× schneller als GPU** (sparse/unvektorisiert). Orchestrierung über `tick_systems` (`artificial_society/systems/registry.py:125`), an die neue Module sich self-registrieren (relevant für M3 `qd_archive.py`, M4 `coevolution.py` — kein Hot-File-Edit nötig).

### 6b. Aktionsraum — gemischt (BELEGT)

- Primitive Aktionen **FIX**: 8-Enum `PRIMITIVE_ACTIONS = ["rub","strike","place_on_heat","bundle","blow","carry","eat","bind"]` (`artificial_society/systems/invention.py:52`).
- Strukturen fix (5).
- **Material-Raum OFFEN/rekursiv:** `combine_vectors` (`artificial_society/environment/materials.py:430`) erzeugt `mat_A + mat_B → mat_C` (adjacent-possible). Property-Indizierung `IDX` (`materials.py:47`).

**Implikation (VERMUTET, diagnose-gestützt):** Die Ausdruckskraft lebt in der **Material-Rekombination** — exakt der Achse, die der blinde Recombiner ausbeutet. C2 korrekt gelesen: nicht die Raumgröße ist der Engpass, sondern **Erreichbarkeit über Trittsteine**. Eine QD/Novelty-Pressung (M3) über einen *statischen, niedrigdimensionalen* Deskriptor (1 Property-Dim + 8 fixe Aktionen) liefert daher nur **endliche** Diversitäts-Pressung und versiegt, sobald das fixe Gitter abgedeckt ist (M3 ist deshalb als *fragiler Within-Arm-Hebel* markiert, nicht als Headline-DV).

> **Warnung (BELEGT):** `combine_vectors` konsumiert den **globalen `random`-Stream** (`materials.py:~452-455`, `random.random()`/`random.uniform` in den rub/ignite-Zweigen). Jede imaginierte Bewertungs-Schleife (M1s `_combine_pure`-Zwilling) muss `random.getstate()`/`setstate()` snapshotten (Muster `recombiner.py:54`/`:113`), sonst divergiert der ON-Arm im RNG-Stream gegenüber OFF — die Determinismus-/Golden-Garantie bräche, ohne dass ein reiner "reale Aufrufe == 0"-Test es bemerkte.

---

## 7. Synthese: Lever-Priorität und der harte Mess-Blocker

**BELEGT-gestützte Schlussfolgerung der Diagnose:**

1. **(C) ist der bindende Engpass** → primärer Hebel **M1** (policy-gekoppelte selektive Generierung, seeded-softmax+VoI, **nicht** greedy) und die **M2**-Embodiment-gematchte Baseline als billigste+härteste Falsifikation.
2. **Aber:** M1-M4 + M6 sind auf der **ungerepairten Messung** uninterpretierbar — DV2-Floor (Caveat A), DV1-`uses`-Pollution, DV3-non-computability. Daher ist **M5 (Messreparatur) VORGEZOGEN** als harter Blocker: non-gefloorte arm-symmetrische Metrik, adoptions-basierte Instrumentierung (`record_use` über Transmission/`imitate_from`, **nicht** über Inventar-Konsum — denn die einzige echte Dekrement-Stelle ist `sharp_stone` in `agents/agent.py:752`, ein NAMED material, kein `mat_XXXX`), und `discovered_by`-Durchreichung im Export.
3. **(B) Fidelity** → herabgestuft auf bestätigenden Sweep **M6** (Kill-Kriterium auf DV3 verschoben, M5-gated).

**Verbindliches Label für alle Mechanismen:** **notwendig-nicht-hinreichend.** Begründung gegen die Diagnose: M5 entfernt den Mess-Blocker, *erzeugt aber keine Komplexität* und dreht das Vorzeichen des Defizits nicht (Befund A überlebt die Reform). M1 entfernt den (C)-Generator-Blocker, adressiert aber **nicht** Transmission/Akkumulation über die Population (Fragmentierung) — das Floor-Kill auf der graded-Metrik widerlegt Suffizienz korrekt. Ob (C) zu adressieren *hinreichend* für offene kumulative funktionale Akkumulation ist, ist **offen** und wird durch das gepaarte 1500-Tick-Experiment entschieden, nicht durch dieses Dokument.

---

## 8. Offene Fragen (explizit, nicht garantiert)

- **(A')** Ist der DV2-Floor=2 eine Eigenschaft der *Embodiment/Messung* (statische 6-Task-Basis, `metrics.py:261`, Utilities saturieren by construction bei ~1.0) oder der *Generierung*? → entschieden durch M5s Positiv-Kontrolle (graded-Metrik muss unter synthetischem Tiefen-Input strukturell > 2 ausgeben können).
- **(C')** Kann der embodiment-gematchte Null (M2) den gelernten Arm *trotz* M1 auf der non-gefloorten Metrik schlagen? → falls ja: "Lerndefizit" ist ein **Embodiment-Artefakt**, Lernhypothese widerlegt (Path B).
- **Literatur:** "Cook et al. 2024" (in repo-internen gonogo/path-b-Docs als volle gelernte Policy-Kopplung referenziert) ist **NICHT auffindbar** — Autoren-Klärung nötig (möglicherweise fehlerinnert). Soros & Stanley 2014 (Chromaria): exakte Formulierung der vier notwendigen Bedingungen **UNVERIFIZIERT** — nur als Paraphrase + `[UNVERIFIZIERT]` zitieren.
