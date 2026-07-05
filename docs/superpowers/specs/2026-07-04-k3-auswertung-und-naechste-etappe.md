# K3-Auswertung & nächste Bau-Etappe (M1-Diagnose-Batterie, Re-Run)

Datum: 2026-07-04 · Daten: `battery_k3/` (32 Läufe = 4 Arme × 8 Seeds 41–48, 5000 Ticks, 40×30,
Start-Pop 30, `--min-pop 4 --respawn-count 2`) · Fix (a) aktiv (Nearby-Cache seit 5fec1d0 lebendig).
Arme: **A1 `learn`** (volle Maschinerie), **A2 `nolearn`** (`maybe_train`/`imitate_from` no-op),
**D1 `nosocial`** (`social_learning_step` no-op, PPO an), **A3 `nolearn-nosocial`** (echte Random-Null).
Auswertung: gepaart über Seeds (Welt-Init je Seed identisch bis zum ersten Eingriffs-Tick) +
Zeitreihen (Snapshot alle 250 Ticks) + Demografie + Code-Beleg (`agents/brain.py`, `agents/agent.py`,
`simulation.py`).

---

## 0. Kernergebnis vorab

**Alle vier Arme kollabieren auf denselben künstlichen Respawn-Boden (Pop ~4–5).** Damit ist die
**Gültigkeits-Vorbedingung der Spec (§3: „ein Arm, der nur die Population kollabiert, senkt Discovery
trivial — solche Arme sind ungültig")** für **jeden** Arm verletzt. Kein Arm-Vergleich (H3, H4)
ist unter dieser Demografie sauber entscheidbar. Der eigentliche Blocker ist **nicht mehr Lernen oder
Kultur, sondern die Ökologie/Demografie: die Population trägt sich nicht selbst.** Das supersediert
den Entscheidungsbaum, dessen Blätter alle eine nicht-kollabierte Population voraussetzen.

Die einzigen robusten, monotonen Signale sind ein **leichter learn-Malus** (A1 hat in 8/8 Seeds
niedrigere `mean_energy` als D1, in 6/8 niedrigere als A2; in 6/8 niedrigeres `mean_age` als A2) und
die mechanistische Unmöglichkeit von Lernen bei dieser Lebensdauer (**~3 PPO-Updates pro Agentenleben**).

---

## Analyse 1 — Gepaarte Auswertung & Zeitreihen

### 1a. Arm-Mediane (final, Tick 5000)

| Arm | tool_cut_ratio | cuts_with_tool | discoveries | mean_energy | mean_age | pop | deaths | respawns |
|---|---|---|---|---|---|---|---|---|
| A1 learn | 0.001 | 1.0 | 12.5 | 70.2 | 210.8 | 4 | 111 | 21 |
| A2 nolearn | 0.020 | 18.5 | 11.5 | 78.5 | 262.5 | 4 | 111.5 | 22.5 |
| D1 nosocial | 0.015 | 21.5 | 13.0 | 99.2 | 228.1 | 5 | 104.5 | 20 |
| A3 nolearn-nosocial | 0.002 | 3.0 | 12.0 | 107.3 | 387.5 | 4 | 109 | 20 |

### 1b. Vorzeichen-Bilanz je Seed (A minus B, +/−/0 über 8 Seeds)

| Metrik | A1−A2 | A1−D1 | A2−A3 |
|---|---|---|---|
| tool_cut_ratio | +2 / −4 / 2 (med −0.020) | +2 / −5 / 1 (med −0.014) | +4 / −3 / 1 (med +0.020) |
| cuts_with_tool | +2 / −4 / 2 (med −18.5) | +2 / −5 / 1 (med −20.5) | +4 / −3 / 1 (med +18.5) |
| discoveries | +5 / −2 / 1 (med +1.5) | +3 / −3 / 2 (med 0) | +4 / −3 / 1 (med +0.5) |
| mean_energy | +2 / −6 / 0 (med −27.3) | **0 / −8 / 0** (med −42.2) | +3 / −5 / 0 (med −27.3) |
| mean_age | +2 / −6 / 0 (med −52.0) | +5 / −3 / 0 (med +3.6) | +2 / −6 / 0 (med −91.5) |

**Belege / Lesart:**
- **tool_cut_ratio / cuts_with_tool sind eine Zwei-Seed-Lotterie, kein Behandlungseffekt.** Die
  Verteilung ist stark null-inflationiert: pro Arm tragen 2–3 „Glücks-Seeds" (v.a. 44, 47, teils
  42/43) fast die gesamte Masse (z.B. A2 seed44 = 232 cuts_with_tool, seed41/45/46/48 = 0). Alle
  IQR überlappen massiv (Spec-Schwelle → alle Vergleiche „≈"). Das nicht-additive Muster
  (learn/social-an niedrig, nosocial/PPO-an hoch, nolearn/social-an hoch, Random-Null niedrig) ist
  mit n=8 und dieser Streuung nicht von Rauschen trennbar.
- **Robust ist nur der Energie-Malus von A1:** A1 < D1 in **8/8** Seeds (Vorzeichentest p≈0.008),
  A1 < A2 in 6/8. A1 hat den niedrigsten Median `mean_energy` (70) aller Arme.
- Die Achse mit dem klarsten Muster ist `mean_energy`: **social-AUS-Arme (D1 99, A3 107) > social-AN-Arme
  (A1 70, A2 78)** — Ko-Lokation für Social-Learning erzeugt lokale Nahrungs-Konkurrenz oder Kosten;
  der Social-Reward selbst wird im v2 ohnehin verworfen (s.u.). Spekulativ, nur learn<nosocial ist
  IQR-getrennt.

### 1c. Lernkurven A1 vs A2 (Zeitreihe, Median über Seeds, kumulativ)

`cuts_with_tool` über die Zeit (Snaps 250…5000):
- **A2 nolearn:** 0, 0, 3, 3.5, 7, 7, 8.5 … 18.5 — **stetiger Aufbau** über den ganzen Lauf.
- **A1 learn:** 0, 0, 0, 0 … 0, 0.5, 1, 1 — **flach ~0 bis Tick 4750.**

Das ist das aussagekräftigste Zeitreihen-Signal: **PPO trainiert das (zufällige) Werkzeug-Schneiden
heraus** statt es zu verstärken. Die eingefrorene Zufallspolicy behält ihre Baseline-Tool-Rate; die
lernende Policy driftet davon weg. `discoveries` steigen in beiden Armen früh (bis ~Tick 1500) und
sättigen dann bei ~10–13 — kein Lern-Vorsprung von A1.

**Divergenz-Zeitpunkt vs. Trainings-Onset:** Der erste geplante PPO-Update fällt bei
`ROLLOUT_HORIZON=128` frühestens auf Tick ~128 (ein Transition/Tick). Die A1/A2-Trajektorien
divergieren aber erst spürbar nach dem Populations-Peak (~Tick 500), sobald der Kollaps einsetzt —
d.h. die Divergenz koinzidiert mit dem **demografischen** Regime-Wechsel, nicht mit dem Trainings-Onset.
Vor dem Kollaps (t≤500) sind A1−A2-Energiedifferenzen klein und wechselnd im Vorzeichen.

---

## Analyse 2 — Demografie-Diagnose (der eigentliche Befund)

### 2a. Populations-Verlauf: Overshoot → Crash → Respawn-Boden

Pop-Median-Trajektorie (Start 30), alle Arme nahezu identisch:

```
Start 30 → t250 ~41 → t500 ~44 (Peak) → t750 ~10 (Crash) → t1000 ~5 → t1250+ 4–5 (Boden)
```

- **Gründer-Boom:** Bis Tick ~500 verdoppelt sich die Population (Peak-Median 43–47) durch einen
  synchronen Geburts-Boom (Snapshot-`births` = Σ `children` lebender Agenten, Peak ~36–43).
- **Crash:** Zwischen Tick 500 und 750 bricht die Population von ~44 auf ~10 ein (Median-`deaths`
  springt von 18 auf ~60 kumulativ) — klassischer Malthusianischer Overshoot: die synchron geborene
  Kohorte überschreitet die Nahrungs-Tragfähigkeit des 40×30-Feldes und verhungert synchron.
- **Boden:** Ab ~Tick 1000–1250 ist die Population am `min-pop`-Boden (4–5) festgenagelt.

Erst-Unterschreitung Pop≤5: Median Tick ~1000–1125 in allen Armen.

### 2b. Trägt sich irgendeine Population selbst? — **Nein.**

Steady-State (t>1000), Median pro 250-Tick-Fenster, alle Arme praktisch gleich:

| | births/250t | deaths/250t | respawn-Aufrufe/250t (×2 Agenten) |
|---|---|---|---|
| alle Arme | ~0 (Mittel ~1.0) | 3.0 | 1.0 (→ 2 Zufallshirne) |

Im Boden-Regime ist die Reproduktion praktisch **aus** (Snapshot-`births` ~0–1 = lebende Agenten
haben ~0 zugeschriebene Kinder). Die 4–5 „Bewohner" sind ein **respawn-gespeister Durchlauf**: pro
250 Ticks sterben ~3 und werden ~2 frische **Zufallshirne** eingesetzt. Über den Lauf: ~110 Tode,
~20–22 Respawn-Aufrufe → **~40–44 Zufallshirn-Respawns bei ~107–111 je gesehenen IDs**. Die Mühle
ist gegenüber Batterie 1 reduziert, aber **die Boden-Population ist keine reproduzierende Linie,
sondern ein Zufalls-Nachschub.**

**Ursache (Code):** `agents/agent.py` — Reproduktion erfordert `energy≥60`, `age≥60`, Cooldown 100,
weiblich, UND dichteabhängig `local_food_per_capita ≥ 6.0` (`REPRODUCTION_MIN_FOOD_PER_CAPITA`). Der
Dichte-Gate wurde laut Docstring eingebaut, um den Overshoot-Crash zu dämpfen und die Population
„near carrying capacity" zu halten. Gemessen jedoch: die effektive Tragfähigkeit auf 40×30 mit
`herbs.regrow_rate` 0.06–0.12 liegt **unter der Ersatzrate** → die Population pendelt sich nicht bei
Tragfähigkeit ein, sondern **unter dem Respawn-Boden**. Der Gate verhindert den Crash nicht (er kommt
zu spät gegen die synchrone Gründer-Kohorte) und blockiert danach die Erholung.

### 2c. Ist individuelles Lernen mechanistisch möglich? — **Nein.**

`ROLLOUT_HORIZON=128` (`brain.py:62`): der erste geplante `maybe_train`-Update braucht 128 gelebte
Transitionen; bei Tod flusht `finalize_terminal` den Restbuffer (< 128) mit dem −3.0-Terminal und
trainiert einmal partiell (`simulation.py:327`, `brain.py:1006`).

- Steady-State-Lebensdauer (Agent-Ticks / Tode im Fenster t>1000): **Median ~360–400 Ticks.**
- → **~2.8–3.1 geplante Updates + 1 Terminal-Update ≈ 3–4 PPO-Updates pro Agentenleben.**

PPO braucht Hunderte von Updates, um eine ~100-Tick-Kredit-Kette (Knapping→Kadaver→Schneiden→Essen)
zu propagieren. Mit **3 Updates auf kleinen, todes-terminierten Batches** kann die Policy die
Zufalls-Initialisierung nur *perturbieren*, nicht *lernen*. **Individuelles Lernen hat unter dieser
Demografie keinen mechanistischen Raum.**

### 2d. Ist kulturelle Weitergabe möglich? — Substrat fehlt.

Bei Boden-Pop 4–5 auf 1200 Zellen und Lebensdauern ~360 Ticks existieren Ko-Existenz-Fenster, aber
(i) es gibt keine stabile Mehr-Generationen-Population, durch die transmittiert werden könnte, und
(ii) es akkumuliert kaum Wissen, das weitergegeben würde (Discoveries sättigen bei ~10–13, meist von
Gründern). Zudem sind ~40 % der je gesehenen Agenten frische Zufallshirn-Respawns — die kein Erbe
tragen. **Weder Selektion (kein differentielles Reproduzieren — alle sterben am Boden, aufgefüllt
per Zufall) noch Kultur haben ein tragfähiges Substrat.**

---

## Analyse 3 — Der learn-Malus

Fakten: A1 ist nominal schlechtester Arm bei `mean_energy` (70 vs A3 107) und bei tool-Metriken
(0.001 vs A2 0.020); A1 driftet in der Zeitreihe von der Tool-Baseline **weg** (1c).

### v2-Reward (Code-Beleg, `agent.py:1556`, physics_v2-Zweig)

```
reward = 0.6·(Δenergy)/45 + 0.6·(Δhealth)/50 − 0.3·max(0,(60−energy)/60) + 0.3·curiosity
```
mit `curiosity ∈ [0,2]` (`_assemble_curiosity_v2`). **Der bis dahin akkumulierte v1-Reward
(Forage-Event, Material, Koop, Territorium, Trade, Social Learning) wird im v2 bewusst VERWORFEN**
(Kommentar `agent.py`: „die Mechanik lief, sie zahlt nur nicht"). Der −3.0-Todesmalus kommt einmal
in `finalize_terminal`.

### Hypothese (a): Reward-Fehlallokation (Exploration statt Überleben) — **belegt als plausibel**

Magnituden: Curiosity trägt bis **0.3·2.0 = 0.6 pro Tick** und feuert **jeden** Tick; ein voller
Mahlzeit-Gewinn (+45 Energie) trägt einmalig `0.6·45/45 = 0.6`. Über ein Leben dominiert der
Curiosity-Term den akkumulierten Reward, weil er permanent zahlt, Mahlzeiten aber selten sind. Die
Werkzeug-Kette (strike/knapping→cut→eat) hat im v2 **keinen** geshapten Payoff außer der eventuellen
Energie-Differenz beim Essen — die v1-Material-Belohnung, die genau das früher zahlte, ist gestrichen.
**Der einzige erreichbare Reward-Gradient in 3 Updates zeigt auf direktes Essen (bare-hand) und auf
Exploration, nicht auf die lange Tool-Kette** — konsistent mit `cuts_bare_hand` ~1324 vs
`cuts_with_tool` ~1 und dem Weg-Driften von der Tool-Baseline (1c). *Beleg: Reward-Formel-Magnituden +
explizit verworfene v1-Belohnung. Nicht direkt gemessen: die relative Reward-Zusammensetzung pro Lauf
(nicht instrumentiert) — daher „plausibel", nicht „bewiesen".*

### Hypothese (b): Instabilität/Vergessen durch Todes-Training bei kurzen Leben — **plausibel, nicht von (a) trennbar**

Jeder Tod triggert ein Training auf einem partiellen Buffer (`finalize_terminal`) mit −3.0-Terminal;
bei ~3 Updates/Leben und `max(2, n//4)`-Minibatches sind das wenige, hochvariante Gradientenschritte
auf einer noch nicht konvergierten Policy. Das kann eine anfänglich brauchbare Zufallspolicy eher
destabilisieren als verbessern — konsistent mit A1 < A2/D1 auf Energie/Alter.

### Synthese

Beide Hypothesen reduzieren sich auf **dieselbe Ursache: zu wenige Updates pro Leben (Analyse 2c).**
Mit 3 Updates kann PPO nur in Richtung des dominanten, kurzhorizontigen Reward-Terms (Curiosity/
direktes Essen) perturbieren und verliert dabei gegen die eingefrorene Baseline. Die Metriken erlauben
**keine** saubere Trennung von (a) und (b) — beide sind Symptome des Lebensdauer-Blockers.
**Trainings-Rechenzeit ist ausgeschlossen** (kein Einfluss auf Sim-Ticks, nur Walltime).

---

## H3/H4-Status unter den neuen Bedingungen

**Entscheidungsbaum-Vorbedingung verletzt:** Der Baum (Spec §3) setzt nicht-kollabierte, vergleichbare
Populationen voraus. In K3 sind **alle** Arme kollabiert (Pop 4–5) → jeder Arm ist nach der Spec-eigenen
Regel „ungültig, nicht Hypothese bestätigt". Zudem lief K3 nur die 4 Arme A1/A2/D1/A3 — **die Billiger-Lever-
Arme C2/E1/F1 (Schritt 0) wurden nicht ausgeführt und sind nicht bewertbar.**

- **H4 (Lernen bringt nichts):** Nominal **A1 ≈ A2** auf allen Emergenz-Metriken (IQR-überlappend), und
  A1 liegt **auf** dem Random-Null-Niveau A3 (discoveries 12.5 vs 12; tool 0.001 vs 0.002). Die volle
  Maschinerie ist von der reinen Zufalls-Null nicht zu unterscheiden — und auf Energie/Alter leicht
  schlechter. **Nominell H4-„wahr"**, ABER die Ursache ist demografisch (3 Updates/Leben), nicht bewiesen
  als „Lern-Maschinerie fundamental kaputt". Der sauber ableitbare Satz lautet: **Lernen kann unter
  dieser Demografie nicht helfen, weil es keinen mechanistischen Raum hat.**
- **H3 (Kultur trägt):** **Nicht entscheidbar.** D1 (nosocial) liegt nominal *über* A1 bei Energie und
  ≈ bei Tools — aber beide kollabiert, IQR-überlappend, Signal von 2 Seeds getragen. Der Kausal-Confound
  (toter Nearby-Cache) ist gefixt, doch ein **neuer Confound (Pop-Kollaps) dominiert** und macht D1↔A1
  ungültig. Weder „Kultur trägt" noch „Kultur trägt nichts" ist belegbar.

**Sauber entscheidbar bleibt nur:** (1) Die Demografie trägt sich nicht selbst (Analyse 2). (2) Lernen
hat ~3 Updates/Leben (2c). (3) Ein robuster, aber kleiner learn-Energie-Malus existiert (1b, 8/8 vs D1).
Alles Übrige (H3, H4-Ursache, Tool-Effekte) ist **offen**, weil im kollabierten Regime nicht messbar.

---

## Priorisierte Bau-Empfehlung mit Abhängigkeitskette

**Abhängigkeitskette (Wurzel zuerst):**

```
[1] Selbsttragende Demografie  ──enables──▶  [2] Lern-Lebensdauer + Reward-Rebalance
                                                      │
                                                      └──enables──▶ [3] Kultur-Kanal (H3) + volle Ablations-Batterie
```

Nichts in [2]/[3] ist **messbar**, solange [1] offen ist — jede Arm-Metrik bleibt sonst im
Spec-ungültigen Kollaps-Regime.

### [1] Demografie/Ökologie zu selbsttragender Population — **PRIORITÄT 1 (blockierend)**

Evidenz: Analyse 2a/2b — Overshoot→Crash→Respawn-Boden in allen Armen; Reproduktion im Steady-State
aus; Tragfähigkeit < Ersatzrate. **Dies ist die Voraussetzung für die Messbarkeit von allem Übrigen.**
Stellschrauben (Lane `environment`/`core`), in Wirkungs-Reihenfolge:

1. **Age-structured founders** (höchster Hebel gegen den Overshoot-Crash): Gründer mit gestreuten
   Startaltern statt synchroner Kohorte → keine synchrone Geburts-Welle → kein synchroner Massentod.
   MEMORY notiert dies bereits als „echten Fix" gegen den Gründer-Overshoot (dichteabhängige
   Fruchtbarkeit dämpfte ihn nicht).
2. **Tragfähigkeit anheben**: `herbs.regrow_rate` (0.06–0.12) und/oder `plant_food`-Regrowth erhöhen,
   ODER Metabolismus-Kosten senken — bis die Ersatzrate erreichbar ist. (MEMORY: proportionaler
   Windschaden hielt Vegetation bereits am Leben; hier prüfen, ob der Effekt auf 40×30 reicht.)
3. **Dichte-Gate justieren**: `REPRODUCTION_MIN_FOOD_PER_CAPITA=6.0` ist auf dem kleinen Feld evtl. zu
   streng und blockiert die Erholung nach dem Crash — testen, ob ein niedrigerer Wert die Boden-
   Population wieder wachsen lässt, ohne den Overshoot zurückzuholen.

**Erfolgs-/Messkriterium (harter Gate für [2]/[3]):** Ein 5000-Tick-Lauf, in dem `respawns → ~0`,
`births ≈ deaths` im Steady-State, und Pop **deutlich über** dem `min-pop`-Boden stabil bleibt
(Ziel ~20–40). `min-pop` **nicht** erhöhen — das maskiert nur.

*Verworfen als Erst-Schritt:* alles unter [2]/[3] — nicht messbar vor [1].

### [2] Lern-Lebensdauer + Reward-Rebalance — **PRIORITÄT 2 (nach [1] messbar)**

Evidenz: Analyse 2c (3 Updates/Leben), Analyse 3 (Curiosity ~ Mahlzeit pro Tick, Tool-Kette ohne
Shaping). Zwei Sub-Hebel, beide erst sinnvoll testbar, wenn Agenten lang genug leben:

- **(2a) Reward-Rebalance Survival vs. Curiosity** *(billig, Code-Change vorziehbar, aber Effekt erst
  nach [1] messbar)*: Curiosity-Gewicht senken und/oder ein kleines geshaptes Signal für die
  knapping→cut→eat-Kette wieder einführen (die verworfene v1-Material-Belohnung gezielt teil-reaktivieren).
- **(2b) Kredit-Horizont / mehr Updates pro Leben**: `ROLLOUT_HORIZON` senken (häufigere Updates) und/
  oder Eligibility-Traces/Kredit-Backfill für die ~100-Tick-Kette; der strukturelle Fix ist aber die
  **längere Lebensdauer aus [1]** plus ggf. persistente/Linien-Brains, damit Updates über Generationen
  akkumulieren.

**Messkriterium:** Nach [1] eine A1-vs-A2-Wiederholung — erst jetzt kann `learn` `nolearn` auf
`tool_cut_ratio`/`discoveries` überhaupt schlagen. Vorher ist H4 nicht fair testbar.

### [3] Kultur-Kanal (H3) + volle Ablations-Batterie — **PRIORITÄT 3 (nach [1]+[2])**

Evidenz: Analyse 2d (kein Substrat), H3-Status (nicht entscheidbar). Erst wenn eine stabile Mehr-
Generationen-Population existiert **und** individuelle Brains lernen, gibt es (i) etwas zu
transmittieren und (ii) eine Linie, durch die transmittiert wird. Dann: D1-Ablation erneut + die in
K3 ausgelassenen Arme **C2/E1/F1** (Schritt 0 des Entscheidungsbaums), damit der Baum auf **gültigen**
Daten läuft. Sozial-Kanal-Verstärkung nur, falls H3 dann tatsächlich trägt — jetzt verfrüht.

---

## Anhang — Datenherkunft

Alle Zahlen aus `battery_k3/*.jsonl` (JSONL: `meta` + 20 `snap` @ 250 + `final`; Zähler kumulativ,
außer `births` = Σ `children` lebender Agenten, daher **nicht** über Snaps summierbar). Instrumentierung
per `scripts/m1_pilot.py`-Wrapper (`deaths` = exakter `remove_dead`-Zähler seit Review I-1; `respawns`
= `emergency_respawn`-Aufrufe × `respawn_count`=2 Agenten). Code-Belege: `agents/brain.py`
(ROLLOUT_HORIZON 62, GAMMA_V2 115, finalize_terminal 1006), `agents/agent.py` (v2-Reward ~1556,
Reproduktions-Gates 76–91), `simulation.py` (remove_dead 313/327, MIN_POPULATION/RESPAWN_COUNT 38/39).
