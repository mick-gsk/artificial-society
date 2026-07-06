# v2-Kalorien-Fix — Design (HOT-Core-Fix am v2-Einnahmen-Gap)

Branch `feat/infra-m1-pilot`. Baut auf der Vor-Diagnose
[`2026-07-06-v2-sterblichkeit-diagnose.md`](./2026-07-06-v2-sterblichkeit-diagnose.md)
auf (Nahrungsquellen-Mechanik-Gap, nicht Kosten-Kalibrierung). Dieses Dokument
liefert die quantitative Energiebilanz, verortet den Gap präzise und spezifiziert
**eine** minimale, prinzipientreue Mechanik-Änderung mit Testplan.

READ-ONLY-Analyse; kein Code geändert. Alle Zeilenangaben gegen den HEAD des
Worktrees (`artificial_society/agents/agent.py`,
`artificial_society/environment/physics/actions.py`,
`artificial_society/simulation.py`).

---

## TL;DR

- **Der Gap sitzt auf der EINNAHMEN­seite, spezifisch bei der Fleisch-/Aas-Linie.**
  Der Metabolismus (Ausgaben) ist zwischen v1 und v2 bit-identisch (gemeinsamer
  Code **vor** der `physics_v2`-Verzweigung, `agent.py:1452-1476`) — erneut
  verifiziert.
- **Pflanzen-Kalorien** fließen in v2 problemlos: über den einfachen
  `action["forage"]`-Skalar → `_forage` → `plant_food`-Zellpool
  (`agent.py:1574` → `740-748`). Diesen Kanal feuert selbst eine Zufallspolicy
  zuverlässig (kontinuierliche Aktions-Dim > 0). Er ist **nicht** der Engpass.
- **Fleisch-Kalorien** fließen in v2 **ausschließlich** über das verkörperte
  `do_eat`-Verb auf ein Kadaver-**Objekt** (`agent.py:1113` → `actions.py:348`).
  Der v2-Zweig von `_forage` überspringt die Fleisch-/Aas-Zellpools bewusst
  (Spec B6, `agent.py:735`). Das verkörperte Verb verlangt präzises
  Slot-Targeting **und** Ko-Lokation mit dem Kadaver — eine Hürde, die die
  `nolearn`-Zufallspolicy praktisch nie überwindet. **Der v1-Tod→Nahrung-Kreislauf
  ist damit in v2 de facto geschlossen für alles außer einem gelernten Butcher-Skill.**
- **Wichtige Korrektur/Verfeinerung der Vor-Diagnose:** Der Kadaver ist bereits
  *ohne Werkzeug* essbar (`carcass.nutrition = 0.14 > MIN_NUTRITION_EDIBLE = 0.02`;
  `do_eat` akzeptiert ein Objekt am Boden der eigenen Zelle). Die Barriere ist
  also **nicht** eine erzwungene Schneiden→Essen-Kette, sondern die **geringe
  Zugänglichkeit** des verkörperten Eat-Verbs für eine ungelernte Policy. Das
  verschiebt den Fix von „Kadaver essbar machen" (schon der Fall) hin zu
  „Kadaver-Biomasse an den bereits feuernden `forage`-Skalar andocken".
- **Empfehlung: Option A (verfeinert)** — im v2-Zweig von `_forage`
  (`agent.py:735`) zusätzlich zum Pflanzenbiss opportunistisch einen **rohen
  Handbiss** aus einem ko-lokierten Kadaver-Objekt nehmen (geringerer Ertrag als
  die Werkzeug-Zerlege-Linie). Stellt die v1-Zugänglichkeit des
  Tod→Nahrung-Kreislaufs her, **ohne** den Brain-Contract/Aktionsraum zu berühren,
  massenerhaltend, deterministisch.
- **Optionen B (Zellpool-Rückbau, bricht B6/Erhaltung) und C (reine Kalibrierung,
  durch Daten widerlegt) werden verworfen** (Begründung unten).

---

## 1. Energiebilanz eines typischen v2-Agenten pro Tick

### 1.1 Ausgaben (v1 = v2, bit-identisch)

Gemeinsamer Pfad **vor** der `physics_v2`-Verzweigung (`agent.py:1452-1476`):

```py
move_cost = 0.5 * mods.get("move_cost_mult",1.0) * stage["move_cost_mult"]
                * structure_mods.get("cold_factor",1.0)          # 1452
self.energy = max(0.0, self.energy - move_cost)                   # 1459
...
if self.energy <= 0:  self.health -= 1.5                          # 1471
if self.health <= 0:  self.alive = False                          # 1476  (Hungertod)
```

- Grundumsatz ≈ **0,5 Energie/Tick** (Baseline-Multiplikatoren = 1,0).
- Schwangerschaft: −0,03/Tick (`agent.py:615`).
- Reproduktion: einmalig `REPRODUCTION_COST = 20` (`agent.py:78,963`).
- Verkörperte Arbeit (`do_cut`/`do_strike`) kostet Energie über
  `WORK_SIM_ENERGY_PER_JOULE` **nur wenn** das Verb feuert; `do_eat` kostet
  **keine** Arbeitsenergie. Fatigue (`body.py`) ist ein separater Kanal, kein
  `energy`.

⇒ Netto-Ausgaben in Ruhe/Bewegung: **~0,5/Tick**, im Hunger-Spiral (energy ≤ 0)
zusätzlich −1,5 Health/Tick bis zum Tod. **Kein v2-spezifischer Mehrverbrauch.**

### 1.2 Einnahmen — Pflanzen (v1 = v2)

`_forage` v2/Herbivoren-Zweig (`agent.py:740-748`):

```py
take = min(plant_available, PLANT_ENERGY * eff)   # PLANT_ENERGY = 30
self.energy = min(MAX_ENERGY, self.energy + take) # 1:1-Transfer aus dem Zellpool
```

Ein erfolgreicher Pflanzenbiss = bis zu **30 Energie** = **~60 Ticks** Grundkosten.
Begrenzender Faktor: `plant_food`-Regrow/Ceiling der Zelle, **nicht** die Kosten.
Zugänglichkeit: hoch (skalarer `action["forage"] > 0` genügt).

### 1.3 Einnahmen — Fleisch (v1 ≫ v2)

- **v1** (diet ≥ 0, ~50 % der Agenten, `genetics.py`): über **denselben**
  `forage`-Skalar zusätzlich `carcasses`-Zellpool (`MEAT_ENERGY = 45`/Biss) und
  `meat_food`-Pool. Der Tod kreditiert `add_carcass(world, *pos, CORPSE_ENERGY=36)`
  in den Zellpool (`simulation.py:332`) — **sofort per `forage`-Skalar essbar**.
  ⇒ geschlossener, floor-zugänglicher Tod→Nahrung-Kreislauf; ~1,5× breitere
  Kalorienbasis.
- **v2:** Fleisch existiert nur als Kadaver-**Objekt** (`_spawn_carcass`,
  `simulation.py:335-349`), Masse = `BODY_MASS_DEFAULT_KG = 70 kg`. Zugänglich
  **nur** via `do_eat`-Verb (`actions.py:348`):

  ```
  bite       = min(BITE_MASS_KG=0.3, mass)
  energy/biss = nutrition(0.14) · KCAL_PER_KG_PER_NUTRITION(4000)
                · bite(0.3) · SIM_ENERGY_PER_KCAL(0.032) ≈ 5,38 Energie/Biss
  Gesamt-Kadaver (70 kg) ≈ 0.14·4000·70·0.032 ≈ 1254 Energie  (≈ 5× MAX_ENERGY)
  ```

  Die Biomasse ist also **enorm** (ein Kadaver ≈ 233 Bisse ≈ 1254 Energie), aber
  sie erfordert 233 sequentielle, präzise auf das Kadaver-Objekt gerichtete
  Eat-Verben. Für eine ungelernte Policy (nolearn) ist der erwartete Ertrag
  daraus ≈ 0.

### 1.4 Netto-Fazit

| | Ausgaben/Tick | Pflanzen-Einnahme (forage-Skalar) | Fleisch-Einnahme |
|---|---|---|---|
| v1 (diet ≥ 0) | ~0,5 | bis 30/Biss, hoch zugänglich | Zellpool 36–45/Biss, **hoch** zugänglich (forage-Skalar) |
| v2 (alle) | ~0,5 | bis 30/Biss, hoch zugänglich | Kadaver-Objekt ~1254 total, **niedrig** zugänglich (Eat-Verb) |

**Der Gap ist einseitig auf der Einnahmen­seite und dort auf der
Fleisch-Zugänglichkeit.** Die Bilanz ist im Mittel nicht katastrophal negativ
(mean_energy ~55–68, siehe §3) — sie ist **zu knapp, um Geburten über den
Respawn-Boden zu heben**: v2 verliert die halbe, floor-zugängliche Kalorienbreite
von v1 und den geschlossenen Recyclingkreislauf.

---

## 2. Wo genau sitzt der Gap — Einnahmen, nicht Ausgaben

**Verifikation Ausgaben (Vor-Diagnose bestätigt):** `move_cost`,
`hydration_loss`, Hunger-/Health-Drain und Hungertod stehen alle **vor** der
`if self.physics_v2:`-Verzweigung (`agent.py:1452-1476` läuft, dann erst 1483 die
Verzweigung). Bit-identisch. **Kein Kosten-Gap.**

**Fehlender Kalorienstrom (quantitativ):** In v1 fließt aus jedem Tod
`CORPSE_ENERGY = 36` sofort-zugänglich in den Zellpool. In v2 fließt aus jedem
Tod ein 70-kg-Kadaver mit ~1254 potenzieller Energie, davon **de facto ~0
zugänglich** für nolearn. Bei den gemessenen ~15 Respawns + ~2,5 Geburten pro
250-Tick-Snapshot (≈ Turnover ~17 Tote/Snapshot; §3) entzieht v2 dem
zugänglichen Kreislauf pro Snapshot in der Größenordnung `17 × 36 ≈ 610`
Sofort-Energie, die in v1 rezirkulierte. Über 5000 Ticks (20 Snapshots) ist das
ein strukturell fehlender Sofort-Kalorienstrom, der die v1/v2-Differenz in
mean_energy (§3.1: 82 vs 65) und mean_age (250 vs 217) erklärt.

---

## 3. Kalibrierung an vorhandenen Sweep-Daten

Aggregiert über `sweep_nocrutch/`, `sweep_taxis/`, `sweep_taxfloor/` (nolearn,
Seeds 41–44, je 20 Snapshots à 250 Ticks). Mittel über alle Snapshots/Seeds:

| Condition | phys | taxis | pceil | minpop | pop | mean_energy | mean_age | births/snap | deaths/snap | respawns/snap | births_cum(max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_nocrutch | v1 | – | 3.4 | 2 | 4.2 | **82.1** | **250.0** | 2.20 | 0* | 14.6 | 34 |
| v2_nocrutch | v2 | – | 3.4 | 2 | 4.6 | 65.0 | 216.7 | 2.68 | 69.7 | 15.6 | 33 |
| v2_nocrutch_fat | v2 | – | **6.0** | 2 | 4.6 | 67.8 | 215.1 | 2.58 | 70.8 | 16.8 | 31 |
| taxfloor_off | v2 | false | 2 | 8 | 11.6 | 62.0 | 169.4 | 3.54 | 131.6 | 19.1 | 53 |
| taxfloor_on | v2 | true | 2 | 8 | 11.5 | 53.6 | 191.4 | 4.42 | 121.5 | 15.5 | 63 |
| taxis t1–t6 | v2 | mix | 2–3.4 | 2 | ~4.5 | 56–68 | 189–218 | 2.0–2.6 | 63–70 | 14–16 | 25–33 |

\* v1 `deaths` liest baukonstruktiv 0 (Filter vor dem Zähler,
`simulation.py:626`) — Messartefakt, siehe Vor-Diagnose §1.

**Ablesungen, die das Design tragen:**

1. **Ausgaben-Kalibrierung ist nicht der Hebel.** `plant_ceiling_scale` 3.4 → 6.0
   bewegt mean_age praktisch **nicht** (216,7 → 215,1) und mean_energy kaum
   (65,0 → 67,8). Ebenso `min_food_per_capita` 6 → 30 (t1→t6: mean_age 217 → 218).
   ⇒ **Option C (reine Kalibrierung) ist durch die Daten widerlegt.** Deckt sich
   mit dem Auftrag („plant-ceiling bis 6.0 half nicht").
2. **Der Respawn-Boden dominiert die Demografie.** Respawns/Snapshot (~15–19)
   liegen weit über Geburten/Snapshot (~2–4). **Nicht die Geburten tragen die
   Population, sondern die `emergency_respawn`-Krücke.** Das maskiert den echten
   Gap (Vor-Diagnose §2.3) und definiert zugleich das Erfolgskriterium (§5).
3. **Der v1/v2-Unterschied ist real, aber gedämpft:** bei sonst identischer
   Konfiguration (nocrutch) hat v1 mean_energy 82 vs v2 65 und mean_age 250 vs
   217 — konsistent mit der fehlenden Fleisch-Zugänglichkeit, gedämpft durch den
   gemeinsamen Respawn-Boden.

---

## 4. Optionen — Abwägung

### Option A (EMPFOHLEN, verfeinert) — Kadaver-Objekt an den `forage`-Skalar andocken

**Mechanik:** Im v2-Zweig von `_forage` (`agent.py:735`, innerhalb
`if diet < 0 or self.physics_v2:`, **gegated auf `self.physics_v2`**) zusätzlich
zum Pflanzenbiss: liegt ein Kadaver-Objekt auf der Standzelle
(`world.objects` an `(x,y)`), nimm **einen rohen Handbiss** daraus (kein Werkzeug,
`nutrition = 0.14`). Umsetzung massenerhaltend über den bestehenden
`do_eat`-Pfad (`actions.py:348`), der Masse aus dem Objekt debitiert
(`ledger['eaten']`) und das Objekt bei Verzehr entfernt.

**Warum das der richtige Fix ist:**
- **Stellt die v1-Zugänglichkeit her:** dieselbe triviale `forage`-Aktion, die in
  v1 den Kadaver-Zellpool aß, greift jetzt in v2 auf die Kadaver-Biomasse zu →
  Tod→Nahrung-Kreislauf wieder geschlossen und floor-zugänglich.
- **§4b-treu (physikalisch plausibel, nicht gescriptet):** rohes Fleisch ist mit
  Händen essbar, nur ineffizient. Der Handbiss (0.14) ist ein **innater
  Grundtrieb** (wie Metabolismus/Taxis), die Werkzeug-Zerlege-Linie
  (cut → `raw_meat` nutrition **0.35** → eat, 2,5× dichter) bleibt ein **gelerntes**
  Capability mit klarem Ertragsgradienten. Kein Capability wird verschenkt.
- **Berührt den Brain-Contract NICHT (JA/NEIN → NEIN):** keine neue Aktions-Dim,
  keine Änderung an `act_v2`, an den Objekt-Slots oder den Zulässigkeits-Masken.
  Nutzt den bereits gesampelten `action["forage"]`-Skalar. ⇒ **Kein
  Off-Policy-Bias** (die Move-/Verb-Köpfe und ihr PPO-Buffer bleiben unberührt —
  im Gegensatz zu einem Eingriff in den Verb-/Target-Raum; die Taxis-Lehre, dass
  Verhaltens-Overrides des Move-Kopfs sorgfältig off-policy-neutral gehalten
  werden müssen, wird eingehalten, weil hier gar kein Kopf-Verhalten geändert wird).
- **Massenerhaltend:** Debit über `ledger['eaten']`, kein Minten, **kein
  Doppelzählen** — die Fleisch-Zellpools bleiben AUS (B6 intakt); der Kadaver ist
  weiterhin genau ein Objekt.
- **Deterministisch:** `_forage` ist RNG-frei; `do_eat` nimmt **kein** rng-Argument
  (anders als `do_strike`). Determinismus bleibt.

**Genaue Änderungsstelle:** `agent.py:735-748` (v2-Herbivoren-Zweig von
`_forage`). Neuer Block, gegated auf `self.physics_v2`, der `world.objects` an der
Standzelle nach einem `carcass`-Objekt (und dessen Schnitt-Remainder, ebenfalls
`kind=='carcass'`, sowie `raw_meat`-Fragmenten) abfragt und bei Fund einen
`do_eat`-Biss ausführt; `gain`/Energie 1:1 gutschreiben, `meat_eaten += 1` für die
Metrik. **HOT-Core-FLAG** an `agent.py:735` (Kernpfad, nicht harness-patchbar).

**Gating-Detail (wichtig für Golden):** Der umgebende Zweig
`if diet < 0 or self.physics_v2:` dient **auch** v1-Herbivoren (diet < 0,
`physics_v2 = False`). Der neue Kadaver-Hook **muss** auf `self.physics_v2`
gegated werden, sonst ändert er das v1-Herbivoren-Verhalten und macht das
v1-Golden rot. Mit dem Gate bleibt der v1-Pfad byte-identisch → **v1-Golden grün,
nur v2-Golden rot (erwartet).**

### Option B — Fleisch-Zellpool bei Tod rückbauen (verworfen)

`_spawn_carcass` (`simulation.py:335`) zusätzlich `add_carcass(..., CORPSE_ENERGY)`
in den Zellpool kreditieren und `agent.py:735` für diet ≥ 0 in v2 öffnen.
- **Bruch mit B6** („ein Pfad je Kalorienquelle"): Fleisch existiert dann doppelt
  (Objekt **und** Zellpool). Ohne Verzicht auf das Kadaver-Objekt **verletzt das
  die Massen-Erhaltung** (`phys_objects.py`-Ledger: `boden+hände+eaten+decayed ==
  spawned+from_carcass`) bzw. mintet Energie. Reparatur erfordert, das Objekt
  wegzulassen — dann bricht die Embodiment-/Zerlege-Linie (der learn-Pfad) ganz
  weg.
- Größerer Blast-Radius, mehr Golden-Bruch, konzeptioneller Rückschritt hinter
  die v2-Physik. **Verworfen.**

### Option C — reine Kalibrierung (verworfen)

Pflanzen-Ceiling hoch / Grundumsatz runter. **Durch §3.1 empirisch widerlegt**
(pceil 3.4 → 6.0 bewegt mean_age nicht). Ändert keine Mechanik, schließt den
Tod→Nahrung-Kreislauf nicht, adressiert die halbierte Kalorienbreite nicht.
**Verworfen.**

---

## 5. Testplan (harte Kriterien)

**(i) Energiebilanz-Unit (Mechanik korrekt & erhaltend).**
- v2-Agent auf Zelle mit Kadaver-Objekt (70 kg), `action["forage"]` feuert:
  `energy` steigt um ~5,38 (= 0.14·4000·0.3·0.032); Kadaver-`mass` sinkt um 0,3 kg;
  `layer.ledger['eaten']` steigt um 0,3. Massen-Invariante hält.
- v2-Agent auf reiner Pflanzenzelle (kein Kadaver): Verhalten unverändert
  (Regression gegen bestehendes Pflanzen-Foraging).
- v1-Herbivore (diet < 0, physics_v2 = False): **byte-identisch** zu vorher
  (Gate greift nicht) — Golden-Schutz.
- Determinismus: zwei Läufe gleicher Seed → identische Energie-Trajektorie.

**(ii) mean_age steigt deutlich.** m1_pilot v2 nolearn, Seeds 41–44, sonst wie
`v2_nocrutch`. Kriterium: mean_age steigt **materiell** über die Baseline
(~217) hinaus in Richtung mehrere hundert bis > 1000 Ticks; Richtung/Größenordnung
zählt, nicht ein exakter Wert.

**(iii) DEMOGRAFIE — der eigentliche Beweis.** Taxis ON + moderater Floor
(min_pop 8, wie `taxfloor_on`). Kriterium: über die **hintere Laufhälfte**
(ticks > 2500) fällt `respawns`/Snapshot von ~15 gegen **~0**, während
`births_cum` weiter steigt und `ids_seen_total`-Zuwachs aus Geburten (nicht
Respawns) stammt. **Geburten tragen die Population** = Selbsttragfähigkeit
erreicht. (Sekundär: `emergency_respawn`-Krücke kann testweise entfernt werden —
Vor-Diagnose Lever 3 — um zu prüfen, dass die Population **ohne** Krücke ≥ min_pop
hält.)

**(iv) learn/nolearn-Experiment intakt.** nolearn überlebt neu auf dem innaten
Gnaw-Floor (0.14). learn muss weiterhin eine **messbare Kante** zeigen:
`tool_cut_ratio > 0`, höhere mean_energy/mean_age als nolearn, weil die
Werkzeug-Zerlege-Linie (`raw_meat` 0.35) 2,5× dichter ist als der Handbiss.
Der Ertragsgradient bleibt erhalten → das A/B-Lernsignal wird nicht nivelliert.

**Golden:** v1 grün (Gate), v2 rot (erwartet, Energie-Trajektorien ändern sich)
→ v2-Golden nach Review neu backen.

---

## 6. Anhang — Datei:Zeile-Index

| Zweck | Stelle |
|---|---|
| v2-Zwang Pflanzen-only / Andockstelle Option A | `agent.py:735` (`if diet < 0 or self.physics_v2:`), Pflanzenbiss `740-748` |
| `_forage` Aufruf (v2, forage-Skalar) | `agent.py:1574-1585` |
| Verkörpertes Eat-Verb (einziger v2-Fleischkanal aktuell) | `agent.py:1113` → `actions.py:348` (`do_eat`) |
| do_eat-Energie/Erhaltung | `actions.py:362-383` (`bite`, `energy_gain`, `ledger['eaten']`) |
| Kadaver essbar ohne Werkzeug | `materials_v2.py:105-117` (`nutrition=0.14`), `MIN_NUTRITION_EDIBLE=0.02` (`actions.py:44`) |
| Metabolismus (bit-identisch v1=v2) | `agent.py:1452-1476` |
| Tod → Kadaver-Objekt (v2) vs Zellpool (v1) | `simulation.py:331` (`_spawn_carcass`) vs `332` (`add_carcass`) |
| Konstanten | `agent.py:69,76-79,121-123` · `actions.py:25,26,43,44` · `body.py:26` |
| Sweep-Daten (Kalibrierung §3) | `sweep_nocrutch/`, `sweep_taxis/`, `sweep_taxfloor/` (nolearn `*.jsonl`) |
