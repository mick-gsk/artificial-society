# Grund-Taxis — Design (angeborene symmetrische Nahrungs- & Partner-Taxis)

**Branch:** `feat/infra-m1-pilot` @ 7596f6e · **Status:** Design (READ-ONLY am Code) ·
**Autor:** Design-Analyst (Fable 5) · **Datum:** 2026-07-06

> **Owner-Entscheidung (bindend):** Die Population ist nicht selbsttragend
> (Malthus-Overshoot-Crash → Allee-Falle; Überlebende finden keine Partner, weil
> sich niemand *gezielt* bewegt). Lösung = **symmetrische angeborene Grund-Taxis**:
> gerichtete Grundfortbewegung zu Nahrung **UND** Partner als eingebauter Trieb
> (Analogie: Chemotaxis — angeboren, nicht gelernt). Symmetrisch, damit keine
> Asymmetrie zwischen den Bedürfnissen entsteht. §4b-konform, **weil** es ein
> Grundtrieb ist (wie Hunger/Metabolismus/Konzeption-bei-Ko-Lokation), **kein**
> gescriptetes Capability — Werkzeuge/Sprache/Bauen bleiben gelernt.

Dieses Dokument beantwortet nicht die Frage *ob* (entschieden), sondern *wie* man
den Trieb einbaut, **ohne den PPO-Buffer zu korrumpieren** und **ohne den
learn-vs-nolearn-Vergleich zu entkernen** — die zwei technischen Blocker, an denen
der Vorgänger (Option B, `2026-07-06-reproduktion-fix-design.md`) im Review
(`.superpowers/sdd/review-report-repro-design.md`) scheiterte.

---

## TL;DR

- **Empfohlene Architektur: A (Variante A2) — Taxis als angeborene Lokomotions-Mechanik, die die *Welt-Wirkung* des Move-Kopfs vollständig ersetzt.** Die Bewegung wird nicht mehr von der Policy gesteuert, sondern deterministisch vom Bedürfnis-Gradienten (Hunger→Nahrung, paarungsbereit→Partner) im Wahrnehmungsradius.
- **Berührt den Brain-Contract? NEIN.** Die Move-Dims 0/1 bleiben **physisch im Aktionstensor** (`ACTION_SIZE_V2 = 29` unverändert, `INPUT_SIZE` unverändert, Checkpoints laden weiter). Sie werden nur **verhaltens-inert** — die Welt reagiert nicht mehr auf sie.
- **Lern-Integrität: strukturell gewahrt, ohne `brain.py` anzufassen.** `store_transition_v2` speichert weiter die **gesampelte** Aktion + ihr **echtes** `log_prob` (beide bleiben on-policy, weil wir sie nie überschreiben). Weil die Move-Dims keinen kausalen Einfluss mehr auf `(reward, s')` haben, ist ihr PPO-Gradient bei θ=θ_old **beweisbar erwartungswert-null** → kein Off-Policy-/Credit-Assignment-Bias. Das ist der entscheidende Unterschied zu Option B: dort war `ausgeführte ≠ gespeicherte` Aktion (die Lüge); hier ist die ausgeführte Fortbewegung **gar keine Policy-Aktion**, also gibt es keine Lüge zurückzuschreiben.
- **HOT-Fläche: ~35 Zeilen, ausschließlich in `agent.py`, null Zeilen in `brain.py`.** 2 Konstanten + eine reine, RNG-freie Methode `innate_locomotion` + ein 2-Zeilen-Gate am Andockpunkt `agent.py:1443`.
- **Experiment-Validität:** Taxis regiert **nur** die Lokomotion (Dims 0/1). Die gemessene M1-Zielfähigkeit — **Werkzeug-/Verb-/Ziel-Emergenz** (Dims 2–5 Trigger, 7–28 Verben/Target/Tool) — bleibt **voll policy-gesteuert und in beiden Armen identisch**. Taxis substituiert also nicht die Werkzeug-Fähigkeit; es macht Überleben in beiden Armen zur *kontrollierten Konstante*, und der A/B-Kontrast wird auf Werkzeug-Emergenz gelesen (siehe §6).

---

## 1. Der v2-Bewegungs-/Aktionspfad im Code (Datei:Zeile)

**Aktionsraum (v2):** `brain.py:104` `ACTION_SIZE_V2 = 29`, kontinuierlich, tanh-gestauchte
Gaussian. `brain.py:105` `V1_HEAD_DIMS = 7`. Dim-Map (`brain.py:165-166`): **Dim 0 =
move_x, Dim 1 = move_y**, 2 = forage, 3 = cooperate, 4 = attack, 5 = build, 6 = (7.
v1-Kopf); 7–28 = Verben/Effort/Target-Query/Tool-Query.

**Ist Bewegung eine PPO-Aktion? JA** — Dims 0/1 sind zwei kontinuierliche
Aktionsdimensionen, gesampelt in `brain.act_v2` (`brain.py:385-455`), das
`action_tensor` (Zeile 453) und das summierte `log_prob`
(`_continuous_log_prob`, `brain.py:332-339`, plus Kategorial-Terme 401/426/445) liefert.

**Tick-Fluss** in `Agent.update(self, world, agents, tick, …)` (`agent.py:1285-1295`;
`world` **und** `agents` sind hier im Scope — wichtig für den Hook):

| Zeile | Vorgang |
|-------|---------|
| `1362` | `features = self.local_features(world, agents)` |
| `1373` | `brain_step = self.brain.act_v2(features, hidden, view.feats, view.mask, admissible_masks(view))` |
| `1394` | `action_list = brain_step["action_list"]` |
| `1396-1397` | Dict-Bau: `action = {"move_x": action_list[0], "move_y": action_list[1], …}` |
| `1441` | `self._execute_embodied(world, brain_step, view)` (eine verkörperte Aktion/Tick) |
| **`1443`** | **`self.primitive_move(world, action)`** — führt Bewegung aus `action["move_x/move_y"]` aus |
| `1448+` | forage/cooperate/attack/build-Trigger (aus `action[...]`) |
| `1563` | `self.brain.store_transition_v2(brain_step, reward, …)` |

**`primitive_move`** (`agent.py:592-602`): schwellt `action["move_x"]` bei ±0.33 →
`dx ∈ {-1,0,+1}` (analog `dy`), klemmt an Weltränder, bewegt bei `passable`. (Klemmt
schon heute ohne `action_tensor`-Update — der Präzedenzfall, der aber selten &
physikalisch erzwungen ist.)

**`store_transition_v2`** (`brain.py:470-491`): legt in den Rollout-Buffer
`"action" = brain_step["action_tensor"].squeeze(0)` (Zeile 480, enthält Dims 0/1)
und `"log_prob" = brain_step["log_prob"].squeeze(0)` (Zeile 481). **Das sind die
gesampelten Werte.** `evaluate_actions_v2` (`brain.py:500-539`) rekonstruiert die
log-prob bitgleich über `_continuous_log_prob` → die PPO-Ratio nutzt genau dieses
gespeicherte `(a, log_prob)`-Paar.

**Der bestätigte Defekt von Option B (zur Erinnerung):** Ein Override von
`action["move_x/move_y"]` **vor** 1443 ändert die *ausgeführte* Bewegung, aber
`store_transition_v2` speichert weiter die *gesampelte* → Reward der Taxis-Bewegung
wird der gesampelten Random-Walk-Aktion zugeschrieben = Off-Policy-Bias. **Jede
Architektur hier muss diesen Mismatch strukturell ausschließen.**

---

## 2. Architektur-Bewertung (A / B / C) — Empfehlung A2

### Die Kernfrage
Wie fügt man gerichtete Grundbewegung ein, ohne dass `ausgeführte Aktion ≠
gespeicherte Aktion` wird?

### (A) Taxis als Körper-/Umgebungs-Mechanik VOR/NEBEN der Policy — **EMPFOHLEN (A2)**
Die Fortbewegung folgt dem Bedürfnis-Gradienten; die Policy steuert die **höheren**
Aktionen (Foraging-Trigger, Werkzeug, Interaktion). Zwei Sub-Varianten:

- **A1 (Dims aus dem Aktionsraum entfernen):** `ACTION_SIZE_V2 29→27`, alle
  v1-Kopf-Indizes/`V1_HEAD_DIMS`/Checkpoint-Layout/Golden brechen. **= teuerster
  Fall, Brain-Contract-Bruch. VERWORFEN.**
- **A2 (Dims physisch behalten, Welt ignoriert sie):** Die Move-Dims 0/1 werden
  weiter gesampelt & gespeichert (Contract **unangetastet**), aber die Welt reagiert
  nicht mehr auf sie — die Lokomotion kommt aus der Taxis. **← EMPFOHLEN.**

  **Warum A2 lern-integer ist (der Beweis):** Der Buffer speichert die *gesampelte*
  `(a₀,a₁)` mit ihrem *echten* π_old-`log_prob`. Weil `(a₀,a₁)` keinen Einfluss mehr
  auf `(r, s')` haben, ist der PPO-Policy-Gradient auf diese Dims
  `E[A·∇_θ log π_θ(a₀,a₁|s)]`. Da `A` nicht von `(a₀,a₁)` abhängt und
  `(a₀,a₁) ∼ π_old`, faktorisiert das zu
  `E_s[A(s)·E_{a∼π_old}[∇log π_θ(a|s)]] = E_s[A(s)·0] = 0` bei θ=θ_old (Score-Funktion
  hat Erwartungswert 0). → **Kein Off-Policy-Bias.** Der Move-Kopf wird schlicht ein
  *inerter* Kopf, dessen Gradient zur (Entropie-getriebenen) Prior-Verteilung driftet.
  Es gibt **nichts** in `brain.py` zurückzuschreiben, weil wir nie behaupten, die
  Taxis-Bewegung sei eine Policy-Aktion gewesen.

  **Minor-Caveat (dokumentiert, kein Korrektheitsfehler):** Das summierte `log_prob`
  enthält weiter die 2 inerten Dims; die PPO-Ratio trägt dadurch etwas Zusatz-Varianz
  von 2 von 29 Dims. Bei θ≈θ_old sind diese Terme ≈0, unverzerrt, und **in beiden
  A/B-Armen identisch**. Sie aus dem `log_prob` zu entfernen hieße `act_v2` **und**
  `evaluate_actions_v2` konsistent zu ändern (Divergenz-Risiko, Contract-Nähe) —
  **nicht den Aufwand wert.** Dims drinlassen.

### (B) Deterministischer Bias, der die ausgeführte Bewegung ändert + zurückschreibt
Override `action["move_x/move_y"]` → ausführen → `brain_step["action_tensor"][0:2]`
überschreiben **und** `log_prob` via `_continuous_log_prob` für die erzwungene Aktion
neu berechnen (in/um `store_transition_v2`, `brain.py:470-481`).
- **Nachteile:** (1) berührt `brain.py`-Interna (frozen contract), mehr HOT-Fläche als
  A2. (2) Selbst mit korrektem Rückschreiben ist es **subtil off-policy**: die
  ausgeführte Aktion stammt aus einer *deterministischen* Verhaltensregel, nicht aus
  π_old — ihr als π_old-`log_prob` das Policy-log_prob unterzuschieben lehrt die Policy,
  die Taxis zu **imitieren** (verzerrtes Behavior-Cloning auf dem Move-Kopf). Das
  korrumpiert genau das Navigations-Lernsignal. A2 vermeidet das, weil dort die
  Bewegung *keine* Aktion ist. **Nachrangig zu A2.**

### (C) Taxis nur bei aktivem Bedürfnis, sonst Policy-Bewegung
Auf Bedürfnis-Ticks Taxis-Override (ausgeführt=Taxis, gespeichert=gesampelt →
**re-introduziert den 3b-Off-Policy-Defekt auf genau diesen Ticks**), auf Rest-Ticks
Policy (sauber). Die Override-Ticks feuern „systematisch & zustandskorreliert" im
gemessenen Regime (Review-Einwand). Erfordert per-Tick Exclude-oder-Writeback →
strikt **mehr** Fläche und Fragilität als A2, wo der Move-Kopf **uniform** auf **allen**
Ticks inert ist (keine per-Tick-Sonderbehandlung). **VERWORFEN.**

### Kriterien-Matrix

| Kriterium | A2 (empf.) | B | C |
|---|---|---|---|
| §4b-Treue (ein Schritt/Tick, kein Teleport) | ✅ | ✅ | ✅ |
| Lern-Integrität (kein Off-Policy-Bias) | ✅ strukturell | ⚠️ nur mit Writeback, lehrt Taxis-Imitation | ❌ per-Tick-Defekt |
| Brain-Contract unangetastet | ✅ | ❌ (store_transition/log_prob) | ❌ |
| Minimale HOT-Fläche | ✅ ~35 Z., nur agent.py | ➖ agent.py + brain.py | ➖➖ am meisten |
| Determinismus (RNG zentral) | ✅ RNG-frei | ✅ | ✅ |

---

## 3. Gradient-Wahrnehmung (woran die Taxis sich orientiert)

Deterministisch, **keine `random.*`-Draws** (sonst verschiebt sich der zentrale
seed-getriebene RNG-Stream und alles Downstream perturbiert — harte Auflage, im Merge
zu verifizieren).

**Bedürfnis-Signale**
- `hungry := self.energy < TAXIS_HUNGER_THRESHOLD`
- `mate_ready := self.can_reproduce()` (`agent.py:500-508`; gilt für **beide**
  Geschlechter → sie konvergieren aufeinander; Konzeption bleibt am bestehenden
  female-driven ±5-Gate in `_try_reproduce`, `agent.py:817`).
- **Priorität (Survival-first, deterministisch):** `if hungry → Nahrungs-Taxis;
  elif mate_ready → Partner-Taxis; else → Ruhe (kein Schritt)`. (`can_reproduce`
  verlangt `energy ≥ REPRODUCTION_ENERGY` > Hungerschwelle → Konflikt selten.)

**Nahrungs-Gradient**
- Scan-Box: `FOOD_SEEK_RADIUS` (Empfehlung: `int(self.genes["sense_radius"])`,
  Floor 1 — der Agent bewegt sich zu Gradienten, die er tatsächlich *sensen* kann;
  §4b-treu, kein globaler Radius).
- Ziel: Zelle mit max `cell["food"]` (>0) in der Box (Aggregat `food`, wie es
  `local_features` liest, `agent.py:537`; `_local_food_per_capita` nutzt dasselbe Feld).
- **Deterministische Tie-Breaks:** höchstes `food`; bei Gleichstand geringste
  Chebyshev-Distanz; dann lexikografisch `(dx, dy)` bzw. `(x, y)`.
- Ist die beste Zelle die Standzelle → kein Schritt (bleiben & foragen).

**Partner-Gradient**
- Scan über `agents`-Liste (ordnungsstabil): `a is not self and a.alive and
  a.sex != self.sex and a.can_reproduce() and self.trust.get(a.id, 0.0) >= -0.2`
  (spiegelt die Kompatibilität aus `_try_reproduce`, `agent.py:830`) innerhalb
  `MATE_SEEK_RADIUS` (Empfehlung: **8** — etwas über der ±5-Konzeptionsbox, damit
  Agenten die letzte Lücke aktiv schließen).
- **Deterministische Tie-Breaks:** geringste Chebyshev-Distanz; dann kleinste
  `a.id`.

**Schritt** (identisch zur `primitive_move`-Physik): `dx = sign(target_x - x)`,
`dy = sign(target_y - y)` (je `∈ {-1,0,+1}`), Klemmen an `world.width/height`,
Bewegung nur bei `cell.get("passable", True)`. Genau **ein** Schritt/Tick.

---

## 4. Spezifikation der empfohlenen Architektur (exakte Änderungsstellen)

**Alle Änderungen in `artificial_society/agents/agent.py`. Null Änderung an
`brain.py`. Kein Contract-Bruch (`INPUT_SIZE`/`ACTION_SIZE_V2`/Checkpoints
unangetastet).**

### 4.1 Konstanten — nach `agent.py:89` (bei den Reproduktions-Konstanten)
```python
# Angeborene Grund-Taxis (Chemotaxis-Analogon): symmetrischer Nahrungs- &
# Partnertrieb. Rein deterministisch, KEIN random.* (RNG-Stream unverändert).
TAXIS_HUNGER_THRESHOLD = 120.0   # < ~0.5*MAX_ENERGY(240): "hungrig" → Nahrungs-Taxis
MATE_SEEK_RADIUS = 8             # > ±5-Konzeptionsbox; Partner-Wahrnehmungsradius
# FOOD_SEEK_RADIUS = self.genes["sense_radius"] (Floor 1), zur Laufzeit
```
(Schwellen als Modul-Konstanten → im Sweep justierbar; im meta-Record des Piloten
mitloggen, analog zu `min_food_per_capita`.)

### 4.2 Neue Methode `innate_locomotion` — nach `primitive_move` (~`agent.py:602`)
Reine, RNG-freie Methode: berechnet Bedürfnis → Gradient-Ziel → **ein** Schritt;
nutzt dieselbe Klemm-/`passable`-Logik wie `primitive_move`. Signatur
`innate_locomotion(self, world, agents)`. (Skizze; Implementierung durch core-lead.)
```python
def innate_locomotion(self, world, agents):
    if self.is_sleeping:
        return
    target = None
    if self.energy < TAXIS_HUNGER_THRESHOLD:
        target = self._nearest_food_cell(world)           # Tie: food↑, dist↓, (x,y)
    elif self.can_reproduce():
        target = self._nearest_compatible_mate(agents)    # Tie: dist↓, id↑
    if target is None:
        return                                            # gesättigt → Ruhe
    x, y = self.pos
    dx = (target[0] > x) - (target[0] < x)                # sign, ∈ {-1,0,1}
    dy = (target[1] > y) - (target[1] < y)
    nx = max(0, min(world.width - 1, x + dx))
    ny = max(0, min(world.height - 1, y + dy))
    if world.get_cell(nx, ny).get("passable", True):
        self.pos = (nx, ny)
```
`_nearest_food_cell` / `_nearest_compatible_mate` als kleine reine Helfer (§3-Regeln).

### 4.3 Andock — `agent.py:1443` (Gate statt Ersatz, hält v1-Pfad unverändert)
```python
# vorher: self.primitive_move(world, action)
if self.physics_v2:
    self.innate_locomotion(world, agents)   # angeborener Trieb ersetzt Move-Kopf
else:
    self.primitive_move(world, action)      # v1-Pfad unverändert (Golden grün)
```
Der v2-Pilot ist der einzige betroffene Pfad; v1-Golden bleibt grün.

### 4.4 Was **nicht** angefasst wird
- `brain.py` (`act_v2`, `store_transition_v2`, `_continuous_log_prob`,
  `evaluate_actions_v2`): **unverändert.** Der Buffer speichert weiter das gesampelte
  `(action_tensor, log_prob)` → on-policy per Konstruktion.
- Der `action`-Dict-Bau (1396) & die höheren Trigger (1448+): unverändert — Forage/
  Tool/Build/Cooperate/Attack bleiben policy-gesteuert.

### 4.5 Contract-Risiko — explizit
**KEIN Brain-Aktionsraum-Contract berührt.** Die teure Variante (A1: Dims entfernen →
`ACTION_SIZE_V2`-Bruch) ist bewusst vermieden; die buffer-korrekte Bias-Variante (B)
ist ebenfalls vermieden, weil A2 die Off-Policy-Frage **strukturell** löst statt per
Writeback. Restrisiko = Verhaltensänderung (Golden), nicht Struktur.

### 4.6 Determinismus
`innate_locomotion` + Helfer: reine Funktionen von Positionen, `agent.id`, Chebyshev,
`trust.get`-Lookup, `world.get_cell`. **Kein `random.*`.** Iteration über
ordnungsstabile `agents`-Liste. → Zentraler RNG-Stream unverschoben, seed-für-seed
reproduzierbar. **Merge-Auflage: Grep-Verifikation „kein `random`/`np.random` in der
Methode".**

### 4.7 Golden
Verhaltensänderung im v2-Pfad → `tests/.../test_regression_golden.py` **erwartet rot**
(egal auf Experiment-Branch). Regen nach Merge unter core-lead. v1-Golden bleibt grün
(Gate). Frozen-contract-FLAG an core-lead: `agent.py` wird an 3 Stellen editiert
(Konstanten, neue Methode, Gate) — **core-lead besitzt den Eingriff.**

---

## 5. Testplan (harte Kriterien)

### (i) Unit — Taxis bewegt richtig (`tests/agents/test_grund_taxis.py`)
- **`test_food_taxis_step`**: hungriger Agent, **eine** Food-Zelle NO in `sense_radius`,
  sonst leer → nach `innate_locomotion` sinkt die Chebyshev-Distanz zur Food-Zelle
  strikt (bis auf der Zelle); `dx,dy` zeigen zur Zelle. Deterministisch.
- **`test_mate_taxis_step`**: paarungsbereiter Agent + **ein** kompatibler fruchtbarer
  Gegen-Sex-Partner in `MATE_SEEK_RADIUS` → Distanz **nicht-steigend** und strikt
  sinkend, solange >0, bis ≤ ±5-Box. *(Robuste Assertion — NICHT „strikt monoton für
  immer", weil beidseitige Annäherung an der Grenze oszillieren kann; der Review-Einwand
  gegen brüchige Monotonie-Asserts.)*
- **`test_sated_rests`**: gesättigt (`energy ≥ TAXIS_HUNGER_THRESHOLD`) & nicht
  paarungsbereit → `pos` unverändert.
- **`test_priority_food_over_mate`**: hungrig **und** paarungsbereit → bewegt sich zur
  Nahrung.
- **`test_determinism_no_rng`**: `random.getstate()`/`np.random`-State vor==nach einem
  Batch `innate_locomotion` (kein Draw); zweiter Lauf identische Trajektorie.

### (ii) Lern-Integrität — Buffer bleibt on-policy (der 3b-Guard)
- **`test_buffer_records_sampled_move`**: v2-Tick konstruieren, in dem die Taxis den
  Agenten in eine **andere** Richtung bewegt als die gesampelten Move-Dims. Assert:
  die gespeicherte Transition trägt `action[0:2] == brain_step["action_tensor"][0:2]`
  (die **gesampelten** Werte, **nicht** die Taxis-Richtung) **und**
  `log_prob == brain_step["log_prob"]`. → Beweist: ausgeführte Bewegung wird **nicht**
  als Aktion verbucht; kein `ausgeführt≠gespeichert`-Mismatch (der Option-B-Defekt).
- **`test_move_dims_have_no_world_effect`**: zwei Ticks mit identischem Weltzustand aber
  gegensätzlichen gesampelten Move-Dims → identische resultierende `pos` (Lokomotion
  hängt nur von der Taxis ab). Belegt die Inertheit, auf der der Unbiased-Beweis (§2/A2)
  ruht.

### (iii) Demografie — der eigentliche Beweis (ohne Respawn-Krücke)
Über `scripts/m1_pilot.py`, physics v2, Taxis ON, **`--min-pop 2`** (Respawn-Krücke
praktisch aus), **5000 Ticks**, **≥4 Seeds (Empfehlung ≥8**, Review: fluktuations-
getriebene Allee-Dips werden von 4 Seeds leicht verfehlt; **min-über-Seeds berichten**).
Harte Kriterien im Fenster **Tick 2500–5000**:
- `respawns ≈ 0` (kein einziger `emergency_respawn`-Aufruf 2500–5000 → selbsttragend,
  nicht respawn-gefüttert; `respawns` = Aufruf-Zähler im snap/final-Record).
- `pop mean ≥ 12`, `pop min ≥ 6`.
- `Δbirths_cum ≥ +20`.
- Kontrast: Vor-Fix-Baseline (Taxis OFF) verfehlt dies / braucht Respawns.

### (iv) Experiment intakt — learn vs. nolearn unterscheidbar
Beide Arme **mit Taxis ON** (A1 `learn` vs. A2 `nolearn`/`_freeze_learning`), ≥4 Seeds:
- **Werkzeug-Emergenz-Metrik** (z. B. Tool-Acquisition-Rate / distinkte Verben /
  Build-Events) in `learn` **signifikant > `nolearn`**, während **Pop/Überleben in
  beiden Armen vergleichbar** ist (Taxis unterschreibt Überleben by design in beiden).
- **Konfundierungs-Check:** `nolearn+Taxis` erreicht die von `learn` erreichte
  Werkzeug-Schwelle **nicht** → belegt: Taxis substituiert **nicht** die gemessene
  Zielfähigkeit; der A/B-Kontrast wird sauber auf Werkzeug-Emergenz (nicht auf rohes
  Überleben) gelesen.

---

## 6. Prinzipien- & Validitäts-Note (für core-lead)

- **§4b:** Taxis ist ein *Grundtrieb* (Bewegung folgt Gradient wie Metabolismus dem
  Substrat), **symmetrisch** über Nahrung & Partner — keine Asymmetrie, kein einzelner
  privilegierter Zielbewegungstrieb. Navigation-zu-Nahrung wird damit bewusst zur
  angeborenen Mechanik (Owner-Entscheidung), analog zum bereits eingebauten
  Konzeptions-Gate bei Ko-Lokation. Werkzeuge/Sprache/Bauen bleiben **gelernt**.
- **M1-Ziel-Verschiebung (explizit dokumentieren):** Die abhängige Variable des Piloten
  wandert von „trägt Lernen das *Überleben*?" zu „bringt Lernen *Werkzeug-/höhere
  Emergenz* auf einer per Grundtrieb lebensfähigen Population hervor?". Damit ist der
  Review-Einwand („Taxis macht auch den OFF-Arm selbsttragend → Headline kontaminiert")
  aufgelöst: Überleben ist jetzt *kontrollierte Konstante* in beiden Armen, und Test
  (iv) misst den Trieb-unabhängigen Lern-Beitrag.
- **Temporär/begrenzt kennzeichnen:** Grund-Taxis im Code-Kommentar als *eingebauter
  Trieb* markieren, dessen Kalibrierung (Schwellen/Radien) im meta-Record geloggt und
  im Sweep offen ist.

## 7. Datei:Zeile-Index (Änderungsstellen)

| Stelle | Datei:Zeile | Änderung |
|---|---|---|
| Konstanten | `agent.py:~89` (nach `REPRODUCTION_MIN_FOOD_PER_CAPITA`) | `TAXIS_HUNGER_THRESHOLD`, `MATE_SEEK_RADIUS` |
| Neue Methode | `agent.py:~602` (nach `primitive_move`) | `innate_locomotion` + `_nearest_food_cell`/`_nearest_compatible_mate` |
| Andock/Gate | `agent.py:1443` | `if self.physics_v2: innate_locomotion(world, agents) else: primitive_move(world, action)` |
| **Unverändert** | `brain.py:385-455` (`act_v2`), `470-491` (`store_transition_v2`), `332-339` (`_continuous_log_prob`) | **keine** — Buffer bleibt on-policy per Konstruktion |
| Golden | `tests/.../test_regression_golden.py` | v2 erwartet rot (Regen post-merge, core-lead); v1 grün |
| Tests neu | `tests/agents/test_grund_taxis.py` | §5 (i)+(ii) |
| Harness | `scripts/m1_pilot.py` | Schwellen in meta-Record loggen; §5 (iii)+(iv) fahren |

---

*Referenzen: `2026-07-06-reproduktion-fix-design.md` (verworfene Option B),
`.superpowers/sdd/review-report-repro-design.md` (Review, der B verwarf — Befunde 3b
Lern-Integrität & Experiment-Validität hier strukturell adressiert).*
