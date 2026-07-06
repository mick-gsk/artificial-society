# Reproduktion-Fix — Design (Allee-Falle brechen, prinzipientreu)

**Datum:** 2026-07-06 · **Branch:** `feat/infra-m1-pilot` @ `1a24a7e` · **Status:** Design, READ-ONLY-Analyse
**Kontext:** `docs/superpowers/specs/2026-07-06-v2-sterblichkeit-diagnose.md`, `docs/superpowers/specs/2026-07-05-demografie-diagnose.md`
**Grundsatz (nicht verhandelbar, `docs/roadmap.md` §4b):** Fähigkeiten werden gelernt, nicht gescriptet. Grundmechaniken (essen, altern, fortpflanzen) sind legitime eingebaute Regeln. Reparatur muss physikalisch plausibel bleiben: **kein Teleport-Paaren, kein globaler Paarungsradius, keine Agenten aus dem Nichts.** Ein Paarungs-Bewegungstrieb ist als Grundmechanik zulässig (analog Hunger/Foraging-Trieb).

---

## TL;DR

Die Population trägt sich nie selbst, weil zwei Dinge zusammenwirken: (1) ein Malthusianischer Overshoot-Crash reißt die synchrone Gründerkohorte von ~60 auf ~4–5 Agenten, und (2) — der **bindende** Blocker — die wenigen Überlebenden finden sich auf 1200 Zellen im hartkodierten ±5-Kasten (`agent.py:830-831`) praktisch nie zum Paaren wieder (**Allee-Falle**). Energie, Cooldown und das Food-Gate binden nachweislich **nicht**.

**Empfehlung: Option B — Paarungs-Bewegungstrieb.** Fortpflanzungsbereite Agenten gehen einen Schritt in Richtung des nächsten wahrgenommenen kompatiblen Partners; die Konzeption bleibt an die lokale Ko-Lokation (±5-Kasten) gebunden. Das ist der einzige prinzipientreue Fix, der die Allee-Falle **an der Wurzel** bricht (er macht sogar eine 1m+1f-Restpopulation erholungsfähig), ohne action-at-a-distance-Konzeption. Option A (Radius vergrößern) ist im Dilemma: moderat = wirkungslos, groß genug zum Wirken = der verbotene globale Paarungsradius.

---

## 1. Der Reproduktionspfad im Code

### 1.1 Konstanten (`agent.py:76-93`)
```
REPRODUCTION_ENERGY = 60.0            # :76
REPRODUCTION_COST   = 20.0            # :77  (Mutter −20, Vater −10)
REPRODUCTION_COOLDOWN = 100           # :78
REPRODUCTION_SENSE_RADIUS = 2         # :88  (nur fürs Food-per-capita-Gate, NICHT Paarung)
REPRODUCTION_MIN_FOOD_PER_CAPITA = 6.0# :89
MIN_REPRODUCTION_AGE = 60             # :91
GESTATION_TIME = 40                   # :92
ELDER_AGE = 3500 / AGE_LIMIT = 5000   # :93 ff.
```

### 1.2 `can_reproduce()` (`agent.py:500-508`)
Fruchtbar ⇔ `alive ∧ age≥60 ∧ age<3500 ∧ energy≥60 ∧ reproduction_cooldown≤0 ∧ ¬pregnant`.

### 1.3 `_try_reproduce(world, agents)` (`agent.py:817-858`) — der Engpass
1. Nur **Weibchen** initiieren: `if not self.can_reproduce() or self.sex != "f": return` (`:818`).
2. Food-Gate: `_local_food_per_capita < 6.0 → return` (`:820`).
3. **Partnersuche (die Allee-Klemme):** Liste fruchtbarer Männchen mit
   ```py
   abs(a.pos[0] - x) <= 5 and abs(a.pos[1] - y) <= 5   # :830-831  HARTKODIERTES LITERAL
   and self.trust.get(a.id, 0.0) >= -0.2               # :832
   ```
   Kein Männchen im 11×11-Kasten → `return None` (`:836`). **Bei Fund: Konzeption sofort und ortsungebunden** — es findet **keine** Bewegung/Annäherung statt; die Mutter empfängt von einem bis zu Chebyshev-5 entfernten Männchen (`:846-857`: `pregnant=True`, `gestation=max(20, GESTATION_TIME/eff)`, beide `cooldown=100`).
4. `progress_pregnancy()` (`agent.py:579-593`): `gestation-=1`, `energy-=0.03`; bei `gestation≤0` liefert es `stored_child_genes` zurück → `Simulation.spawn_child_from_parent` (der EINZIGE Geburtsort, disjunkt von Respawn).

### 1.4 WARUM das bei niedriger Dichte scheitert (Allee)
Der ±5-Kasten deckt 121 von 1200 Zellen (~10 %) ab. Bei pop 4–5, je ~½ m/f, verstreut, sitzen fruchtbare Partner fast nie gleichzeitig im selben 11×11-Fenster → Geburten ≈ 0 (Beleg Demografie-Diagnose Befund 3: b_cum +8 über 4500 Ticks). Die Steady-State-Lebensdauer (~360–400 Ticks) ist **länger** als der Zyklus `60+40+100=200`, und me der Überlebenden liegt bei 100–140 (≫ 60) — Zeit und Energie binden also **nicht**. Es bindet allein die **räumliche Auffindbarkeit**. `emergency_respawn` (`simulation.py:305-311`) setzt bei `pop<MIN_POPULATION` (`:639`) `RESPAWN_COUNT` Agenten an **zufälligen** `world.random_land_position()`-Spots mit age 0 nach — die zuerst `MIN_REPRODUCTION_AGE=60` erreichen müssen und meist vorher einzeln verhungern → respawns feuern endlos, `respawns→0` unerreichbar.

---

## 2. Overshoot vs. Allee — was ist das Kernproblem?

- **Overshoot-Crash (Regime 1, `agent.py:1354-1360` Hungertod):** 30 synchrone Gründer brüten bis Tick 250 auf ~60; die Kohorte überzieht die von der Pflanzendecke gesetzte Tragfähigkeit, `plant_food` wird schneller befressen als es nachwächst, kollektive Verhungerung → pop 60→~5 (Tick 250–750).
- **Wirkt die dichteabhängige Fruchtbarkeit (`REPRODUCTION_MIN_FOOD_PER_CAPITA`, c9a2525)?** Sie ist die richtige logistische Rückkopplung, aber sie **verhindert den Crash nicht**, weil (a) Agenten bis MAX_ENERGY 240 horten und über das Personal-Energy-Gate ohnehin fruchtbar bleiben, (b) das Food-per-capita-Gate nur einen 5×5-Nahbereich (`REPRODUCTION_SENSE_RADIUS=2`) misst und in der Wachstumsphase satter Zellen erfüllt ist. Die eigentliche Tragfähigkeit hängt an `SCARCITY_CEILING_FACTOR` (`resources.py:45`) und ist **nicht harness-exponiert** (Befund 2).
- **Ist der Crash das Kernproblem?** **Nein.** Der Crash setzt nur die niedrige Dichte; der **bindende** Blocker für Selbsttragfähigkeit ist die **fehlende Erholung danach** = Allee. Selbst ohne Crash würden 4–5 Agenten auf 1200 Zellen nicht brüten. Und da der Overshoot (bei synchroner Kohorte) real passiert, muss der Fix die **Erholung aus niedriger Dichte** herstellen — genau das leistet ein Bewegungstrieb, und zwar bis hinunter zu pop 2 (1m+1f finden sich). Overshoot-Dämpfung (Decke/Fertilität) ist komplementär, aber (i) außerhalb der Reproduktionsmechanik und (ii) nicht harness-exponiert → hier **out of scope**.

---

## 3. Bewegung im v2-Pfad: gerichtet oder random?

**Zentraler Befund für die Größe des Eingriffs:** Es gibt **keine** gescriptete Ziel-Bewegung. Der einzige Mover ist `primitive_move(world, action)` (`agent.py:594-602`), aufgerufen im Act-Loop bei **`agent.py:1443`**. Richtung kommt allein aus `action["move_x"/"move_y"]` (Schwelle ±0.33 → dx,dy ∈ {−1,0,1}), also **aus der gelernten Brain-Policy**. Bei Zufalls-Init ist das effektiv Random-Walk.

Wichtig: **Auch Foraging ist nicht gerichtet gescriptet.** `_forage` (`agent.py:604-671`) frisst nur die **aktuelle** Zelle; zur Nahrung *hinlaufen* lernt das Brain. Die Task-Prämisse „analog zum bestehenden Nahrungs-Foraging" heißt also: es gibt keinen bestehenden Nahrungs-Lauf-Trieb, an den man andockt — ein Partnersuche-**Bewegungs**-Trieb ist ein **neuer** kleiner Mechanismus (kein bloßer Parameter). Das ist der Grund, warum Option B ein echter (wenn auch chirurgischer) Zusatz ist und nicht nur ein Knopf.

---

## 4. Optionsabwägung

### Option A — Paarungsradius (Konzeptionskasten) moderat vergrößern
`abs(...)<=5` als `MATE_SEARCH_RADIUS` extrahieren und z. B. 5→12 setzen.
- **Pro:** ~2 Zeilen; call-time patchbar; deterministisch; kein Bewegungs-Override; gelernte Lokomotion unangetastet.
- **Contra (disqualifizierend):** Die Konzeption bleibt **action-at-a-distance**. Auf 40×30 deckt Radius 12 ~625/1200 ≈ 52 % der Karte — die Mutter empfängt von einem Männchen am halben Kartenrand, **ohne dass sich jemand bewegt**. Das **ist** der laut Grundsatz verbotene „globale Paarungsradius". Und: um pop 4–5 zuverlässig zu paaren, müsste der Radius nahezu global werden. Moderat (z. B. 8) → Allee bei noch niedrigerer Dichte ungebrochen („verschiebt das Problem nur"). **A ist entweder wirkungslos oder unprinzipiell.**

### Option B — Paarungs-Bewegungstrieb (EMPFOHLEN)
Fruchtbare Agenten, die einen fruchtbaren, kompatiblen Partner der Gegen-Sex innerhalb einer **Wahrnehmungsreichweite** spüren, gehen einen Schritt auf ihn zu. Konzeption bleibt an den bestehenden ±5-Ko-Lokationskasten gebunden.
- **Pro:** Prinzipientreu — Wahrnehmung (einen entfernten Partner *spüren*) ist physikalisch zulässig; Konzeption bleibt **lokal**; die Bewegung schließt die Lücke über mehrere Ticks. Bricht die Allee **an der Wurzel**: sogar pop 2 (1m+1f) wird erholungsfähig — genau die Bedingung für „selbsttragend ohne Krücke". Deterministisch machbar (kein rng).
- **Contra:** Neuer Helfer (~15 Z.) + Andock-Edit im Act-Loop (~4 Z.) + 1–2 Konstanten. Berührt das **frozen** `agent.py` an **zwei** Stellen. Überschreibt die gelernte Lokomotion in fruchtbaren+partner-sichtigen Ticks (Minderheit des Lebens). → **FLAG an core-lead** (siehe §5.4/§5.5).

### Option C — Energie/Cooldown justieren
Diagnose widerlegt Bindung: me 100–140 ≫ 60; Lebensdauer ≫ Zyklus. Ändert nichts an der **Auffindbarkeit**. **Als Standalone verworfen.**

**Entscheidung:** **B**, mit der billigen Konstanten-Extraktion aus A gebündelt (macht den Konzeptionskasten sauber sweepbar). B ist robust, wo A prinzipiell scheitert.

---

## 5. Empfohlene Änderung — exakte Stellen

> **Scope-Warnung (Task-Vorgabe):** Der Fix ist ~20–25 Zeilen, berührt aber das **frozen `agent.py`** an **zwei** Stellen (neue Methode + Act-Loop-Andock) plus Konstanten — mehr als die triviale 2-Zeilen-Radius-Anhebung. Das ist **bewusst und unvermeidbar**: die einzige Ein-Stellen-Alternative (Option A) verletzt den Grundsatz. Der Eingriff bleibt chirurgisch, aber **core-lead muss ihn besitzen** (frozen contract).

### 5.1 Konstanten — nach `agent.py:89`
```py
MATE_SEARCH_RADIUS = 5    # Konzeptions-Ko-Lokation (ersetzt das Literal in _try_reproduce; bleibt 5)
MATE_SEEK_RADIUS   = 12   # WAHRNEHMUNG für den Fortpflanzungstrieb (nur Spüren, NIE Konzeption)
```

### 5.2 `_try_reproduce` entklammern — `agent.py:830-831`
`abs(a.pos[0]-x) <= 5 and abs(a.pos[1]-y) <= 5` → `... <= MATE_SEARCH_RADIUS ...`. Rein mechanisch identisch (Wert bleibt 5), macht den Konzeptionskasten benannt/sweepbar. **Keine** Verhaltensänderung durch diese Zeile allein.

### 5.3 Neuer Helfer (neue Methode in `Agent`, z. B. nach `primitive_move`, ~`agent.py:602`)
```py
def _seek_mate_step(self, agents):
    """Fortpflanzungstrieb: ein Schritt Richtung nächstem fruchtbarem, kompatiblem
    Partner der Gegen-Sex innerhalb MATE_SEEK_RADIUS. DETERMINISTISCH (kein rng):
    nächster per Chebyshev-Distanz, Ties per agent.id. Gibt (mx,my) im
    ±0.33-Aktionsraum oder None zurück. Konzeption bleibt an ±MATE_SEARCH_RADIUS
    Ko-Lokation in _try_reproduce gebunden — dieser Trieb bringt Partner NUR
    zusammen, er empfängt nie auf Distanz."""
    x, y = self.pos
    best, best_key = None, None
    for a in agents:
        if a is self or not a.alive or a.sex == self.sex:
            continue
        if not a.can_reproduce() or self.trust.get(a.id, 0.0) < -0.2:
            continue
        dx, dy = a.pos[0] - x, a.pos[1] - y
        cheb = max(abs(dx), abs(dy))
        if cheb == 0 or cheb > MATE_SEEK_RADIUS:
            continue
        key = (cheb, a.id)
        if best_key is None or key < best_key:
            best_key, best = key, (dx, dy)
    if best is None:
        return None
    dx, dy = best
    mx = 1.0 if dx > 0 else -1.0 if dx < 0 else 0.0
    my = 1.0 if dy > 0 else -1.0 if dy < 0 else 0.0
    return (mx, my)
```
Beide Sexe suchen (Gegen-Sex-Filter) → schnellere Konvergenz; das Weibchen führt zusätzlich `_try_reproduce` aus. Reihenfolge im Loop ist günstig: Move bei `:1443`, Reproduktion bei `:1483` → nach dem Annäherungsschritt kann die Konzeption **im selben Tick** greifen, sobald ±5 erreicht ist.

### 5.4 Andock im Act-Loop — **vor** `agent.py:1443` (`self.primitive_move(world, action)`)
```py
if stage.get("can_reproduce", True) and self.can_reproduce():
    _md = self._seek_mate_step(agents)
    if _md is not None:
        action["move_x"], action["move_y"] = _md
self.primitive_move(world, action)
```
Der Trieb überschreibt **nur** die Move-Komponenten (nicht forage/attack/build), und nur wenn ein Partner tatsächlich gespürt wird (seltenes, gebundenes Fenster).

### 5.5 Prinzipien- & Lern-Integritäts-Note für core-lead
- **§4b-Rechtfertigung:** Der Trieb ist eine **Grundmechanik der Fortpflanzung** (wie Metabolismus/Altern), **kein** gelerntes Capability. Wahrnehmung eines entfernten Partners ist zulässig; **konzipiert wird nie auf Distanz** — das ±5-Ko-Lokations-Gate bleibt.
- **Executed ≠ intended:** Das Override erzeugt eine Abweichung zwischen gewählter und ausgeführter Move-Aktion. Präzedenz existiert bereits: `primitive_move` klemmt Moves gegen Unpassierbarkeit/Grenzen (`agent.py:598-602`). Core-lead sollte prüfen, ob die Brain-Trainings-Transition die **ausgeführte** Aktion sieht (empfohlen) — sonst ist es ein Umwelt-Effekt gleicher Art wie die Wand-Klemme. Optionale Abschwächung: Trieb nur greifen lassen, wenn die Policy selbst still steht (`|move_x|<0.33 ∧ |move_y|<0.33`), dann stört er die gelernte Lokomotion minimal.

### 5.6 Determinismus (rng zentral!)
`_seek_mate_step` verbraucht **kein** `random.*` → die zentrale, seed-getriebene RNG-Stream-Reihenfolge bleibt **unverändert**; der Lauf ist seed-für-seed reproduzierbar. **Harte Auflage an die Implementierung:** im Helfer **niemals** `random.*` aufrufen (das würde den geteilten Stream verschieben und alles Downstream perturbieren). Tie-Break per `agent.id`, Distanz per Chebyshev — beides deterministisch.

### 5.7 Golden
**Erwartet ROT.** Sobald fruchtbare Agenten Partner spüren, ändern sich Bewegungstrajektorien → `tests/test_regression_golden.py` (gegen `tests/golden_trajectory.json`) divergiert. Auf diesem Experiment-Branch **vorab akzeptiert**; core-lead regeneriert das Golden nach Merge. Hier egal.

---

## 6. Testplan

### (i) Liveness-Unit-Test — `tests/agents/test_mate_seeking.py`
Vorlage: `tests/agents/test_density_dependent_fertility.py` (`_sim`, `_make_pair`, `_flood_food`).
- **test_seek_step_points_toward_mate:** Fruchtbares f bei (2,2), fruchtbares m bei (8,5), sonst leer. `f._seek_mate_step([f,m])` == `(1.0, 1.0)` (Vorzeichen von dx,dy). Symmetrisch für m.
- **test_seek_none_without_partner:** Nur f (kein kompatibler Partner) → `None`. Auch: Partner außerhalb `MATE_SEEK_RADIUS` → `None`; unfruchtbarer Partner (age<60) → `None`; `trust < -0.2` → `None`.
- **test_converge_and_conceive:** f und m auf ~10 Zellen Abstand (grid 15×12), ample food, alle anderen entfernt; N≈15 Ticks `sim.step()`; assert Chebyshev-Distanz fällt monoton auf ≤ `MATE_SEARCH_RADIUS` **und** danach `mother.pregnant` (bzw. ein Geburts-Event). Ohne den Trieb (Baseline) bleiben sie getrennt und werden nie schwanger.
- **test_determinism:** zwei `Simulation(seed=S)` mit identischer Sequenz → bitgleiche Positions-/pregnant-Trajektorie (verankert §5.6).

### (ii) Demografie-Nachweis (entscheidend) — ohne Respawn-Krücke, selbsttragend über 5000 Ticks
Krücke praktisch aus über `--min-pop 2` (Respawn feuert nur bei pop < 2, d. h. pop ≤ 1) — vgl. `sweep_nocrutch/`. `--min-pop`/`--respawn-count` patchen `simulation.MIN_POPULATION/RESPAWN_COUNT` call-time (`scripts/m1_pilot.py:615-616`, CLI `:780-790`).

```bash
for s in 1 2 3 4; do
  python scripts/m1_pilot.py --exp nolearn --seed $s --ticks 5000 \
    --min-pop 2 --respawn-count 2 --grid-w 40 --grid-h 30 --pop 30 \
    --out sweep_repro
done
```
Pfad-unabhängige Zähler pro Snapshot: `respawns` (`m1_pilot.py:659-666`), `births_cum` (`:684-692`), `pop`.

**Erfolgskriterien (Mittel über 4 Seeds, `--min-pop 2`):**
1. **Selbsttragend, nicht krücken-gehalten:** `respawns`-Aufrufe über den ganzen Lauf ≈ 0 (Ziel 0; hart: ≤ 2 Einzelbatches, und **keine** in Tick 2500–5000). Äquivalent: `respawnte Agenten ≪ births_cum`.
2. **Population hält über der Allee-Schwelle:** nach dem initialen Crash `pop` durchgehend **> 2**, `min(pop)` (Tick ≥1000) **≥ 6**, `mean(pop)` (Tick 1000–5000) **≥ 12**. Kein Einfrieren bei 2.
3. **Geburten tragen die Kurve:** `births_cum` wächst in der zweiten Hälfte stetig — Δ`births_cum`(Tick 2500→5000) **≥ +20** (vs. Diagnose-Baseline +8 über 4500 Ticks). Zufluss ≈ Abfluss über Geburten, nicht Respawns.
4. **Kontrast zur Vor-Fix-Version:** dieselbe Kommandozeile auf `1a24a7e` kollabiert auf pop ≤ 2 / respawn-getragen (respawns feuern durchgehend). Der Fix dreht 4→erfüllt.

**Sekundär (Nicht-Regression):** ganze Suite `pytest -q` grün außer dem erwartet roten `test_regression_golden.py` (§5.7). `tests/agents/test_density_dependent_fertility.py` bleibt grün (Konstanten-Rename ist wertidentisch).

---

## 7. Datei:Zeile-Index (Änderungsstellen)
- `agent.py:89` (+2) — Konstanten `MATE_SEARCH_RADIUS=5`, `MATE_SEEK_RADIUS=12`
- `agent.py:830-831` — Literal `<= 5` → `<= MATE_SEARCH_RADIUS` (wertidentisch)
- `agent.py:~602` (neu) — Methode `_seek_mate_step(self, agents)` (~18 Z., deterministisch, kein rng)
- `agent.py:1443` (davor, +4 Z.) — Andock: Move-Override wenn `can_reproduce()` und Partner gespürt
- **Frozen-contract-FLAG:** 2 Editierstellen + 1 neue Methode in `agent.py` → core-lead-Ownership; Golden erwartet rot (regen nach Merge)
