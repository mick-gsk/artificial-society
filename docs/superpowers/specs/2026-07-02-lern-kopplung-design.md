# Lern-Kopplung (Schnitt 1, Bau-Schritt 3) — Design

**Datum:** 2026-07-02
**Status:** Rev. 2 — überarbeitet nach 4-Linsen-Agenten-Review (Architektur, Physik, RL, Tests); Re-Review läuft
**Eltern-Spec:** `2026-07-02-realphysik-emergenz-schnitt1-design.md` (§5 Gehirn-Kopplung, §6 Bau-Reihenfolge Schritt 3, §10 offene Punkte)
**Basis:** main @ 9c7b6f0 (Physik-v2-Kern + Embodiment v1 + Welt-Vektorisierung Tier 1+2)

## Bindende Prinzipien (aus Eltern-Spec, gelten für jeden Abschnitt)

1. **Gelernt, nie gewährt:** Keine Fähigkeit wird geschenkt oder getriggert. Die Policy wählt
   WAS sie greift/schlägt/schneidet — Sampling, nie Argmax, nie Skript.
2. **Realitäts-Gate:** Jede neue Konstante ist real geankert (Eintrag in `calibration.py`,
   CI-Test erzwingt Tabelle + Doc-Sync). Alles Gebaute wäre in der Realität so möglich.
3. **Erhaltung:** Masse und Energie werden exakt erhalten. Insbesondere: Tode münzen Energie
   entweder in v1-Pools ODER in v2-Objekte — nie beides, und nie zusätzlich als Direkt-Loot.
4. **Nur Überleben + Neugier:** Im v2-Modus keinerlei Shaping-Boni. Belohnung =
   physiologische Homöostase + Tod + intrinsische Neugier (0.3-gewichtet).
5. **Golden bleibt grün bei Flag aus:** `physics_v2=False` (Default) ändert kein Verhalten.
6. **GPU-Roadmap-kompatibel:** Objekt-Schicht bleibt sparse (Registry, nicht in Zell-Arrays),
   kollidiert nicht mit `cell_store.py`-SoA und der geplanten Torch-residenten Engine.
7. **Kein neuer Lamarckismus:** Dieses Design fügt keine Vererbungspfade für Gelerntes hinzu
   (Entfernung der bestehenden = Plan 4 Kultur-Korrektur).

## Team-Entscheidungen an der Grenze intrinsisch/Prior (aus dem RL-Review)

Zwei Review-Vorschläge berühren die Grenze zur User-Entscheidung „nur intrinsische Neugier,
kein Shaping". Beide wurden übernommen, mit dieser Begründung:

- **Count-based Novelty über Eigenschafts-Buckets** (C4): zahlt für nie erlebte Punkte im
  Eigenschaftsraum (1/√n), domänen-agnostisch, keine designer-gewählte Event-Menge —
  Standard-Form intrinsischer Motivation, kein Shaping. (Ein Bonus auf DiscoveryV2-*Events*
  wäre dagegen shaping-nah und wird NICHT eingebaut.)
- **Optimistischer Init-Bias der Verb-Köpfe** (C2): +0.2 auf die Verb-Means bei
  Initialisierung. Das ist ein struktureller Prior (Manipulations-Babbling wie beim
  Kleinkind), keine Belohnung — konform mit der User-Entscheidung „genetischer Prior".

---

## Abschnitt A — Gesamtarchitektur & Plan-Schnitt

**v2-Modus als additive Schicht.** `Simulation(physics_v2=True)` (Default `False`). Die
bestehende Ökologie (Pflanzen-Regrowth, Wasser, Wetter, Jahreszeiten, Tag/Nacht, Krankheit)
läuft unverändert weiter. NEU kommt die **Objekt-Schicht** dazu: eine sparse Registry
physischer `PhysObject`s an Grid-Positionen, getrennt von den Zell-Arrays. Jeder Agent
bekommt im v2-Modus `Body` + `Hands` (aus `environment/physics/body.py`) — **bereits in
Plan 3a, mit Default-`strength=0.5`**; das `strength`-Gen kommt in 3b. Damit ist 3a
end-to-end ohne Gehirn-Änderung lauffähig und testbar.

**Integration folgt den bestehenden Mustern:** Der Pro-Tick-Anteil der Objekt-Schicht
(Verwesung, Spawning) hängt sich als registriertes System ein
(`register_system("physics_v2_objects", ..., tick=...)`, Muster `systems/registry.py` /
`systems/_builtins.py`) — kein verstreutes `if physics_v2:` in der Tick-Schleife. Das Flag
selbst wird `Simulation`-Kwarg und Agent-Attribut (gesetzt bei `spawn_*`), gebraucht nur an
den Umschaltpunkten: Todes-Routing, Belohnungsmodus, Wahrnehmung/Aktion.

**Neue/geänderte Dateien:**

| Datei | Verantwortung |
|---|---|
| `artificial_society/environment/phys_objects.py` (neu) | `ObjectLayer`: Registry Position→Objekte, Spawning, Verwesung, Massen-Ledger, per-Welt `DiscoveryV2` |
| `artificial_society/environment/physics/actions.py` (neu) | Verkörperte Aktions-Mechanik: `do_grasp/do_release/do_strike/do_cut/do_eat` — bindet Body/Hands/Prozesse/ObjectLayer, ohne Gehirn-Wissen; eigene Kalibrierungs-Konstanten-Liste `CALIBRATED_ACTION_PARAMS` |
| `artificial_society/environment/physics/calibration.py` | neues kind `"spawn"` in `VALID_KINDS`; Gate-Anbindung für `actions.py`/`phys_objects.py`-Konstanten |
| `artificial_society/agents/perception_v2.py` (neu) | Objekt-Slots: Feature-Extraktion, Maskierung, **Slot-Identitäts-Tracking über Ticks** (id-basiert, für das Causal-Model-Target) |
| `artificial_society/agents/genetics.py` | neues Gen `strength` (16.; `GENE_RANGES`-Eintrag, Mutations-σ in der 0.012-Klasse wie `plasticity`) |
| `artificial_society/agents/brain.py` (Hot) | Attention-Set-Encoder, neue Aktionsköpfe inkl. kategorialer Slot-Auswahl, PPO-Anpassungen (γ, Minibatches, KL-Stop, log_std-Floor), Causal-Model-Anbindung; Planner-Pfad im v2-Modus deaktiviert |
| `artificial_society/agents/agent.py` (Hot) | v2-Obs-Bau, v2-Aktions-Mapping, v2-Belohnungsmodus (neuer physiologischer Term, Terminal-Transition beim Tod), Body/Hands-Besitz, Loot-Abschaltung |
| `simulation.py` | Flag-Kwarg + Checkpoint-Payload, Kadaver-Umschaltpunkt in `remove_dead()`, Hamilton-Rewards im v2-Modus aus |
| `artificial_society/systems/causal_model.py` (Neuimplementierung, Ideen aus `archive/`) | Verteilungs-Vorwärtsprädiktor über Objekt-Eigenschaften; epistemische Neugier |

**Aufräumen im Zuge von 3a:** das komplett unverdrahtete Alt-Modul
`systems/world_objects.py` (toter Vorläufer, referenziert nicht-existente Methoden) wird
nach `archive/` verschoben, um Namens-/Konzept-Kollision mit der neuen ObjectLayer
auszuschließen.

**Plan-Schnitt (zwei Pläne unter diesem einen Design):**

- **Plan 3a — Welt & verkörperte Aktionen:** ObjectLayer, Spawning, Kadaver-Umleitung,
  Body/Hands-Besitz (Default-strength), `actions.py` inkl. aller Exploit-Fixes,
  Energie-Kopplung Essen/Arbeit, Erhaltungs-Tests. Vollständig testbar ohne
  Gehirn-Änderung (geskriptete Test-Treiber rufen `do_*` direkt).
- **Plan 3b — Gehirn:** Wahrnehmung (Attention über Objekt-Slots), Aktionsköpfe
  (Verben + Queries + kategoriale Slot-Auswahl + Effort), Belohnungs-Umbau, Causal-Model,
  `strength`-Gen, Checkpoint-Versionierung, PPO-Anpassungen. Baut auf 3a auf; erst nach
  3a-Merge.

---

## Abschnitt B — Welt-Schicht (Plan 3a)

### B1. ObjectLayer

```python
class ObjectLayer:
    def __init__(self, width, height, rng): ...
    def add(self, obj: PhysObject, pos: tuple[int,int]) -> None   # validiert props ∈ [0,1], Masse > 0
    def remove(self, obj: PhysObject) -> None                     # Identität, nicht Wert
    def objects_at(self, pos) -> list[PhysObject]
    def objects_near(self, pos, radius) -> list[tuple[PhysObject, tuple[int,int]]]  # Chebyshev
    def total_mass(self) -> float                                  # NUR Boden; Ledger s.u.
    self.discovery: DiscoveryV2                                    # PRO WELT (Instanz)
    self.ledger: dict                                              # spawned/from_carcass/eaten/decayed
```

- Interne Struktur: `dict[pos, list[PhysObject]]` + Rückwärts-Map `id(obj)→pos`. Sparse,
  O(1) je Zugriff. **Explizit: die ObjectLayer lebt NICHT in `world.obj[y][x]` und schreibt
  keine Zell-Arrays** — sie ist ein Schwester-Attribut `world.objects` neben `world.F/S`
  (Integrationspunkt `World.__init__` nach `build_arrays()`, Migration über das bestehende
  `ensure_array_storage()`-Muster). Damit bleibt die GPU-Garantie (Prinzip 6) intakt.
- `add()` validiert zur Laufzeit alle 13 Props ∈ [0,1] und `mass_kg > 0`.
- **Ledger-Vollständigkeit:** Die Erhaltungs-Invariante (D1) zählt Bodenmasse **plus die
  Hände aller lebenden Agenten**. Stirbt ein Agent, fallen gehaltene Objekte per
  `do_release` an seine Position (kein Leck). Es gibt kein Culling; Verwesung fließt
  explizit in `ledger["decayed"]`.
- `DiscoveryV2` wird pro Welt instanziert (`ObjectLayer.discovery`). Das ungenutzte
  Modul-Singleton `DISCOVERY_V2` (`discovery.py:52`, Export in `physics/__init__.py`) wird
  entfernt — es hat keine Konsumenten und lädt zu geteiltem Zustand zwischen Welten ein.

### B2. Spawning (Biome-gebunden, kalibrierte Massen)

Beim Welt-Aufbau (`physics_v2=True`) und mit langsamer Regeneration (Poisson-Rate pro
Biom-Zelle und Tick, so dass Vorkommen über einen Pilot-Lauf nicht versiegen und nicht
fluten):

| Material | Biome | Masse je Objekt (uniform) | Realer Anker |
|---|---|---|---|
| `granite` | mountain | 0.5–8 kg | Lesesteine/Geröll |
| `flint` | mountain (Knollen), grassland (selten, Kiesel) | 0.3–4 kg | Feuerstein-Knollen in Kreide/Schotter |
| `dry_wood` | forest | 0.5–6 kg | Totholz-Äste |
| `plant_fiber` | grassland, swamp | 0.05–0.4 kg | Gras-/Bastbündel |
| `clay_moist` | swamp, Ufer (Nachbarzelle zu water) | 0.5–5 kg | Ufer-Lehm |

**Nicht** gespawnt werden `berries`/`raw_meat`/`water` als Objekte: Beeren/Pflanzennahrung
bleibt v1-Zell-Foraging (ein einziger Pfad für Pflanzenkalorien — kein Doppel-Futter),
Fleisch entsteht ausschließlich aus Kadavern (B3), Wasser bleibt Zell-Ressource.

**Kalibrierung:** Spawn-Dichten/-Raten bekommen Einträge unter dem **neuen kind
`"spawn"`** (`VALID_KINDS` wird erweitert; das bestehende kind `material` prüft strikt
gegen `MATERIALS_V2`-Namen und würde Spawn-Parameter als Orphans abweisen). Der
Realitäts-Gate-Test wird um die `"spawn"`-Quelle und um `CALIBRATED_ACTION_PARAMS` (B4/B5)
erweitert — neue Konstanten, die nicht registriert sind, machen das Gate rot.

### B3. Kadaver-Umleitung (Erhaltung!)

**Konkreter Umschaltpunkt:** `Simulation.remove_dead()` (`simulation.py:251–259`) ist heute
der einzige, ursachen-agnostische Todes-Aggregationspunkt und ruft
`add_carcass(self.world, *agent.pos, CORPSE_ENERGY)`. Im v2-Modus ersetzt genau dieser
Aufruf sich durch die Objekt-Erzeugung. Zusätzlich betroffen: der **Attack-Loot-Pfad**
(`agent.py:630–631`, `loot = target.energy * 0.3; self.energy += loot`) — im v2-Modus
**loot = 0**; Energie aus einem Kill gibt es ausschließlich über den Kadaver. Sonst würde
derselbe Tod doppelt gemünzt (Loot ist Energie ohne Massen-Gegenwert und für den Ledger
unsichtbar).

1. **v1-Pfad stillgelegt:** kein `add_carcass`-Credit auf Zell-Pools, kein Loot.
2. **Ein Kadaver-Objekt:** `make_object("carcass", mass_kg=body_mass)` an der
   Todesposition; `body_mass` aus dem Body des Verstorbenen (in 3a vorhanden).
   Gehaltene Objekte des Verstorbenen fallen zu Boden.
3. Kadaver sind schneidbar (v1-Anker: 25-kg-Kadaver untragbar → Zerteilen lohnt, bewiesen
   in `test_tool_pressure.py`); Fragmente erben Props vom Prozess, nie vom Label.
4. **Verwesung** (pro Tick, über das registrierte System): `mass *= (1 − DECAY_RATE)` mit
   `DECAY_RATE = ln(2) / (10 Tage · TICKS_PER_DAY 240) ≈ 2.89e-4` (Anker: forensische
   Taphonomie, Weichgewebe-Halbwertszeit temperiert ~7–14 Tage; `TICKS_PER_DAY` aus
   `daynight.py`). Verwesung koppelt zusätzlich intensiv: `nutrition *= (1 − DECAY_RATE)`
   und `toxicity += TOX_SPOILAGE_PER_TICK` (gekappt bei 0.6; Anker: Verderb — frisches
   Fleisch ist nahezu unbedenklich, verdorbenes gefährlich). Verweste Masse fließt in
   `ledger["decayed"]` (bilanzierter Abfluss).

Flag aus → exakt v1-Verhalten (Golden-Garantie + expliziter Verzweigungstest, D1).

### B4. Verkörperte Aktions-Mechanik (`actions.py`)

Eine Aktion pro Agent und Tick (real: eine Manipulation pro Zeitschritt). Alle Funktionen
nehmen `(agent_body, hands, object_layer, pos, ...)` und geben ein Ergebnis-Objekt zurück
(für Logging/Neugier/Tests). Kein Zugriff auf Gehirn oder Belohnung. Alle Konstanten
dieses Abschnitts stehen in `CALIBRATED_ACTION_PARAMS` und sind Gate-pflichtig.

- **`do_grasp(target)`** — Ziel: Boden-Objekt im Chebyshev-Radius 1. Scheitert (No-op) bei
  vollen Händen (`MAX_HELD=2`) oder Massen-Budget-Überschreitung.
- **`do_release(held)`** — legt Objekt an eigener Position ab.
- **`do_strike(striker_held, target, effort)`** — bindet **verpflichtend**
  `body.exert_strike(effort)` (Ermüdung steigt immer, auch bei wirkungslosem Schlag).
  Gelieferte Energie:
  `E = min(body.strike_energy_j(effort), 0.5 * striker.mass_kg * V_MAX_STRIKE**2)`
  mit `V_MAX_STRIKE = 14 m/s` (Anker: Hammerschlag-/Knapping-Endgeschwindigkeit 10–15 m/s).
  Nachgerechnet (Physik-Review): 0.05-kg-Kiesel liefert max ~4.9 J < jede Flint-Schwelle
  (Exploit zu); 0.5–2-kg-Schlagstein liefert die vollen 49–50 J > Flint-Schwellen bis
  4 kg — **Knapping bleibt möglich** (Regressionstest D2 hält beides fest).
  Ziel: Boden-Objekt an eigener Position ODER das andere gehaltene Objekt.
- **`do_cut(blade_held_or_none, target, effort)`** — Klinge aus der Hand oder bloße Hand
  (`BARE_HAND_*`-Konstanten aus Physik v2). **Klingen-Massen-Faktor** (beidseitig
  begrenzt):
  `blade_factor = clamp(sqrt(m/BLADE_MASS_REF), 0.2, 1.0) * clamp((BLADE_HANDLE_MAX*2 − m)/BLADE_HANDLE_MAX, 0.2, 1.0)`
  mit `BLADE_MASS_REF = 0.15 kg` (brauchbare Schlacht-Abschläge ≥ ~150 g) und
  `BLADE_HANDLE_MAX = 1.0 kg` (einhändig führbares Schneidwerkzeug ≤ ~1 kg — darüber ist
  es ein Hammer, kein Messer; Ethnographie Lithik). Ein 20-g-Splitter schneidet langsam,
  ein 4-kg-Rohbrocken ebenfalls (Faktor 0.2) — **das „8-kg-Skalpell" ist zu**, roher
  Großstein ersetzt kein geknapptes Werkzeug. Schneiden kostet Arbeit:
  `CUT_WORK_J = 15 + 35*effort` über denselben Ermüdungspfad (`FATIGUE_PER_JOULE`).
- **`do_eat(target)`** — Ziel gehalten oder am Boden. Ein Biss pro Tick:
  `bite = min(BITE_MASS_KG, target.mass_kg)` mit `BITE_MASS_KG = 0.3`. Wirkung s. B5.
  Objekte mit `nutrition ≤ 0.02` sind wirkungslos (Steinbeißen = No-op).
- **Überlast-Drop:** steigt `hands.held_mass` über die aktuelle `carry_capacity` (z. B.
  weil Ermüdung sie senkt), wird zu Tick-Beginn das schwerste gehaltene Objekt zwangsweise
  abgelegt, bis das Budget wieder eingehalten ist.

### B5. Energie-Kopplung (kcal ↔ Sim-Energie), real geankert

Nutrition-Dimension ist kalibriert als kcal/100 g ÷ 400 (raw_meat 0.35 ≙ 140 kcal/100 g).
Kopplung an die v1-Energieskala (MAX_ENERGY 240):

- `SIM_ENERGY_PER_KCAL = 0.032` — **Anker:** eine 1-kg-Fleischmahlzeit
  (0.35·4000 = 1400 kcal) ergibt ≈ **45 Sim-Energie ≙ v1 `MEAT_ENERGY`**. Die v1-Ökonomie
  bleibt erhalten. Der Gate-Test prüft das **Produkt** (nutrition-Konvention ×
  `SIM_ENERGY_PER_KCAL` ≙ MEAT_ENERGY ± 1), nicht die Einzelwerte — zwei Konstanten,
  eine Bilanz, kein stilles Driften.
- **Kadaver-Rekalibrierung:** `carcass.nutrition` wird von 0.30 auf **0.14** gesetzt
  (Anker: essbarer Anteil eines Wildkörpers ~40 % („dressed yield"); 0.35·0.40 = 0.14 —
  die intensive nutrition mittelt über Knochen/Haut/Innereien). Damit liefert ein
  70-kg-Kadaver ≈ 1250 Sim-Energie ≈ 5× MAX_ENERGY — ein wertvoller Gruppen-Fund statt
  42 Agenten-Tage aus einem Tod (Kalorienbomben-Fix). `carcass.toxicity` wird von 0.10
  auf **0.02** gesetzt (frisches Fleisch nahezu unbedenklich); Gefahr entsteht über
  Verwesung (B3.4).
- Essen: `energy += nutrition * 4000 * bite_kg * SIM_ENERGY_PER_KCAL`, Objektmasse sinkt
  um `bite_kg` (`ledger["eaten"]`).
- Toxizität: `health -= toxicity * bite_kg * TOX_DAMAGE_PER_KG` mit
  `TOX_DAMAGE_PER_KG = 20` (Anker: 0.3 kg stark toxischen Materials (0.8) ≈ 5 Health ≙
  spürbar, wiederholt tödlich; verdorbenes Fleisch bei tox 0.6 ≈ 3.6 Health/Biss).
- Mechanische Arbeit kostet metabolisch: `energy -= joules / MUSCLE_EFFICIENCY / 4184 *
  SIM_ENERGY_PER_KCAL * 1000` — Anker `MUSCLE_EFFICIENCY = 0.25` (Brutto-Wirkungsgrad
  Skelettmuskel 20–25 %). Bewusst klein (200 Schläge ≈ 0.3 Sim-Energie): der reale
  Begrenzer ist die **Ermüdung**, die Energie-Kopplung ist Erhaltungs-Buchhaltung.

Alle neuen Konstanten (`SIM_ENERGY_PER_KCAL`, `V_MAX_STRIKE`, `BLADE_MASS_REF`,
`BLADE_HANDLE_MAX`, `BITE_MASS_KG`, `CUT_WORK_J`-Parameter, `MUSCLE_EFFICIENCY`,
`TOX_DAMAGE_PER_KG`, `TOX_SPOILAGE_PER_TICK`, `DECAY_RATE`) stehen in
`CALIBRATED_ACTION_PARAMS` bzw. kind `spawn` und sind Gate-pflichtig (B2).

### B6. v1-Systeme im v2-Modus (pro System: Mechanik / Belohnung getrennt)

Der Code trennt heute NICHT zwischen „System läuft" und „System zahlt" — beides ist oft
dieselbe Zeile (`reward += ...`). Deshalb hier beide Spalten explizit, mit Fundstellen:

| System | Mechanik v2 | Reward v2 | Fundstelle Reward |
|---|---|---|---|
| Ökologie (growth, seasons, weather, daynight, herbs, resources, water) | AN | — | — |
| Zell-Foraging Pflanzen | AN | **AUS** (Event-Bonus 0.05·gained; Kalorien zahlen über Homöostase) | `agent.py` Forage-Pfad |
| Zell-Fleisch/Aas-Pools | **AUS** (ersetzt durch Kadaver-Objekte, B3) | — | `simulation.py:258`, `agent.py:518–563` |
| Attack/Kampf | AN (Kampf-Mechanik bleibt) | **AUS** (Loot `agent.py:630–631` = 0; Attack-Reward-Rückgabe `agent.py:1179` = 0) | s. links |
| v1-Erfindung (materials.py, DiscoveryRegistry, beide Trigger-Pfade, Goal-Stack-Planner) | **AUS** | **AUS** (+0.5/+1.0 entfallen mit) | `agent.py:1200–1216`, `agent.py:1142` |
| Kochen (+0.3) / Fermentation | **AUS** (Fermentation ist heute schon unverdrahtet) | **AUS** | `agent.py:1218–1222` |
| Bauen v1 (structures) | AN | unverändert (v1-Bau hat keinen eigenen Bonus-Term im v2-Streichumfang) | — |
| Territorium | AN (Claims) | **AUS** | `agent.py:1192` |
| Kooperation/Sharing | AN (Mechanik) | **AUS** (`COOP_SHARE_REWARD`) | `agent.py:670–687, 1175` |
| Sprach-Tokens | AN (Mechanik) | **AUS** (+0.1-Bonus) | `agent.py:225` |
| Hamilton-Verwandten-Rewards | — | **AUS** | `simulation.py:492` |
| Stämme, Krankheit, Endokrin, Handel (economy.py) | AN | etwaige Reward-Terme **AUS** | `agent.py:1224–1225` |

Das Sozialgefüge läuft also vollständig weiter (Kanal für kulturelle Weitergabe,
Meilenstein-Kriterium 2) — es wird nur nicht mehr extrinsisch bezahlt.

---

## Abschnitt C — Gehirn (Plan 3b)

### C1. Wahrnehmung: Objekt-Slots + Attention-Set-Encoder

**Slots:** K = 8 nächste Boden-Objekte im Chebyshev-Radius 4 + 2 Hand-Slots = **10 Slots**.
Leere Slots = Null-Vektor + Maske. Pro Slot 17 Features: 13 Props + `mass_kg/25` (gekappt
bei 1; Anker 25-kg-Kadaver) + `dx/4, dy/4` (relativ, normiert) + `held`-Flag.

**Slot-Identitäts-Tracking (`perception_v2.py`):** über `id(obj)` wird pro Agent eine
Map Vortick-Slot → Jetzt-Slot geführt. Sie dient NUR dem Causal-Model-Target (C4) und dem
Logging — das Gehirn selbst sieht weiterhin nur Eigenschaften, nie Identitäten.

**Set-Encoder (permutationsinvariant, in `brain.py`):**

```
embed_i = LayerNorm(Linear(17 → 32))(slot_i)              # geteilte Gewichte
q_att   = Linear(96 → 32)(h_prev)                          # Query aus GRU-Zustand des VORTICKS
attn    = masked_softmax(q_att·K^T / sqrt(32))             # 1 Head
obj_ctx = concat(attn·V, masked_max(V))                    # attn-Pool ⊕ Max-Pool = 64 dim
```

- **Alle Slots maskiert (kein Objekt in Sicht, leere Hände) ⇒ `obj_ctx = 0`** — kein NaN
  aus Softmax über leerer Menge (expliziter Test, D3).
- Max-Pool zusätzlich zum Attention-Pool: ein einzelner gewichteter Mittelwert kann „das
  nächste" und „das schärfste" nicht gleichzeitig repräsentieren; Kosten bei 10 Slots
  vernachlässigbar.
- GRU-Input: `concat(encoder(obs_57), obj_ctx)` → `GRUCell(128+64=192 → 96)`. Die 57
  v1-Obs-Dims bleiben unverändert.

Wahrnehmung ist reine Eigenschafts-Wahrnehmung: kein Objekt hat eine ID im Gehirn, ein nie
gesehenes Material ist ein neuer Punkt im Eigenschaftsraum (Generalisierung per
Konstruktion).

### C2. Aktionsraum: Verben + Queries im Embedding-Raum + kategoriale Slot-Auswahl

Neue Policy-Köpfe:

| Kopf | Dims | Typ | Semantik |
|---|---|---|---|
| Verben | 5 | kontinuierlich (Gaussian+tanh) | grasp, release, strike, cut, eat; Aktivierung > 0.5 = „will"; höchste gewinnt; eine verkörperte Aktion/Tick |
| Effort | 1 | kontinuierlich | [0,1] für strike/cut |
| Target-Query | 8 | kontinuierlich | Absichts-Vektor im **gelernten Embedding-Raum** |
| Tool-Query | 8 | kontinuierlich | dito, für Werkzeug aus der Hand |
| Slot-Auswahl | 2× kategorial | Kategorial-Sampling | s. u. |

**Auswahl-Mechanik (ersetzt argmin-L2 über rohe Props):** Für das aktive Verb werden die
zulässigen Slots bestimmt (grasp: Boden r≤1; strike/cut/eat-Ziel: Boden an eigener
Position ∪ gehalten; Tool: nur gehalten). Auswahl-Logits = `query · W_sel(embed_i) / √8`
über zulässige Slots; **Ziehung kategorial aus softmax(logits)**, die log-prob geht in die
Policy-log-prob ein. Das schafft (a) einen echten Gradientenpfad in Query UND
Slot-Embeddings (statt advantage-gewichtetem Rauschen durch eine Auflösung außerhalb des
Graphen), (b) erfüllt „Sampling, nie Argmax" wörtlich, und (c) derselbe Encoder formt
Wahrnehmung und Absicht. Leere zulässige Menge ⇒ No-op (keine log-prob-Beteiligung).

Gesamt: 7 v1-Köpfe + 22 neue kontinuierliche Dims + 2 kategoriale Auswahlen.
**`research_drive`** (Dim 7 der v1-Köpfe) ist schon heute tote Kopplung (Ausgabe wird
nirgends konsumiert); im v2-Modus wird die Dim zusätzlich **aus der log-prob-Summe
ausgenommen** (totes Rauschen im PPO-Ratio), der Kopf bleibt für Form-Kompatibilität.

**PPO-Anpassungen (v2-Modus):**

- `GAMMA 0.97 → 0.99` (Kredit-Horizont ~33 → ~100 Ticks; die Kette
  Knapping→Kadaver→Schneiden→Essen liegt real 50–500 Ticks auseinander).
- `PPO_EPOCHS 20 → 4`, Minibatches (4×32 aus dem 128er-Buffer), KL-Early-Stop bei 0.02 —
  20 Epochen auf einem einzigen Batch kollabieren σ und töten die Verben, bevor Kadaver
  und Klinge je koinzidieren (Verb-Extinktions-Analyse im RL-Review).
- **log_std-Floor −0.8 für die neuen Köpfe**, Entropy-Koeffizient pro Kopf-Gruppe
  (bestehende 0.004, neue 0.01).
- **Init-Bias +0.2 auf die Verb-Means** (Manipulations-Babbling-Prior, s. Grenzentscheid
  oben).
- Beim **Tod**: Terminal-Transition wird gespeichert (`done=True` erreicht den Buffer)
  und der Restbuffer mit Terminal geflusht und trainiert — heute stirbt ein Agent, bevor
  `store_transition` den Todes-Tick je sieht, und der Buffer verfällt untrainiert.

### C3. Belohnung: nur Überleben + Neugier — neu spezifiziert

**Wichtig (RL-Review, Critical):** Der bisher angenommene „v1-Homöostase-Term" existiert
nicht — der 0.6-Term (`agent.py:1142`) ist Goal-Stack-Completion-Shaping des
v1-Erfindungsplaners, den B6 abschaltet. Es gibt heute keinen Term, der Energie/Health in
Belohnung münzt, und keinen Todes-Malus. Der physiologische Anteil wird daher NEU gebaut:

```
r_energy  = 0.6 * ΔE / 45.0            # Δ Energie pro Tick; Normierung: 1 volle Mahlzeit ≙ +0.6
r_health  = 0.6 * ΔH / 50.0            # Verletzung/Vergiftung wird gefühlt
r_deficit = −0.3 * max(0, (60 − E)/60) # Hungerstress unter 25 % MAX_ENERGY, kontinuierlich
r_death   = −3.0                        # Terminal-Malus auf der Todes-Transition
r_int     =  0.3 * curiosity            # C4
```

Alles davon ist Physiologie (Zustandsgradienten + Terminal), nichts ist ein Event-Bonus.
**Ersatzlos gestrichen (nur v2-Modus):** alle Terme der B6-Reward-Spalte. Essen lohnt,
weil ΔE positiv ist; Werkzeuge lohnen, weil sie pro Ermüdung/Zeit mehr Kalorien
erschließen (13×-Delta bewiesen). Das ist die Kern-Wette dieses Schnitts; der Pilot
(Plan 5) misst sie. Fallback bei Nicht-Lernen laut User-Entscheidung: weitere
real-physiologische Zwischensignale (z. B. Sättigungs-/Schmerzgradienten), niemals
Event-Boni — nur datengetrieben nach Pilot-Befund.

### C4. Neugier: drei Quellen, rausch-fest

Der bestehende Intrinsic-Mechanismus hat drei vom RL-Review belegte Pathologien: das
Target ist teils selbstbezüglich (u. a. `last_reward` — enthält die Neugier selbst),
`rew_err = pred_reward.abs()` ist gar kein Fehlerterm, und Running-Std-Normalisierung
zahlt eine ewige Rausch-Rente auf irreduzible Zufälle (der Strike-RNG würde zur
Neugier-Farm). Neugier wird deshalb so gebaut:

```
curiosity = 0.25 * nextslot_err + 0.50 * causal_epistemic + 0.25 * novelty
```

1. **`nextslot_err`** — Vorhersagefehler des World-Model-Heads auf den **rohen, maskierten
   Slot-Features (17×10)** des nächsten Ticks (NICHT auf `obj_ctx`, das mit den eigenen
   Gewichten wandert und nie konvergiert). Aus dem bestehenden next-obs-Target werden die
   selbstbezüglichen/nichtstationären Dims ausgeschlossen: `last_reward` (Dim 26), die 12
   Episodic-Retrieval-Dims, die 3 Causal-Memory-Dims, die 8 Hormon-Dims. Normalisierung
   **per Dim** mit laufendem Mean+Std (nicht global, nicht nur Std). Der bisherige
   `rew_err`-Term wird ersatzlos gestrichen.
2. **`causal_epistemic`** — das Causal Model (unten) prädiziert **Verteilungen** (μ, σ per
   Dim, NLL-Training). Neugier-Anteil = nur die Überraschung jenseits des gelernten
   Rauschens: `mean(max(0, z² − 1))` mit `z = (x − μ)/σ`. Ein auskonvergierter, rein
   aleatorischer Prozess (Fragment-Zufall im `strike`) zahlt damit gegen 0 — ewiges
   Steineschlagen als Neugier-Farm ist zu.
3. **`novelty`** — Count-based über Eigenschafts-Buckets, pro Agent:
   `key = tuple(floor(p/0.25) für 13 Props)`, `novelty = 1/√n(key)` beim Wahrnehmen/
   Halten/Erzeugen, LRU-gekappt (4096 Keys). Direkt aus dem Archive-Prototyp
   (`_bucket_counts`/`info_gain`) übernommen — zahlt entlang der ganzen Kette (erstes
   Fragment, erste scharfe Kante, erstes Fleischstück), zustandsbasiert, ohne
   designer-gewählte Events (Grenzentscheid oben).

**Causal Model (`systems/causal_model.py`):** ehrliche **Neuimplementierung** in torch
(der Archive-Prototyp ist numpy, 12-Prop/8-Aktionen — strukturell nicht übernehmbar;
übernommen werden seine Ideen: `err_ema`, Bucket-Counts). Eingabe:
`detach(gru_h) ⊕ verkörperter Aktions-Vektor (22) ⊕ detach(embed des gewählten Slots)`;
Ausgabe: (μ, log σ) über die 17 Slot-Features **desselben Objekts** im nächsten Tick
(Identitäts-Tracking aus C1; bei Zerstörung durch strike: das massereichste Fragment; aus
der Wahrnehmung gefallen: maskiert, kein Loss). `detach` ist entschieden: der eigene
Optimizer (LR × Plastizitäts-Gen) trainiert NICHT durch GRU/Encoder hindurch — das Causal
Model liest die Repräsentation, es formt sie nicht (Repräsentations-Dynamik bleibt bei
PPO).

### C5. Gene, Checkpoints, Planner

- **Neues Gen `strength`** (16.; `genetics.py` ist key-basiert, kein Breaking Change),
  Mutations-σ in der 0.012-Klasse (wie `plasticity` — NICHT 0.25, das wäre ~25 % der
  Range pro Generation). Speist `Body(strength=...)`; ersetzt in 3b den 3a-Default 0.5.
- **Checkpoint:** `physics_v2` und `CHECKPOINT_FORMAT_VERSION = 2` werden Teil des
  Pickle-Payloads. `_load_checkpoint` prüft beides **vor** dem bestehenden
  broad-`except` und **re-raised** bei Mismatch (heute würde ein werfender Guard still
  verschluckt und „frisch gestartet" — stiller Datenverlust statt harter Schranke).
  Bekanntes Gotcha bleibt: stales root-`checkpoint.pkl` vor Läufen löschen.
- **Planner-Pfad:** `plan_action`/`imagine_rollout` (brain.py) sind mit der neuen
  Architektur inkompatibel (füttern `encoder(pred_next_obs)` ohne `obj_ctx` in die GRU;
  wählen per `argmax` — Off-Policy-Bias im PPO-Ratio und Konflikt mit Prinzip 1). Im
  v2-Modus **deaktiviert**; Aktionen kommen ausschließlich aus dem Policy-Sampling.
  (Flag aus: unverändert.) `inherit_weights_from` überspringt Shape-Mismatches still
  (brain.py:180) — beim Architektur-Wechsel gewollt, wird im Code kommentiert.

---

## Abschnitt D — Tests, Golden, Metriken, Nicht-Ziele

### D1. Erhaltung (Plan 3a, härtester Gate-Teil)

- **Massen-Ledger-Invariante:**
  `bodenmasse + händemasse_aller_lebenden + ledger[eaten] + ledger[decayed] == ledger[spawned] + ledger[from_carcass]`
  (`math.isclose`, 1e-9).
  **Als Zufallssequenz-Fuzzer** (kein Hypothesis im Repo — handgeschrieben mit
  `random.Random(seed)`): 50 Seeds × 200 zufällige Aktionen aus
  {spawn, grasp, release, strike, cut, eat, decay-tick, overload-drop,
  agent-stirbt-mit-2-Objekten}, Invariante nach JEDER Aktion. Begründung: die
  Erhaltungsaussage ist die härteste Garantie des Designs und darf nicht von der
  Seed-Wahl des Autors abhängen. Dazu Beispieltests als schnelle Smoke-Checks.
- **Mikro-Invarianten:** `sum(fragment.mass) == target.mass` (strike) und
  `extracted.mass + remainder.mass == target.mass` (cut), je exakt.
- **Kadaver-Einmal-Münzung:** Tod im v2-Modus ⇒ genau ein Kadaver-Objekt mit `body_mass`,
  kein v1-Pool-Credit, kein Loot; gehaltene Objekte liegen am Boden.
- **Expliziter Verzweigungstest Flag aus:** `physics_v2=False` ⇒ Tod erzeugt weiterhin
  `CORPSE_ENERGY`-Zell-Credit und KEIN PhysObject; Kill vergibt weiterhin Loot (direkter
  Assert beider Zweige — Golden beweist nur den Default-Pfad, nicht die Verzweigung).
- **Kill-ohne-Doppel-Energie:** v2-Kill: Angreifer-Energie unverändert (kein Loot),
  Kadaver-Objekt vorhanden.

### D2. Exploit- & Kalibrierungs-Regression (Plan 3a)

- ½mv²-Kappung: 0.05-kg-Kiesel liefert ≤ 4.9 J, egal welcher Effort.
- **Positiv-Regression Knapping:** 0.5-kg-Schlagstein (Granit) bricht 0.8-kg-Flint bei
  Effort 0.8 — hält fest, dass eine spätere Verschärfung von `V_MAX_STRIKE` Knapping
  nicht still killt.
- Klingen-Massen-Faktor beidseitig: 20-g-Splitter < 40 % eines 300-g-Abschlags UND
  4-kg-Brocken < 300-g-Abschlag (gleiche Props).
- exert↔strike-Bindung: jeder `do_strike` erhöht Ermüdung, auch bei No-op-Wirkung.
- Überlast-Drop: Ermüdung senkt Kapazität ⇒ schwerstes Objekt fällt, Ledger stimmt.
- Essen: 1 kg raw_meat ⇒ 45 ± 1 Sim-Energie (**Produkt-Test**, s. B5); Granit essen ⇒ +0.
- Verwesung: nutrition sinkt, toxicity steigt (Kappe 0.6), Masse fließt in
  `ledger[decayed]`.
- **Gate-Erweiterungs-Test:** alle Konstanten aus `CALIBRATED_ACTION_PARAMS` und alle
  kind-`spawn`-Einträge haben Kalibrierungs-Eintrag + Doc-Zeile; Orphan-Prüfung für die
  neuen Kinds (sonst ist das Gate grün, ohne je zu prüfen).
- DiscoveryV2 pro Welt: zwei `Simulation`-Instanzen ⇒ `id(a.world.objects.discovery) !=
  id(b.world.objects.discovery)`; Modul-Singleton-Export entfernt.

### D3. Gehirn (Plan 3b)

- Permutations-Invarianz: Slot-Shuffle ⇒ identische **Verteilungsparameter** (μ, log_std
  der Köpfe, zurücksortierte attn-Gewichte) — Vergleich VOR dem Sampling, nicht auf
  Samples (RNG-Konsum-Differenzen machen Sample-Vergleiche flaky).
- All-Masked: kein Objekt in Sicht, leere Hände ⇒ `obj_ctx == 0`, kein NaN, Verben lösen
  No-op aus.
- Form-Tests: 29 kontinuierliche Dims + 2 Kategorial-Köpfe, GRU-Input 192,
  Checkpoint-Guard re-raised bei Alt-Datei/Flag-Mismatch.
- Kategoriale Auswahl: log-prob der Slot-Ziehung geht in die Policy-log-prob ein; leere
  zulässige Menge ⇒ No-op ohne log-prob-Beteiligung; `research_drive` fehlt in der
  log-prob-Summe (v2).
- Belohnungs-Reinheit: gemockte Events (Erfindung, Kochen, Kooperation, Territorium,
  Sprache, Hamilton) ⇒ Belohnung enthält exakt nur r_energy/r_health/r_deficit/r_death/
  r_int (Test rechnet gegen).
- v1-Trigger tot: `physics_v2=True`, Invention-Trigger-Funktionen per Spy überwacht ⇒
  0 Aufrufe über N Ticks.
- Neugier: (a) selbstbezügliche Dims nachweislich nicht im nextslot-Target; (b)
  aleatorischer Prozess (fixe Verteilung) ⇒ `causal_epistemic → 0` nach Konvergenz;
  (c) Bucket-Novelty fällt mit 1/√n.
- Terminal-Transition: sterbender Agent schreibt `done=True`-Transition, Buffer wird
  trainiert.
- Causal-Model: NLL sinkt auf synthetischer deterministischer Sequenz; σ wächst auf
  synthetisch verrauschter.

### D4. Pilot-Vorbereitung (für Plan 5; Zuordnung 3a/3b markiert)

Metriken-Hooks (Logging, kein Verhalten): **nach 3a messbar:** Fragmente je Tick,
Schnitte mit Werkzeug vs. Hand (geskriptet), gegessene kcal je Quelle, Ledger-Flüsse.
**Erst nach 3b vollständig:** DiscoveryV2-Events je Agent (braucht Agenten-Aktion),
Griff-/Schlag-/Schneide-Raten aus der Policy, Neugier-Zerlegung (drei Quellen separat
geloggt — Pilot-Diagnostik). Der Pilot (Schritt 5) schließt erst nach 3b an.

### D5. Nicht in diesem Design (bewusst)

Kultur-Korrektur (Plan 4: Lamarck-Pfade raus), Sprache über Objekte, Feuer/Kochen
gelernt, Bauen aus realen Materialien, GPU-Batching der Objekt-Schicht UND der
Objekt-Wahrnehmung (der Slot-Encoder liegt im heißen Pro-Tick-Pfad — bekannte
Folgeaufgabe für die Tier-5-Engine, dort explizit einplanen), Trade/Ritual-De-Scripting,
Geschwindigkeits-Malus beim Tragen (Ermüdung reicht), Mehr-Agenten-Kooperation an einem
Objekt.
