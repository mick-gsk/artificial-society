# M1-Diagnose-Batterie — gezielte Ablation der Lern-/Emergenz-Knobs im Physik-v2-Pfad

Datum: 2026-07-04 · Setting: `Simulation(physics_v2=True)`, Welt 40×30, pop 30, 5000 Ticks/Lauf ·
Harness: Erweiterung von `scripts/m1_pilot.py` (In-Prozess-Runtime-Patches, keine Paket-Änderung).

Ziel: Nicht nur EIN A/B (Lernen ON/OFF), sondern eine Reihe, deren Ergebnismuster verrät, welche
Optimierung als Nächstes am meisten bringt. Trennt H1 (Horizont) · H2 (intrinsische Dosierung) ·
H3 (soziale Weitergabe) · H4 (Lernen als Ganzes).

---

## 0. Vorab-Verifikation: Welche Knobs sind im v2-Pfad überhaupt wirksam?

Dies ist das Fundament der Batterie. Ergebnis mit Datei:Zeile-Belegen (Repo-Stand as-pilot):

### F1 — Läuft der World-Model-Planer (PLAN_HORIZON / imagine_rollout) im v2-Pfad?
**Nein — der Planer ist im v2-Pfad tot.** Der v2-Aktionspfad ruft `act_v2` (`agents/agent.py:1352`
`if self.physics_v2:` → `:1359` `brain_step = self.brain.act_v2(...)`; Kommentar `:1355`:
„plan_action/imagine_rollout sind mit der v2-Architektur [nicht kompatibel]"). `act_v2`
(`agents/brain.py:385`) sagt im Docstring `:386` explizit **„Sampling, nie Argmax; Planner
deaktiviert (C5)"** und zieht die Aktion direkt aus `forward_v2` + `Normal.rsample` (`brain.py:397–400`).
`imagine_rollout` (`brain.py:638`, nutzt `PLAN_HORIZON`/`PLAN_HORIZON_RESEARCH` in `:642/:726`) wird
im v2 nie aufgerufen.
→ **Konsequenz: H1 kann NICHT über `PLAN_HORIZON` getestet werden (toter no-op).** Der äquivalente
v2-Horizont-Knob ist **`GAMMA_V2 = 0.99`** (`brain.py`, Kommentar „Kredit-Horizont ~100 Ticks
(Knapping→Kadaver→Schneiden→Essen)"), gelesen im GAE-Loop von `_train_v2` (`brain.py:918`,
`delta = rewards[t] + GAMMA_V2 * next_values[t] * ...`). H1 wird über GAMMA_V2 operationalisiert.

### F2 — Wo wirkt NOVELTY_WEIGHT / intrinsic_reward im v2-Pfad? Fließt Intrinsik in `_train_v2`?
**`NOVELTY_WEIGHT` (0.15, `brain.py:72`) ist im v2 tot** — es wird ausschließlich in
`imagine_rollout` (`brain.py:684`) benutzt, und der Planer läuft im v2 nicht. Ebenso ist
`Brain.intrinsic_reward` (`brain.py:810`, prediction-error-curiosity × NGU-Novelty) tot: der Aufruf
steht im `if not self.physics_v2:`-Zweig (`agent.py:1505` → `:1509`).
**Der lebende v2-Neugier-Kanal ist `Agent._assemble_curiosity_v2`** (`agent.py:~1017`):
`curiosity = clamp(0.25·nextslot_err + 0.50·causal_epistemic + 0.25·novelty, 0, 2)`. Diese Größe
fließt in den v2-Reward (`agent.py:~1538`: `reward = 0.6·ΔE/45 + 0.6·ΔH/50 − 0.3·deficit +
0.3·curiosity`), wird als `effective_reward` via `store_transition_v2` (`agent.py:~1548`) gepuffert
und **dann in `_train_v2` als Belohnung trainiert** (GAE über `rewards[t]`). Also: **ja, Intrinsik
fließt in `_train_v2` — aber über die Reward-Spur, nicht als separater Term.** Der Knob, der
Exploration im v2 tatsächlich dosiert, ist **nicht `NOVELTY_WEIGHT`**, sondern der Curiosity-Gewicht-
Faktor (inline `0.3·curiosity` bzw. die 0.25/0.50/0.25-Gewichte in `_assemble_curiosity_v2`) — ein
Inline-Literal, kein Modul-Konstante → nur per Methoden-Wrap patchbar (siehe C1/C2).

### F3 — Wirkt `social_learning_step` auf das v2-VERHALTEN, oder nur `imitate_from` auf die Gewichte?
**Beide Kanäle berühren v2, und `social_learning_step` läuft im v2 mit.** Der Aufruf steht
`if tick % 3 == 0` (`agent.py:1473`) **außerhalb** der `not self.physics_v2`-Guards, gilt also auch
für v2. `social_learning_step` (`systems/social_learning.py`) hat drei Effekte: (1) Kausal-Sequenz-
Transfer `causal_mem.receive_transmitted(...)`, (2) aktives Lehren, (3) `agent_brain.imitate_from(
other_brain)` — Gewichtsangleichung. **Kanal auf v2-Verhalten:** (a) `imitate_from` nudged die von
v2 GETEILTEN Gewichte (encoder/gru/policy) direkt. (b) Der Kausal-Transfer wirkt indirekt: v2 liest
`causal_memory.feature_vector()` als Obs-Dims 34–36 (`agent.py:518`, Teil der 57-Obs, die in
`forward_v2` gehen). → **„nosocial" isoliert BEIDE Kanäle** (Gewichte + Kausal-Features). Hinweis:
Der Social-Reward (+0.05…0.08) zahlt im v2 nicht — er wird beim Reward-Reassign (`agent.py:~1538`,
C3) verworfen; nur die Verhaltens-/Gewichtseffekte bleiben.

### F4 — Weitere leicht patchbare Konstanten mit großem Emergenz-Hebel im v2-Pfad (max. 2)
1. **`ENTROPY_COEF_NEW = 0.01`** (`brain.py`, Konstanten-Block) — Entropie-Bonus der neuen
   Manipulations-Köpfe (Verben/Effort/Slot-Queries), gelesen **call-time** im Loss von `_train_v2`
   (`loss = ... − ENTROPY_COEF_NEW · ent_new.mean()`). Direkter Hebel auf Verb-Exploration
   (strike/cut/grasp). **Patchbar** (Modul-Global, Body-Referenz).
2. **`VERB_INIT_BIAS = 0.2`** (`brain.py`) — optimistischer Init-Bias auf die Verb-Means
   („Manipulations-Babbling-Prior", Spec C2), gelesen in `Brain.__init__`
   (`self.policy_mean.bias[VERB_SLICE] = VERB_INIT_BIAS`). **Patchbar, aber construction-time**:
   nur wirksam, wenn VOR dem `Simulation(physics_v2=True)`-Bau gesetzt (Brains werden bei Sim-Init
   gebaut). Im Runner-Prozess trivial (Konstante setzen → dann Sim bauen).

### Knob-Patchbarkeits-Tabelle (Projekt-Gotcha roadmap §7: def-time-Default = toter no-op)
| Knob | Ort | v2 lebendig? | Runtime-patchbar? | Mechanik |
|---|---|---|---|---|
| `PLAN_HORIZON`, `PLAN_HORIZON_RESEARCH` | `brain.py:70/71`, nur `imagine_rollout`-Body | **nein (Planer tot)** | Body-lesbar, aber **effektlos** | — |
| `NOVELTY_WEIGHT` | `brain.py:72`, nur `imagine_rollout:684` | **nein** | Body-lesbar, aber **effektlos** | — |
| `PLAN_CANDIDATES` | `brain.py`, **def-time Default-Arg** | — | **NEIN** (roadmap §7: toter no-op) | — |
| `GAMMA_V2` | `_train_v2`-Body (GAE) | **ja** | **ja** (Modul-Global, call-time) | `brain.GAMMA_V2 = x` |
| `ENTROPY_COEF_NEW` | `_train_v2`-Body (Loss) | **ja** | **ja** (call-time) | `brain.ENTROPY_COEF_NEW = x` |
| `VERB_INIT_BIAS` | `Brain.__init__` | **ja** | **ja, construction-time** | vor Sim-Bau setzen |
| Curiosity-Gewicht (`0.3·curiosity`, 0.25/0.50/0.25) | Inline-Literale in `agent.py` step / `_assemble_curiosity_v2` | **ja** | nur per **Methoden-Wrap** | `Agent._assemble_curiosity_v2` umhüllen |
| `social_learning_step` | `agent.py:1473`, `systems/social_learning.py` | **ja** | **ja** per no-op | `social_learning.social_learning_step = lambda ...: 0.0` |
| `Brain.maybe_train` / `imitate_from` | `brain.py` | **ja** | **ja** per no-op (= bestehender nolearn-Arm) | wie `m1_pilot._freeze_learning` |

---

## 1. Harness & Patch-Mechanik (CONSTRAINT a)

Basis ist das bestehende `scripts/m1_pilot.py`: es setzt Patches **im Runner-Prozess VOR dem Import
von `Simulation`** (`_freeze_learning()` patcht `Brain.maybe_train`/`imitate_from` zu no-ops) und
schreibt pro Lauf eine JSONL mit `_snapshot()`. Jede Variante unten ist eine solche Patch-Funktion.
Regeln:
- **PYTHONHASHSEED=0 pinnen** (roadmap §7) — sonst divergieren Seeds prozessübergreifend.
- Modul-Konstanten (`GAMMA_V2`, `ENTROPY_COEF_NEW`, `VERB_INIT_BIAS`) werden als
  `import artificial_society.agents.brain as brain; brain.X = val` gesetzt, **vor** dem
  `from artificial_society.simulation import Simulation`. Für `VERB_INIT_BIAS` ist das zwingend
  (construction-time). Für die call-time-Konstanten reicht „vor dem ersten `sim.step()`".
- Methoden-Wraps (`social_learning_step`, `_assemble_curiosity_v2`) ersetzen das Attribut auf
  Klasse/Modul, ebenfalls vor Sim-Bau. Kein Paket-Code wird editiert (HOT-Files unberührt).
- Seedgleichheit: RNG-Guards in `social_learning` werden VOR den Seiteneffekten ausgewertet
  (m1_pilot-Docstring) — ein no-op verschiebt den Zufallsstrom nicht; Welt-Startzustand identisch.
- Neue CLI: `--arm` um die Varianten erweitern; jeder Arm ist eine Patch-Fn in einer
  `VARIANTS`-Registry. `_snapshot()` bleibt unverändert (die Metriken decken alle Hypothesen ab).

---

## 2. Experiment-Matrix

Basis-Arme 5 Seeds, Varianten 3 Seeds (CONSTRAINT b). Referenz: jede Variante ist gegen **A1 (learn)**
UND **A2 (nolearn)** interpretierbar (CONSTRAINT c) — A1 = volle Maschinerie, A2 = eingefrorene
Random-Policy (Untergrenze). Alle Varianten laufen auf dem **learn**-Substrat (PPO an), sofern nicht
anders vermerkt, damit der isolierte Knob der einzige Unterschied zu A1 ist.

| # | Name | Gepatchte Konstante/Funktion (Modul-Pfad) | Seeds | Testet |
|---|---|---|---|---|
| A1 | `learn` | — (unverändert) | 5 | H4-Referenz (volle Maschinerie) |
| A2 | `nolearn` | `Brain.maybe_train`→no-op, `Brain.imitate_from`→no-op (`agents/brain.py`) | 5 | H4 (Lernen als Ganzes) + Untergrenze |
| B1 | `gamma-short` | `brain.GAMMA_V2 = 0.80` (`agents/brain.py`, `_train_v2` GAE) | 3 | H1 (Kredit-Horizont, Planer-Ersatz) |
| C1 | `curio-off` | Wrap `Agent._assemble_curiosity_v2` → gibt `0.0` zurück (`agents/agent.py`) | 3 | H2 (Intrinsik trägt nichts?) |
| C2 | `curio-high` | Wrap `Agent._assemble_curiosity_v2` → `3.0 × Originalwert` | 3 | H2 (Intrinsik unterdosiert?) |
| D1 | `nosocial` | `social_learning.social_learning_step = lambda a, ag, t: 0.0` (`systems/social_learning.py`) | 3 | H3 (Kultur-Kanal isolieren) |
| E1 | `entropy-high` | `brain.ENTROPY_COEF_NEW = 0.05` (`agents/brain.py`, `_train_v2` Loss) | 3 | Zusatz-Lever: Verb-Exploration |
| F1 | `verb-bias-high` | `brain.VERB_INIT_BIAS = 0.6` **vor Sim-Bau** (`Brain.__init__`) | 3 | Zusatz-Lever: Manipulations-Prior |

**Gesamt: 5 + 5 + 3·6 = 28 Läufe** (unter dem 30-Budget; 16 Kerne → ~2 Wellen à 14 Prozesse,
5000 Ticks/Lauf).

### Erwartetes Ergebnismuster je Hypothese (Kern-Metrik: `cuts_with_tool` und `tool_cut_ratio`;
Sekundär: `discoveries`, `fragments_total`, `verbs_fired["strike"/"cut"]`; Konfound-Kontrolle: `pop`,
`mean_energy`)

- **H1 wahr (Horizont/Kredit-Reichweite bindend):** B1 (gamma-short) senkt `tool_cut_ratio`/
  `discoveries` deutlich unter A1 — die ~100-Tick-Kette Knapping→Kadaver→Schneiden→Essen zerfällt bei
  γ=0.80. A1 selbst zeigt dann klar > 0 Tool-Cuts. *Empfehlung: Kredit-Backfill / Eligibility-Traces /
  längerer effektiver Horizont bauen.*
- **H2 wahr (Intrinsik falsch dosiert):** C1 (curio-off) fällt Richtung A2 (Exploration bricht weg,
  `discoveries`↓); ODER C2 (curio-high) hebt `discoveries`/`tool_cut_ratio` klar über A1 (war
  unterdosiert). *Empfehlung: Curiosity-Gewicht / Novelty-Ziel neu tunen oder besseren
  Neugier-Schätzer bauen.*
- **H3 wahr (soziale Weitergabe ist Träger):** D1 (nosocial) fällt deutlich unter A1 bei sonst
  gleicher `pop`. *Empfehlung: Kultur-Kanal (Kausal-Transfer/Imitation) verstärken/priorisieren.*
  **H3 falsch (Kultur trägt nichts):** D1 ≈ A1. *Empfehlung: Kultur-Kanal deprioritisieren.*
- **H4 wahr (Lernen bringt nichts):** A1 ≈ A2 auf allen Emergenz-Metriken. *Empfehlung: Bevor
  Feintuning — die Lern-Kopplung selbst reparieren (Reward erreicht die Knapping-Aktion nie).*
- **Zusatz-Lever:** Hebt E1 oder F1 `tool_cut_ratio`/`discoveries` über A1, ist Exploration (nicht
  Kredit) der billige nächste Schritt.

---

## 3. Auswerte-Schema → Empfehlung („als Nächstes bauen: X")

Pro Arm über Seeds: Median + IQR von `tool_cut_ratio`, `cuts_with_tool` (absolut), `discoveries`,
`fragments_total`; sowie `pop`/`mean_energy` als Konfound-Kontrolle (ein Arm, der nur die Population
kollabiert, senkt Discovery trivial — solche Arme sind ungültig, nicht „Hypothese bestätigt").
Reporting via erweitertem `scripts/m1_report.py` (aggregiert die JSONL-`final`-Records).

**Entscheidungsbaum (in Reihenfolge auswerten):**
1. **A1 ≈ A2** (learn schlägt frozen-random NICHT auf `tool_cut_rat`/`discoveries`)?
   → H4 dominant. Kein Knob wird die Diagnose ändern. **Baue: Kredit-Zuweisung/Kopplung** (der
   mechanistische Werkzeug-Payoff existiert in der Physik, wird aber nie der Aktion gutgeschrieben).
   Bestätige indirekt über Muster in Schritt 2.
2. **A1 > A2, aber B1 ≈ A1** (Horizont-Verkürzung schadet kaum, Tool-Cuts bleiben ~0):
   → Kredit-Reichweite ist NICHT das Nadelöhr; das Signal erreicht die frühe Aktion (Knapping)
   ohnehin nicht. **Baue: Credit-Backfill/Reward-Attribution entlang der Kausalkette** (nicht bloß
   längerer Horizont).
3. **A1 > A2 und B1 ≪ A1** (γ-Kürzung bricht Tool-Cuts): H1 bestätigt.
   **Baue: Eligibility-Traces / längeren effektiven Kreditpfad** (der lange Horizont wird genutzt,
   ist aber fragil).
4. **C1 ≪ A1 oder C2 ≫ A1:** H2 bestätigt. **Baue: Neugier-Dosierung/-Schätzer** (billiger Tune-Win).
5. **D1 ≪ A1:** H3 bestätigt. **Baue: Kultur-Kanal** (Kausal-Transfer + Imitation verstärken).
6. **E1/F1 ≫ A1:** Exploration ist unterdosiert — **Baue: Explorations-Bonus** (Entropie/Prior)
   als billigsten ersten Schritt, vor der teuren Kredit-Reparatur.

Priorität bei mehreren Treffern: H4/Kredit (Schritt 1–3) schlägt H2/H3/Exploration, weil ohne
funktionierende Kredit-Zuweisung kein Feintuning greift.

---

## 4. Interpretations-Vorbehalte — was die Batterie NICHT unterscheiden kann

- **Credit-Backfill / retroaktive Reward-Attribution ist KEIN Runtime-Knob.** Es gibt keine
  Modul-Konstante, die eine Eligibility-Trace oder ein Rückwärts-Kredit-Schema an/ausschaltet; das
  wäre neuer Paket-Code. Die Batterie kann H4 also nicht direkt „Credit-Backfill würde helfen"
  beweisen. **Indirekte Stützung:** Das Muster „A1 > A2, aber JEDE reward-formende/horizont-Variante
  (B1, C1, C2, E1) lässt `tool_cut_ratio` nahe null" bei nachweislich vorhandenem mechanistischem
  Payoff (physik-seitig erhöht `cut` mit scharfer Kante den Ertrag) ist die Signatur eines
  Kredit-Zuweisungs-Defekts: der Reward existiert, erreicht die auslösende Knapping-Aktion aber nie —
  invariant gegen Horizont und Gewicht. Umgekehrt: schlägt B1 stark durch, ist der lange Horizont
  live und Backfill/Traces sind der logische nächste Baustein.
- **World-Model-Qualität vs. Policy** ist nicht separierbar: `nextslot_err` (Teil der v2-Curiosity)
  wird geloggt, aber es gibt keinen Runtime-Schalter, der nur den Welt-Modell-Kopf einfriert. Ein
  schlechtes `predict_world_v2` würde die Neugier verrauschen, ohne dass C1/C2 das von echter
  Under-/Overdosierung trennen könnten.
- **`nolearn` (A2) ist keine REINE Zufallspolicy:** `_freeze_learning` schaltet nur PPO
  (`maybe_train`) und `imitate_from` ab. Der Kausal-Sequenz-Transfer über Obs-Dims 34–36
  (`causal_memory.feature_vector`) fließt in A2 WEITER. A2 ist also „random-Gewichte + noch fließende
  Kultur-Features". Für eine strikte Random-Null müsste A2 zusätzlich `nosocial` tragen — bewusst
  nicht getan, um den bestehenden Harness bit-kompatibel zu halten; bei H3-Interpretation
  mitdenken (D1 isoliert den Kanal sauberer als der A1↔A2-Vergleich).
- **VERB_INIT_BIAS (F1) mischt zwei Effekte:** stärkerer Prior erhöht Verb-Feuerrate UND verschiebt
  den Startpunkt der Policy; ein F1-Effekt beweist „mehr Manipulation hilft", nicht welcher der
  beiden Teilmechanismen. Als Explorations-Grobtest ausreichend, nicht als Feinursache.
- **Interaktionen sind unbeobachtet:** jede Variante ändert genau einen Knob gegen A1; die Batterie
  ist ein Screening (Haupteffekte), kein faktorielles Design. Ein Knob, der nur in Kombination mit
  einem zweiten wirkt (z. B. Horizont × Neugier), bleibt unsichtbar.
