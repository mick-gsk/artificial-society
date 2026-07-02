# DESIGN — Agent-internes Kausalmodell (PROTOTYP)

> **STATUS: EXPLORATIVER PROTOTYP — NICHT Teil der M1–M6-Experimentmatrix.**
> Kein Publikationsanspruch, keine Baseline/Kill/M5-Gating-Apparatur. Ziel: qualitativ
> sehen, ob Agenten ein *übertragbares* Vorwärtsmodell der Material-Physik bilden
> (= Kausaltheorie) und ob ein epistemischer Antrieb das Erfindungsverhalten verändert.
> Promotion zu einem echten Hebel (M7) ist eine spätere, separate Entscheidung.

## Motivation — die 5 Lücken

Die heutige `CausalMemory` (systems/culture.py) ist trotz des Namens eine flache
Kontingenztabelle `(action, mat_a, mat_b) → {successes, attempts, reward_EMA}` über
*Oberflächen-Labels*. Sie steht auf Pearls Sprosse 1 (Assoziation). Fünf Bausteine fehlen,
damit Agenten Kausaltheorien wie Menschen bilden:

| # | Lücke | Heute | Fix in diesem Prototyp |
|---|-------|-------|------------------------|
| L1 | Latente Variablen statt Labels | Schlüssel = Material-Name | Agent perzipiert `PROP_DIMS` (12-d) der Inputs |
| L2 | Strukturierte Hypothese | flaches dict | Sensitivitäts-Probe ∂out/∂in → lesbare „Theorie"-Tabelle |
| L3 | Intervention / Experiment | `random.choice` über Bedarf | Auswahl nach erwartetem Informationsgewinn (seeded Softmax) |
| L4 | Agent-eigenes Vorwärtsmodell | keins (combine_vectors = Welt) | pro-Agent numpy-MLP `f(φ) → ŷ` |
| L5 | Prediction-Error-Lernsignal | nur Reward | `surprise = ‖ŷ − y‖` als epistemischer Term |

L2 und L6 (kompositioneller Transfer) fallen weitgehend *gratis* an, wenn das Modell über
latente Eigenschaften statt über Labels parametrisiert ist.

## Modell-Realisierung

- **numpy-MLP, nicht torch.** Bewusst: vermeidet Entanglement mit dem globalen torch-RNG-Stream
  des Brains. Architektur: Input `φ = [props_a(12) ‖ props_b(12) ‖ action_onehot(8)] = 32`
  → 1 Hidden-Layer (tanh, Breite 32) → Output 12, sigmoid auf [0,1]. ~1.4k Parameter/Agent.
- **Online-Lernen:** ein SGD/MSE-Schritt pro beobachteter Kombination (manuelle Backprop, numpy).
- **Determinismus:** Gewichts-Init über einen **lokalen, aus `agent.id` geseedeten**
  `np.random.Generator` — isoliert (perturbiert den globalen Stream nicht) und reproduzierbar.
  Konform zu CLAUDE.md („eine explizit geseedete Instanz"), kein bare global-numpy.

## Architektur & Lane-Disziplin

- **Neues Modul** `artificial_society/systems/causal_model.py` (systems-Lane). Klasse `CausalModel`:
  `predict(φ)`, `observe(φ, y)` (= ein Lernschritt, gibt surprise zurück), `sensitivity()`
  (Probe für L2), `info_gain(cands)` (L3-Score), `transfer_error(pairs)` (Messung).
- **KEINE Hot-File-Edits.** `agents/agent.py` und `environment/materials.py` bleiben unangetastet.
  Das Modell wird **lazy** in den Invention-Funktionen (systems-Lane) angelegt:
  `getattr(agent, "causal_model", None)`; bei Flag ON + None → erzeugen.
- **Flag** `AS_CAUSAL_MODEL` (default OFF), Helper `_cm_enabled()` analog zu `_m1_enabled()`.

## Datenfluss pro Invention-Versuch (Flag ON)

Eingehängt in **beide** Pfade (`agent_try_invention`, `agent_invent_from_need`) dort, wo
`new_vec` entsteht:

1. **L1** `props_a/props_b = get_vector(...)` (existiert bereits) → `φ`.
2. **L4** vor `combine_vectors`: `pred = model.predict(φ)`.
3. Welt rechnet `new_vec = combine_vectors(...)` (unverändert).
4. **L5** `surprise = ‖pred − new_vec‖`; `model.observe(φ, new_vec)`; epistemischer Term
   `k·surprise` fließt in den von der Funktion zurückgegebenen Reward (→ CausalMemory.record).
5. **L3** Kandidatenwahl `(mat_a, mat_b, action)` wird ON nach erwartetem Informationsgewinn
   gebogen (Proxy: Modell-Fehler-EMA + count-basierte Novelty über grob-diskretisierte
   Prop-Region), seeded Softmax über den lokalen Generator. Ändert *welche* Kombination
   probiert wird → Trajektorie divergiert (nur ON, beabsichtigt).

> Hinweis: Der Reward-Rückgabewert der Invention-Funktionen wird am Call-Site
> (agent.py:1173/1183, **Hot File**) heute nur als Boolean genutzt (+0.5/+1.0 flat), die
> Magnitude fließt NICHT ins Brain-RL. Der epistemische Term wirkt im Prototyp daher über
> **L3 (Selektion)** und CausalMemory, nicht direkt aufs Brain-RL. Eine spätere
> Magnitude-Propagation wäre ein core-lead-gerouteter Einzeiler — bewusst out-of-scope.

## Determinismus-Invariante (harte Akzeptanzbedingung)

Flag **OFF**: kein Modell angelegt, kein predict/observe, keine Selektionsänderung,
**null zusätzliche globale RNG-Draws** → `tests/test_regression_golden.py` und der
Headless-Digest bleiben byte-identisch. Vor jedem Commit verifiziert.
Flag **ON**: reproduzierbar über die pro-Agent-Seeds.

## Observability — so wird Kausaltheorie-Bildung sichtbar

- **Transfer-Metrik (Kerntest):** Modell auf nie kombinierten Materialpaaren evaluieren;
  mittlerer Vorhersagefehler vs. naive Baseline (predict-mean). Fehler < Baseline auf
  ungesehenen Paaren ⇒ echte Generalisierung = Theorie gebildet.
- **L2-Lesbarkeit:** Finite-Differenzen-Sensitivität ∂output/∂input_prop → Tabelle
  („hardness↑, dryness↑ ⇒ heat_emission↑").
- **Verhaltens-Readout:** Discovery-Rate / funktionale Tiefe ON vs. OFF.
- Ausgabe als Skript unter `docs/research/causal-model-proto/` — nicht in die Matrix verdrahtet.

## Tests

- `tests/test_regression_golden.py` grün bei Flag OFF (hart).
- `tests/systems/test_causal_model.py`: (a) Modul-Unit: Modell lernt (Fehler ↓ auf
  wiederholtem Paar); (b) Reproduzierbarkeit bei festem Seed; (c) Transfer schlägt
  Baseline auf einem synthetischen Spielzeug-`combine`; (d) OFF = kein Agent-State/kein
  globaler RNG-Konsum (lazy-init wird nicht betreten).

## Branch & Grenzen

Branch `feat/research-causal-model-proto` aus `feat/infra-research-umwelt-m5`.
Bewusst draußen: M1–M6-Matrix, Publikationsmetrik, Baseline/Kill/M5-Gating.
