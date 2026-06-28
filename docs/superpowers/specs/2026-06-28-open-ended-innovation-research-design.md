# Forschungsdesign — Open-ended, sozial getriebene Innovation in *Artificial Society*

- **Datum:** 2026-06-28
- **Status:** Entwurf zur Abnahme (gehärtet durch adversariales Reviewer-Panel + Code-Machbarkeitsprüfung)
- **Zieltyp:** Peer-reviewed Paper (KI-Forschung)
- **Primäres Venue (MVP):** ALOE @ ICLR / CoLLAs (Workshop) · **Archival:** *Artificial Life* Journal · ggf. NeurIPS/ICLR MARL-Track
- **Repo:** `artificial-society/` (deterministische, reproduzierbare Multi-Agenten-Emergenz-Simulation)

> **Wichtigster Kontext für jede:n Leser:in dieses Dokuments.** Eine erste, naive Fassung
> dieses Designs ("wir zeigen open-ended Innovation, getrieben von sozialer Transmission")
> wurde durch ein adversariales 3-Personen-Reviewer-Panel geprüft, das den Code
> *instrumentiert* hat. Ergebnis: die naive Fassung wäre **abgelehnt** worden, weil ihre
> Kernsignale durch einen **nicht-lernenden Zufalls-Rekombinierer** reproduzierbar sind.
> Dieses Dokument ist die *gehärtete* Fassung. Abschnitt 3 dokumentiert die fatalen Befunde,
> damit nachvollziehbar ist, warum das Design so aussieht, wie es aussieht.

---

## 1. Kernbeitrag (titelgebend)

> *Eine konfound-kontrollierte Methodik, um **echte gelernte/kulturell übertragene** open-ended
> Innovation in Multi-Agenten-Welten von **Artefakten des Rekombinations-Operators** zu trennen —
> angewandt auf Artificial Society. Die zentrale empirische Frage: Erzeugt **gelernte soziale
> Transmission** kumulative Innovation, die einen **rechen- und datengematchten Zufalls-/Frozen-Null**
> mit CI-Trennung übertrifft, und gibt es eine **kritische Transmissionstreue** auf dem
> kulturellen (horizontalen) Kanal, unterhalb derer der Vorsprung kollabiert?*

**KI-Relevanz.** Liefert (a) ein wiederverwendbares Mess- und Kontroll-Protokoll für
"open-ended learning"-Claims (gegen das verbreitete Problem, dass Umgebungsdynamik als
Lernfortschritt fehlgedeutet wird) und (b) einen empirischen Testbed-Befund über die
Bedingungen kultureller Akkumulation in neuronalen Multi-Agenten-Populationen.

**Warum diese Rahmung stärker ist als "schau, Emergenz!".** Sie ist *durch Konstruktion*
review-fest: wir erheben die Konfounds selbst und kontrollieren sie. Beide Ausgänge sind
publizierbar (siehe §11 Kill-Kriterien): Schlägt gelernte Transmission den Null → positives
"open-ended-learning"-Resultat. Schlägt sie ihn nicht → methodisches/warnendes Resultat
("naive Emergenz-Metriken sind durch Rekombinations-Operatoren konfundiert; hier ist die
Kontrolle"). Wir registrieren beide vorab.

---

## 2. Forschungsfragen & Hypothesen (null-relativ formuliert)

Alle DVs werden als **Delta gegenüber einem gematchten Null-Modell** mit Konfidenzintervallen
berichtet. Ein Effekt zählt nur, wenn er das Null-Band verlässt.

- **H1 — Gelernte Open-Endedness (über Null):** Funktionale Innovationskomplexität *und*
  -diversität wachsen unter gelernten/sozialen Agenten **anhaltend und adaptiv** und übertreffen
  den rechengematchten Zufalls-/Frozen-Null mit CI-Trennung (MODES persistence-gefiltert +
  Bedau mit korrektem Neutral-Shadow). *Falsifizierbar: kein CI-getrennter Vorsprung ⇒ H1 verworfen.*
- **H2 — Soziales Autocurriculum:** Soziale Populationen erreichen funktionale Komplexität,
  die **rechen- UND datengematchte asoziale** Agenten nicht erreichen — und zwar isolierbar auf
  den **Transmissionskanal** (nicht auf bloßes Gewichts-Kopieren oder zusätzliche Belohnung/Population).
- **H3 — Ratchet-Schwelle auf dem kulturellen Kanal:** Es existiert eine kritische
  **horizontale** Transmissionstreue θ\*, oberhalb akkumuliert Komplexität, unterhalb kollabiert
  sie **auf das asoziale Null-Niveau** (nicht nur "weniger").
- **H4 — Demografie/Topologie:** Populationsgröße und Netzwerk-*Struktur* modulieren erhaltbare
  Komplexität — **nicht monoton** (wir testen explizit den Derex-&-Boyd-/SAPIENS-Befund, dass
  volle Vernetzung nicht optimal sein muss).
- **H-Kontrolle (Default, nicht Ablation):** Die Effekte überleben **`NEW_DISCOVERY_BONUS=0`**,
  d.h. ohne direkte Belohnung für Beobachter-Neuheit. *Verschwindet Open-Endedness ohne
  Neuheits-Bezahlung, wird das berichtet.*

---

## 3. Der entscheidende Befund: warum das naive Design scheitert

Konvergent über alle drei Reviewer (mit Code-Belegen):

1. **Zufalls-Rekombinierer reproduziert das Headline-Signal.** Ein rein zufälliger Agent
   (kein PPO, kein Gedächtnis, kein soziales Lernen) erreicht **~9.769 distinkte "Entdeckungen"
   in 20k Kombinationen** (weiter steigend) und **Abhängigkeits-DAG-Tiefe 22**. "Anhaltende
   Diversität und kumulative Tiefe" ist also Default-Verhalten des Operators, **kein** Autocurriculum.
2. **Invention ist umgebungs-, nicht policy-gekoppelt.** In `agent.py` feuert Invention auf
   festem Zeitplan (`tick % 3 == 0 and random.random() < inv_prob`, `inv_prob =
   INVENTION_BASE_PROB 0.08 + INVENTION_CURIOSITY_MULT 0.05*curiosity`); Inputs via
   `random.choice(available)` (`invention.py:126-129`); Output ist `combine_vectors()`
   (`materials.py:430`). Die PPO-Policy wählt **nicht**, was kombiniert wird. ⇒ Die DAG-Tiefe ist
   Eigenschaft der fixen Reaktionsalgebra + eines Random-Walks, **keine gelernte Fähigkeit**.
   Das kollabiert H1 und die Rahmung "open-ended *learning*".
3. **"Kumulative Tiefe" ist ein Zähl-Artefakt** des 0.08-Dedup-Gitters (`materials.py:243`) +
   rekursiver Rückkopplung. Funktionale Komplexität (nichtnull-Property-Dimensionen ~5.9→~8/12,
   dann flach; Vektornorm ~1.5 Plateau) sättigt, während die DAG-Tiefe weiter auf 22 wächst.
   Kettenlänge ≠ "auf Schultern stehen".
4. **`NEW_DISCOVERY_BONUS=4.5` (`invention.py:85`) ist eine Neuheits-Pumpe.** Agenten werden
   direkt für beobachter-neue Artefakte bezahlt, unabhängig vom adaptiven Wert — genau die
   neutrale Drift, die MODES/Bedau ausschließen sollen. (Zusätzlich +1.0/+0.5 Inline-Rewards +
   Curiosity-Term konfundieren die DV.)
5. **θ\* misst den falschen Knopf.** `INHERIT_FIDELITY` (`simulation.py:35`) steuert nur die
   **vertikale** (Eltern→Kind) Transmission; H2s "Motor" ist **horizontales** soziales Lernen mit
   separater, hartkodierter Treue (`FIDELITY_BASE=0.72` + Trust, `social_learning.py:26-27`).
   Unter-θ\*-Läufe können über den horizontalen Kanal weiter ratchen.
6. **θ\* droht tautologisch zu sein.** Niedrige Treue = korrupte Rezepte = weniger tiefe Rezepte
   ist fast definitorisch. Schlimmer: unter Treue wird die Aktion/Material **durch Zufall ersetzt**
   (`culture.py:78-92`) — also aktive Fehlinformations-Injektion, nicht bloß "nicht kopiert".
7. **Seeds sind NICHT gematcht.** Nur das globale `random`-Modul wird geseedet (`rng.py`); überall
   `random.random()`. Soziales Lernen aus / Low-Fidelity-Zweig ändert die Anzahl der Draws ⇒
   **desynchronisiert den gesamten Downstream-RNG-Strom**. "Seed i, sozial AN" und "Seed i, sozial
   AUS" sind unverwandte Welten — die gepaarte Varianzreduktion ist zerstört.
8. **Asoziale Ablation nicht rechen-/datengematcht** und konfundiert drei Kanäle: Rezept-Transmission
   + Gewichts-Imitation (`imitate_from`, `brain.py:186`) + Kind-Vererbung (`INHERIT_FIDELITY`) +
   ein Reward-Strom. Soziale Agenten bekommen strikt mehr Lernsignal, mehr effektive Erfahrung und
   überleben häufiger (mehr effektive Population) — jede Differenz könnte ein Budget-Artefakt sein.

**Konsequenz für das Design:** Die folgenden Punkte sind keine Verbesserungen, sondern
**Existenzbedingungen** des Papers (§8 Mechanismus-Voraussetzungen) und müssen umgesetzt sein,
bevor die ersten Experimente zählen.

---

## 4. Die Null-Baselines (zentral, nicht Fußnote)

Jede Bedingung wird gegen **rechen- und datengematchte** Nulls berichtet:

| Null | Definition | Schließt aus |
|---|---|---|
| **Random-Recombiner** | identische Inventions-Maschinerie, aber Inputs/Aktion uniform zufällig; keine Policy, kein Gedächtnis, kein soziales Lernen | "Operator erzeugt das Signal" |
| **Frozen-Brain** | trainiertes/initialisiertes Netz, **keine** Gradienten-Updates; gleiche Env-Schritte | "Lernen trägt nichts bei" |
| **Random-Action** | Policy durch uniforme Aktionsverteilung ersetzt | "Verhalten egal" |
| **Neutral-Shadow (Bedau)** | prinzipielles Neutralmodell *desselben generativen Prozesses* mit ausgeschalteter Selektion (NICHT der Random-Agent — der ändert den generativen Prozess) | "adaptiv vs. Drift" |

**Pre-registrierte Verwerfungsregel:** Ein H1/H2/H3-Effekt zählt nur, wenn er den jeweils
relevanten Null mit getrennten Bootstrap-CIs übertrifft (gleiche #Kombinationsversuche/Tick,
gleiches Seed-Protokoll, gleiche Gradienten-Updates, gleiche Reward-Magnitude).

---

## 5. Experiment-Matrix

**Unabhängige Variablen (sauber parametrisiert, siehe §8):**

| IV | Werte | Testet | Konfound-Kontrolle |
|---|---|---|---|
| Agententyp | sozial-lernend / asozial-datengematcht / frozen / random-action / random-recombiner | H1, H2 | Budget-Matching (§9) |
| Soziale Kanäle | Rezept-Transmission / Gewichts-Imitation / Kind-Vererbung — **einzeln** faktoriell | H2 | isoliert je Kanal |
| Transmissionstreue θ (horizontal) | dichtes Gitter 0.0→1.0, dicht nahe Übergang | H3 | je Kanal separat + Joint-Sweep |
| Degradationsmodell | drop / mutate / partial-resolve | H3 | θ\*-Invarianz-Check |
| Populationsgröße & Netzwerk-Topologie | klein/mittel/groß × {voll, fragmentiert, dynamisch} | H4 | finite-size scaling |
| `NEW_DISCOVERY_BONUS` | **0 (Default)** / 4.5 (Sensitivität) | H-Kontrolle | DV-Reward-Entkopplung |
| Dedup/named-Schwelle | 0.08/0.10 ± Sensitivitäts-Sweep | Robustheit | §10 |

**Abhängige Variablen (Metriken in §6):** funktionale Komplexität, irreduzible Abhängigkeitstiefe,
persistence-gefilterte Diversität, MODES (change/novelty/complexity/ecology), Bedau
adaptive-vs-neutral, Lernkurven (Return, Rediscovery-Rate, Time-to-depth-k), Fitness-Proxies
(Population, Energie — nur nach Energie-Audit §8).

---

## 6. Metriken (code-verankert, artefakt-resistent)

1. **Funktionale Komplexität** (ersetzt rohe DAG-Tiefe): minimale Anzahl distinkter, *nicht-seed*
   Intermediär-Materialien, die für einen Fitness-/Bedarfs-Gewinn **erforderlich** sind, der aus
   Seeds allein nicht erreichbar ist (irreduzible Abhängigkeitstiefe). **Validierung per Knock-out:**
   entferne ein Intermediär ⇒ Kind unerreichbar UND Fitness fällt. Quelle: `DISCOVERY_REGISTRY.entries`
   (Rezept-Tupel mit `mat_XXXX`-Eltern; topologisch sortieren, Zyklen abfangen).
2. **Diversität** mit Persistenz-/Adoptionsfilter (eine Entdeckung zählt erst, wenn sie von ≥k
   Agenten über ≥t Ticks genutzt wird) — gegen die 0.08-Gitter-Proliferation.
3. **MODES** (Dolson et al. 2019): change/novelty/complexity/ecology potential, persistence-gefiltert.
4. **Bedau evolutionary activity** mit **explizit konstruiertem Neutral-Shadow** (Selektion aus,
   gleicher generativer Prozess; Lattice-Statistik des 0.08-Gitters nachbilden) — adaptive vs. neutrale Aktivität.
5. **Lernnachweis:** Lernkurven (Return, Rediscovery-Rate, Time-to-first-depth-k) müssen mit
   Training steigen; Ablation des PPO-Updates muss die Innovationskurve **abflachen**.
6. **Sensitivitätskurven** für alle Beobachter-Parameter (Dedup 0.08, named 0.10, Persistenz-k/t).

> Hinweis zur Messbarkeit: `causal_memory` hat Kapazität 32 mit Eviction (`culture.py:32-44`) —
> für Messfenster Kapazität erhöhen/Eviction aussetzen oder nur Populations-Aggregate
> (`CultureTracker`) verwenden. `StatisticsTracker` cappt bei 200 Ticks (`statistics.py:38-42`) —
> muss für Plateau-Tests aufgehoben werden (§8 P1).

---

## 7. (zusammengeführt in §6/§8)

---

## 8. Mechanismus-Voraussetzungen & Engineering-Backbone

Reihenfolge = kritischer Pfad. **P-A bis P-C sind Existenzbedingungen** (ohne sie kein valider Claim).

- **P-A — Policy-Kopplung der Invention.** Material-/Rezeptwahl durch die gelernte Policy routen
  (Aktionsraum erweitert um Wahl der Inputs+Aktion, oder World-Model/Curiosity-Head schlägt
  Kombinationen vor), sodass *was* erfunden wird Funktion gelernter Repräsentationen ist — nicht
  `random.choice`. Plus Lernkurven + Frozen/Random-Action-Baselines. *(Ohne P-A ist die
  AI-learning-Behauptung tot.)*
- **P-B — RNG-Substreams je Subsystem.** `random.Random`/`numpy.Generator` mit
  `SeedSequence.spawn` pro Subsystem (Bewegung, Invention, soziales Lernen, Reproduktion, Welt).
  AUS-Bedingung konsumiert gleich viele Draws im geteilten Strom (oder Strom isoliert).
  **Determinismus-Test:** Forage-/Bewegungs-Strom byte-identisch sozial-AN vs -AUS. Erst dann sind
  gepaarte Seeds gültig ("matched-seed contract").
- **P-C — Vereinheitlichte/parametrische Transmissionstreue.** Beide `receive_transmitted`-Call-Sites
  (vertikal `simulation.py`; horizontal `social_learning.py:75,84`) über eine konfigurierbare
  `TRANSMISSION_FIDELITY` führen; θ\* pre-registriert auf dem **horizontalen** Kanal definieren;
  jeden Kanal dokumentieren + Joint-Sweep. Nicht-Hot-Parametrisierungspfad (Config/Kwarg statt
  Edit an `simulation.py:35`).
- **P-D — Saubere, kanal-getrennte Asozial-Schalter** + datengematchter Asozial-Control
  (z.B. Self-Play/Replay gibt isolierten Agenten gleich viele effektive Erfahrungen).
  `campfire_knowledge_sharing` (`social_learning.py:104`, derzeit dormant) bei Bedarf verdrahten.
- **P-E — Reward-Entkopplung.** `NEW_DISCOVERY_BONUS=0` als Default; Inline-+1.0/+0.5- und
  Curiosity-Terme dokumentieren/optional ausschalten; "essentials"/Legacy-Rezepte als
  `LEGACY_RECIPES`-Set + Check-Funktion zentralisieren (derzeit über `materials.py`+`invention.py` verstreut).
- **P-F — Energie-Erhaltungs-Invariante + Audit.** Σ(Agenten-Energie)+Σ(Zell-Food) als
  geschlossenes Budget; Regrowth/Geburten/Carcass als Quellen/Senken auditieren; Conservation-Test.
  Phase 4 fixt bisher nur ein Leck (Forage debitiert 1:1, `agent.py:519-535`). Fitness/Bedau-Claims
  sind sonst Energie-Bilanz-Artefakte.
- **P0 — Experiment-Infrastruktur (nicht-hot, in `scripts/` o. neuem Modul):** Multi-Seed-Runner
  (Subprozess mit `PYTHONHASHSEED=0`), Sweep-Harness (Seeds×Bedingungen), Config-System für die
  IVs, **strukturierter Export** (CSV/Parquet, vor dem 200-Cap abgegriffen), Aggregation + CI-Plots.
- **P1 — Instrumentierung:** ungecappte volle Zeitreihen je Tick persistieren; Innovations-Lineage/
  funktionale Komplexität/gefilterte Diversität pro Tick loggen; `stats.update` im Harness ticken
  (Base-`step()` tickt es nicht).
- **P2 — Phase 5 mergen** (de-script; fertig+getestet auf `feat/systems-phase5-descript` +
  `core/phase5-inventory-essentials`; Merge-und-Golden-Regen, keine Blocker) — nötig für die
  Glaubwürdigkeit von H2/H-Kontrolle.
- **P3 — MODES/Bedau-Analysepaket** (offline über exportierte Daten; inkl. Neutral-Shadow-Konstruktion).

**Hot-File-/Determinismus-Disziplin:** Alle Knöpfe über Nicht-Hot-Seams; Verhaltensänderungen ⇒
Golden bewusst durch `core-lead` neu generieren (Roadmap §6).

---

## 9. Statistik & Pre-Registration

- **Gepaarte, gematchte Seeds** (erst nach P-B gültig); gleiche Env-Schritte, Gradienten-Updates,
  Wall-Clock/FLOPs, Reward-Magnitude über Arme.
- **Heavy-tailed/zero-inflation:** robuste Statistik (Median, Bootstrap-CIs, rank-basierte Tests),
  explizites **Extinktions-Handling** (informative Missingness modellieren, nicht wegmitteln);
  Stichprobe **n≥30, ggf. deutlich mehr** nach Power-Analyse aus einem Pilot.
- **θ\*-Kritikalität korrekt:** dichtes Gitter nahe Übergang; formaler Changepoint-Test (no-break vs
  one-break) **mit Bootstrap-CI auf θ\***; finite-size scaling über Populationsgrößen; Hysterese;
  **Theorie-Vorhersage** von θ\* aus einem Transmissions-Ketten-Modell (erwartetes Rezept-Überleben
  vs. Innovationsrate) und Abgleich.
- **Multiple-Comparison / garden-of-forking-paths:** alle DVs/IVs/Metrik-Varianten + primäre
  Hypothesen + Verwerfungsregeln **vorab registrieren**; Korrektur (z.B. Benjamini-Hochberg);
  Sensitivitäts- klar von konfirmatorischen Analysen trennen.

---

## 10. Bedrohungen der Validität & Gegenmaßnahmen

| # | Einwand (Reviewer) | Gegenmaßnahme im Design |
|---|---|---|
| F1 | Zufalls-Rekombinierer erzeugt H1-Signal | Random/Frozen-Null als **primäre** Baseline; alle DVs null-relativ (§4) |
| F2 | Invention nicht policy-gekoppelt | **P-A**: Policy-Kopplung + Lernkurven + PPO-Ablation flacht Kurve ab |
| F3 | DAG-Tiefe = Zähl-Artefakt | **Funktionale/irreduzible Komplexität** + Knock-out-Test (§6) |
| F4 | `NEW_DISCOVERY_BONUS` = Neuheits-Pumpe | **=0 als Default** (§5, P-E); Bonus nur als gelabelte Sensitivität |
| F5 | θ\* auf falschem Kanal | **P-C**: θ\* auf horizontalem Kanal, je-Kanal + Joint-Sweep |
| F6 | θ\* tautologisch / Noise-Injektion | Degradationsmodelle variieren; Kritikalität + Theorie-Vorhersage (§9) |
| F7 | Seeds nicht gematcht | **P-B**: RNG-Substreams + matched-seed contract + Determinismus-Test |
| F8 | Asozial-Ablation nicht budget-gematcht | **P-D**: Kanal-Trennung + daten-/rechengematchter Control |
| M1 | Bedau-Neutral-Shadow fehlt/unspezifiziert | **P3**: prinzipielles Neutralmodell, Lattice-Statistik nachgebildet |
| M2 | Welt zu arm ⇒ funktionales Plateau | Plateau als **legitimes Resultat** vorab erlaubt; ggf. Nicht-Stationarität als Erweiterung |
| M3 | 200-Tick-Cap verdeckt "sustained" | **P1**: ungecappte Zeitreihen |
| M4 | Globaler `random` + Singleton-Registry ⇒ Kontamination | **P-B** + Registry-Reset/Isolation je Run |
| M5 | Freie Beobachter-Parameter dominieren | Sensitivitäts-Sweeps 0.08/0.10/k/t (§6) |
| M6 | Energie nicht erhalten | **P-F**: Invariante + Audit + Test |
| M7 | n unterpowert / heavy-tailed | robuste Statistik + Power-Analyse + Extinktions-Handling (§9) |
| M8 | Multiple Comparisons | Pre-Registration + Korrektur (§9) |
| M9 | KI-Beitrag vs. Melting Pot/Crafter/… unklar | Beitrag = **Methodik/Kontroll-Protokoll** (nicht "noch eine Sim"); klare Abgrenzung zu *Artificial Generational Intelligence* (§16) |

---

## 11. Erfolgs- & Kill-Kriterien (vorab registriert)

- **Positiver Ausgang (Path A):** Gelernte/soziale Agenten übertreffen *alle* gematchten Nulls in
  funktionaler Komplexität mit getrennten CIs **und** es existiert ein θ\* auf dem kulturellen Kanal,
  unterhalb dessen der Vorsprung auf Asozial-Niveau kollabiert ⇒ "open-ended-learning"-Paper.
- **Negativer/methodischer Ausgang (Path B):** Kein CI-getrennter Vorsprung über Null und/oder
  funktionale Komplexität plateaut auf Null-Niveau ⇒ methodisches/warnendes Paper ("Rekombinations-
  Operator-Konfound + Kontroll-Protokoll; naive Emergenz-Claims sind nicht haltbar").
- **Kill-Kriterium für die zentrale These:** Bleibt der Lern-/Sozial-Vorsprung auch nach P-A–P-D
  innerhalb des Null-Bands, wird **kein** "Open-Endedness durch Lernen"-Claim erhoben — Resultat
  wird als Path B ehrlich berichtet.

---

## 12. Scope: MVP vs. Erweiterung

- **MVP (Workshop, ALOE/CoLLAs):** P-A–P-F + P0/P1 + H1+H2+H-Kontrolle; zwei Kernabbildungen —
  (a) funktionale Komplexität über Zeit, *sozial vs. asozial-gematcht vs. random/frozen* (n≥30, CIs);
  (b) θ\*-Sweep (horizontaler Kanal) mit Changepoint-CI. MODES/Bedau als Mess-Backbone.
- **Journal-Erweiterung (*Artificial Life*):** H3-Kritikalität voll (finite-size scaling, Hysterese,
  Theorie-Match), H4-Demografie/Topologie inkl. Derex-&-Boyd-Nicht-Monotonie, längere Läufe,
  mehrere Welt-Konfigurationen.

---

## 13. Open Science / Reproduzierbarkeit

Pre-Registration (OSF) der Hypothesen, Nulls, Metriken, Verwerfungsregeln · jede Abbildung
reproduzierbar aus *Config + Seed-Liste* · `PYTHONHASHSEED=0`-Protokoll dokumentiert ·
matched-seed contract als Test · Code+Daten+Notebooks released, **Zenodo-DOI** ·
"reproduce-all-figures"-Pipeline. Der bestehende Determinismus-Kontrakt ist Vorteil — *aber*
P-B ist Voraussetzung, sonst täuscht er Reproduzierbarkeit nur vor.

---

## 14. Roadmap & Abhängigkeiten — **Pilot-first (gewählt)**

**Stage 0 — Pilot & Entscheidungs-Gate** *(zuerst; entrisikt den teuren P-A-Umbau).*
- **Bauen (billig, nicht-hot wo möglich):** P-B (RNG-Substreams + matched-seed contract),
  die drei **Null-Baselines** (random-recombiner, frozen-brain, random-action), die
  **funktionale Komplexitätsmetrik** (§6.1) + Knock-out-Validierung, und ein **minimaler
  ungecappter Export** (Teilmenge P0/P1).
- **Pilotlauf:** gelernte/soziale Agenten vs. **random-recombiner**, rechen-/datengematcht,
  n≈10–15 Seeds, über genug Ticks, dass der Null sein Plateau zeigt.
- **Gate-Regel (vorab fixiert):** Übertrifft die *funktionale* Komplexität der gelernten/
  sozialen Agenten den Random-Recombiner-Null mit **getrennten Bootstrap-CIs**?
  - **Ja ⇒ Path A:** voller Mechanismus-Fix (P-A Policy-Kopplung + P-C/P-D/P-E + P-F + P2)
    lohnt sich → positiver "open-ended-learning"-Claim.
  - **Nein/grenzwertig ⇒** entweder **P-A nachrüsten und Pilot wiederholen** (Lernen war
    nur nicht policy-gekoppelt), **oder Path B** (Methodik/Kontroll-Paper) — datenbasiert.

```
Stage 0:  P-B ─→ Null-Baselines ─→ funktionale Metrik ─→ Pilot ─→ [GATE]
                                                                    │
              Ja ─→ Path A: P2 + Phase4 + P-F + P-A + P-C/P-D/P-E + P0/P1 ─→ MVP-Experimente ─→ P3 ─→ Paper (positiv)
              Nein ─→ P-A nachrüsten & Pilot↺   ODER   Path B: Methodik-Paper (Kontrollen + Null-Befund)
```

Stage 0 ist Tage–Wochen; Path A danach mehrwöchig bis -monatig (P-A = größter Einzelposten,
Agenten-/Action-Space-Umbau, berührt Hot-Files ⇒ `core-lead`, Golden-Regen).

---

## 15. (zusammengeführt in §14)

---

## 16. Verwandte Arbeiten & Referenzen

**Verifiziert (alle 12 bestätigt; Korrekturen eingearbeitet):**
- Hughes et al. 2024, *Position: Open-Endedness is Essential for Artificial Superhuman Intelligence*, ICML 2024 (PMLR v235), arXiv:2406.04268.
- Clune 2019, *AI-GAs: AI-generating algorithms…*, arXiv:1905.10985 (Manifest/Preprint).
- Wang et al. 2019, *POET*, arXiv:1901.01753 / GECCO 2019; **Enhanced POET** 2020, ICML 2020 (PMLR v119).
- Leibo et al. 2019, *Autocurricula and the Emergence of Innovation from Social Interaction: A Manifesto…*, arXiv:1903.00742.
- Leibo et al. 2021, *Scalable Evaluation of MARL with Melting Pot*, ICML 2021 (PMLR v139).
- Lazaridou & Baroni 2020, *Emergent Multi-Agent Communication in the Deep Learning Era*, arXiv:2006.02419 (Survey).
- Dolson, Vostinar, Wiser, Ofria 2019, *The MODES Toolbox*, *Artificial Life* 25(1):50–73.
- Bedau, Snyder & Packard 1998, *A Classification of Long-Term Evolutionary Dynamics*, Artificial Life VI (+ Bedau & Packard 1992) — Neutral-Shadow-Methodik.
- Lewis & Laland 2012, *Transmission fidelity is the key to the build-up of cumulative culture*, Phil. Trans. R. Soc. B 367:2171–2180.
- Henrich 2004, *Demography and cultural evolution… the Tasmanian case*, American Antiquity 69(2).
- Taylor et al. 2016, *Open-Ended Evolution: Perspectives from the OEE Workshop*, *Artificial Life* 22(3).
- Hafner 2021, *Crafter* (arXiv:2109.06780, ICLR 2022); Matthews et al. 2024, *Craftax*, ICML 2024.

**Neu hinzuzunehmen (vom Panel als fehlend markiert — Priorität oben):**
- **Cook, Cohen, Hughes, …, Foerster et al. 2024, *Artificial Generational Intelligence: Cultural Accumulation in RL*, NeurIPS 2024, arXiv:2406.00392** — *die* wichtigste Abgrenzungsarbeit (erste allgemeine RL-Modelle mit emergenter kultureller Akkumulation; in-context vs. in-weights Generationen). **Must-cite + must-distinguish.**
- Zhang/Lehman/Stanley/Clune 2023 *OMNI* (arXiv:2306.01711); Faldor et al. 2024 *OMNI-EPIC*, ICLR 2025 — Open-Endedness via Foundation Models.
- Nisioti et al. 2022 *SAPIENS* (arXiv:2206.05060) — Netzwerkstruktur formt Innovation in RL (H4-Präzedenz).
- Channon, Bedau, Packard, Taylor 2024 — OEE-Special-Issue / *On the Open-Endedness of Detecting Open-Endedness*, *Artificial Life* 30(3).
- UED/Autocurricula: Parker-Holder et al. 2022 *ACCEL* (ICML); Jiang et al. 2023 *minimax* (arXiv:2311.12716); Dennis et al. 2020 *PAIRED*.
- Perez et al. 2024 — *Cultural evolution in populations of LLMs* (Transmissions-/Ratchet-Vergleichsklasse).
- Lehman & Stanley 2011 *Novelty Search*; Mouret & Clune 2015 *MAP-Elites* (QD-Fundament für "Diversität UND Komplexität").
- Derex & Boyd 2016 (PNAS) — partielle Vernetzung erhält kulturelle Diversität (nuanciert H4: voll ≠ optimal).

---

## 17. Offene Entscheidungen (für die nächste Runde)

1. **Scope-Commitment:** ✅ **ENTSCHIEDEN — Pilot-first (Stage 0, §14).** Erst die billigen
   Kontrollen (P-B + Null-Baselines + funktionale Metrik) + ein Pilotlauf; das Gate-Ergebnis
   entscheidet Path A (voller Mechanismus-Fix) vs. Path B (Methodik-Paper) datenbasiert.
2. **Venue-Reihenfolge** bestätigen (ALOE/CoLLAs zuerst, *Artificial Life* als Erweiterung).
3. **Solo vs. Team/Lanes** (das Repo ist auf parallele Entwicklung ausgelegt — passt zum P0/P1/P-A-Split).
