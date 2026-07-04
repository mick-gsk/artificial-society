# M1-Diagnose-Batterie — Ergebnis & Empfehlungen (2026-07-04)

Auswertung der 31 Läufe aus `battery_results/` (9 Arme, 5000 Ticks, physics_v2, 40×30, Start-Pop 30;
Spec: `2026-07-04-m1-diagnose-batterie.md`). Kernaussage vorweg: **Zwei der neun Arme haben nie
stattgefunden** (nosocial-Patches waren wirkungslos, D1==A1 und A3==A2 byte-identisch), **der
Sozial-/Kultur-Kanal ist im v2-Pfad komplett tot** (toter Nearby-Cache), und **die Population läuft
als Respawn-Mühle** (Zufalls-Neuspawns statt Geburten tragen die Population). Damit sind H3
unmessbar, Selektion und Kulturkontinuität strukturell ausgehebelt, und der vorläufige
H4-Befund („A1 ≈ A2") ist mit dieser Batterie nicht entscheidbar. Vor jeder weiteren Messung
müssen die Konfounde K1–K3 beseitigt werden.

## (a) Ergebnisübersicht

Finale Werte bei Tick 5000, Median über Seeds (Spannweite in Klammern). `tcr` = tool_cut_ratio.

| Arm | n | tcr | cuts_with_tool | discoveries | fragments | pop | mean_energy |
|---|---|---|---|---|---|---|---|
| A1 learn | 5 | 0.034 (0.002–0.108) | 97 (7–289) | 14 (11–15) | 39 (3–315) | 11 (9–12) | 65 (17–73) |
| A2 nolearn | 5 | 0.006 (0.000–0.058) | 17 (0–134) | 12 (11–14) | 5 (2–112) | 10 (8–13) | 65 (28–82) |
| A3 nolearn-nosocial | 3 | 0.004 | 12 | 12 | 2 | 10 | 65 | 
| B1 gamma-short | 3 | 0.027 (0.000–0.084) | 69 (0–239) | 12 | 68 | 10 | — |
| C1 curio-off | 3 | 0.000 (0.000–0.055) | 0 (0–129) | 13 | 6 | 10 | — |
| C2 curio-high | 3 | 0.001 (0.000–0.009) | 2 (0–23) | 10 | 2 | 13 | — |
| D1 nosocial | 3 | 0.034 | 97 | 15 | 39 | 11 | — |
| E1 entropy-high | 3 | 0.007 (0.000–0.065) | 20 (0–167) | 13 | 16 | 9 | — |
| F1 verb-bias-high | 3 | 0.027 (0.000–0.054) | 91 (0–136) | 13 | 44 | 8 | — |

**A3- und D1-Zeilen sind KEINE Messungen** — sie sind Byte-Kopien von A2 bzw. A1 (Prüfung 1).

**Welt-Seed dominiert den Arm-Effekt.** Seed 41 ist in praktisch jedem Arm ≈ 0 (tcr 0.000–0.004
über alle 9 Arme), Seed 42 in praktisch jedem Arm hoch (tcr 0.027–0.065, einzige Ausnahme
C2 = 0.009). Die Varianz zwischen Welt-Seeds ist größer als jede Arm-Differenz; mit n=3 je
Varianten-Arm ist damit fast keine „deutlich über/unter"-Aussage aus Spec §3 belegbar
(Gültigkeits-Schwelle M-1: IQRs überlappen überall).

## (b) Die fünf Pflicht-Prüfungen

### 1. Identitäts-Check: D1==A1 und A3==A2 — die nosocial-Arme haben nie stattgefunden

Record-weiser Vergleich der JSONL (alle snap- und final-Records, `walltime_s` ausgenommen) auf den
geteilten Seeds 41/42/43: **D1 ist auf allen drei Seeds byte-identisch zu A1, A3 byte-identisch zu
A2** — bis in jede `verbs_fired`-Zählung und jede `mean_energy`-Nachkommastelle. Der
`social_learning_step`-No-op-Patch hatte exakt null Effekt, nicht einmal auf den RNG-Strom.

Ursache im Quellcode — **der Nearby-Cache ist tot, `social_learning_step` returned immer vor dem
ersten RNG-Zug:**

- `artificial_society/agents/agent.py:203-204` — `ensure_fields` initialisiert
  `_cached_nearby_agents = []` und `_cached_nearby_radius = 2`.
- `artificial_society/agents/agent.py:452-455` — `_nearby_cached` returned den Cache, sobald
  `cached is not None and cached_radius == radius`. Die leere Init-Liste ist nicht `None` und der
  Radius passt (2) → **jeder Radius-2-Aufruf liefert für immer `[]`**. Eine Invalidierung pro Tick
  existiert nirgends (`grep '_cached_nearby_agents = None'` → kein Treffer im Paket).
- `artificial_society/systems/social_learning.py:53-55` — `social_learning_step` holt
  `agent._nearby_cached(agents, 2)`, bekommt `[]` und returned `0.0`, **bevor** irgendein
  `random.random()` gezogen oder Zustand mutiert wird. Deshalb ist der Patch RNG-neutral und die
  Läufe sind byte-identisch.

Mitbetroffen (gleicher toter Cache): `systems/economy.py:28` (Trade),
`agents/agent.py:804` und `agents/agent.py:1347` (soziale Wahrnehmung / local_features).
Das ist exakt der Audit-Befund „toter Nearby-Cache schaltet Social Learning/Trade/ToM ab" vom
Branch `core/audit-fixes` — **dieser Fix ist in der Lineage dieses Worktrees
(`feat/infra-m1-pilot` @ 9e10eb6, basierend auf main @ 356836f) nie angekommen.**

**Konsequenz (eigenständiger Hauptbefund):** Der Sozial-/Kultur-Kanal ist im v2-Pfad substanzlos —
kein Kausal-Sequenz-Transfer, kein Lehren, keine Gewichts-Imitation, kein Trade, keine
Trust-Bildung über Beobachtung. Für **Meilenstein-1-Kriterium 2 („kulturelle Weitergabe")** heißt
das: Es gibt aktuell **keinen funktionierenden Übertragungskanal, der etwas weitergeben könnte** —
unabhängig davon, ob Agenten je etwas Weitergabewürdiges lernen. H3 ist mit dieser Batterie
prinzipiell unmessbar (die Arme D1/A3 waren Placebos).

### 2. Respawn-Floor: die Population ist eine Respawn-Mühle aus Zufallshirnen

`artificial_society/simulation.py:38-39` — `MIN_POPULATION = 8`, `RESPAWN_COUNT = 6`;
`simulation.py:639-640` — jeder Tick mit `len(agents) < 8` triggert `emergency_respawn()`;
`simulation.py:304-310` — der Respawn baut `Agent.spawn_random(...)` + `attach_body(a)`, und
`attach_body` (`agents/agent.py`, Plan-3b-Pfad) baut ein **frisches v2-Brain mit
Zufalls-Initialisierung** (Lamarck-Vererbung im v2-Pfad per User-Entscheidung entfernt) — jeder
Respawn ist ein kognitiver Totalreset.

Befund aus den Zeitreihen (alle 31 Läufe):

- Verlauf überall gleich: 30 → kurzer Geburten-Overshoot (43–51 bei Tick 250–500) → Kollaps auf
  ~10 bis Tick 750–1000 → Dauerschwingen im Band 8–14. **Minimum-Pop = 8–9 in jedem Lauf** (exakt
  der MIN_POPULATION-Boden); 75–90 % aller Snapshots liegen bei pop ≤ 14 (= Boden 8 + ein
  Respawn-Batch 6).
- Umschlagsrate (Untergrenze aus Snap-zu-Snap-Δpop; wahre Churn höher, weil 250-Tick-Fenster
  interne Zyklen verdecken): **≥ 25–46 neue Agenten je Lauf nach dem Start**, bei nur **0–8
  sichtbaren Geburten** (`births` zählt Kinder noch lebender Eltern — Undercount, aber die
  Größenordnung stimmt: der Overshoot-Peak ist bei Tick 1000 restlos weggestorben). Also sind
  **≥ 70–100 % aller Nachrücker Not-Respawns mit Zufallshirn**; die stehende Population (~10)
  wird über 5000 Ticks **mindestens 3–5× komplett umgeschlagen**.
- Todesrate ≥ ~55 Tode / 5000 Ticks bei ~10–12 Lebenden → mittlere Lebensdauer grob **≤ ~1000
  Ticks** (obere Schranke; real kürzer).

**Konsequenz für ALLE Arme:** (i) Es gibt keine generationsübergreifende Selektion — wer stirbt,
wird durch Zufall ersetzt, nicht durch Nachkommen Erfolgreicher. (ii) Kulturkontinuität ist
strukturell ausgehebelt, selbst wenn der Kanal aus Prüfung 1 funktionierte: Träger sterben, Ersatz
kommt wissensfrei. (iii) Individuelles Lernen (A1) hat pro Agent nur ein ≤ ~1000-Tick-Fenster und
beginnt danach wieder bei Zufalls-Gewichten — die Batterie vergleicht also nicht „gelernte vs.
zufällige Population", sondern „Population mit kurzen Lernepisoden vs. Zufallspopulation".
Der Dauerzustand aller Arme ist näher an einer Random-Null mit Lern-Flackern als an einer
lernenden Gesellschaft.

### 3. Lernkurven: der Anstieg in der zweiten Hälfte ist kein sauberes Lernsignal

Per-Intervall-Inkremente aus den snap-Zeitreihen, Hälfte 1 (Tick 0–2500) vs. Hälfte 2 (2500–5000),
`cuts_with_tool` (Median | per Seed):

| Arm | H1→H2 (Median) | Hälfte 2 per Seed |
|---|---|---|
| A1 learn | 12 → 80 | 11, 92, 80, 108, 7 (Seeds 41–45) |
| A2 nolearn | 12 → 15 | 0, 72, 0, 40, 15 |
| B1 gamma-short | 6 → 63 | 0, 63, 139 |
| F1 verb-bias | 21 → 40 | 0, 115, 40 |
| C2 curio-high | 0 → 2 | 0, 11, 2 |

- **Pro Lernsignal:** A1s Median steigt 12→80 (tcr letzter 5 Snaps 0.036 vs. A2 0.006); im
  seed-gepaarten Vergleich (gleicher Welt-Seed!) liegt A1 in 4 von 5 Paaren vor A2.
- **Contra:** (i) Auch das **eingefrorene Zufallshirn A2 steigt** in 3/5 Seeds in Hälfte 2 (bis 72
  Tool-Cuts) — ein Anstieg ohne jedes Lernen. Ein Teil des „Lernsignals" ist also Welt-Ratchet
  (Fragmente/scharfe Objekte akkumulieren → spätere Tool-Cuts werden wahrscheinlicher, egal wie
  dumm die Policy ist) plus Glücks-Episoden einzelner Agenten. (ii) B1 mit γ=0.80 erreicht
  A1-Niveau — wäre der Anstieg gelernte lange Kredit-Ketten, müsste γ=0.80 ihn zerstören.
  (iii) Die per-Seed-Streuung (7–108) ist größer als der Median-Unterschied; IQRs überlappen.
- `discoveries` ist als Lernmetrik unbrauchbar: 10–14 in allen Armen, ~alles in Hälfte 1
  (Hälfte-2-Median 0–1) — ein kleiner endlicher Entdeckungsraum, den auch Zufallsrauschen in
  2500 Ticks absättigt.

**Fazit:** Ein echtes, von der Welt-Drift trennbares Lernsignal ist mit n=5, dieser
Seed-Dominanz und ohne Respawn-/Alters-Instrumentierung **nicht belegbar** — aber auch nicht
widerlegt (4/5 gepaarte Seeds pro A1 sind suggestiv, Sign-Test p≈0.19).

### 4. Energie-/Verhaltens-Plausibilität: chronische Knappheit, ungerichtetes Fuchteln

- **Energie:** Median über alle Snapshots ~66–70 bei `MAX_ENERGY = 240` (`agents/agent.py:68`),
  Minima bis ~11 — die Agenten leben chronisch bei ~28 % des Maximums, nahe der Sterbeschwelle.
  Das erklärt die Respawn-Mühle aus Prüfung 2 und frisst als permanenter Überlebensdruck jede
  Explorations-Marge (Reward wird vom Essens-/Sterbe-Term dominiert).
- **Verben:** Der v2-Aktionsraum hat 5 Verben (grasp, eat, cut, release, strike). Verteilung final
  (A1 s41): grasp 26 %, eat 25 %, cut 22 %, strike 13 %, release 14 % — gegen 20 % uniform ist das
  **nahezu Gleichverteilung mit leichtem grasp/eat-Überhang**, in A2 (Zufallshirn) fast identisch
  (27/26/19/13/15). Es gibt keine erkennbare gezielte Manipulationsstruktur.
- **Cut-Volumen vs. Ertrag:** `cut` feuert 5200–7900× je Lauf, verbindet in 34–45 % der Fälle
  (~2300–3000 Schnitte) — davon sind **0–289 mit Werkzeug** (tcr ≤ 0.108). Die ~7000 cut-Feuerungen
  sind Rauschen einer Policy, die das Verb zieht, ohne Werkzeug-Kontext herzustellen; die
  Kette „Werkzeug greifen → damit schneiden" entsteht fast nie.

### 5. Entscheidungsbaum neu bewertet — was die Daten wirklich tragen

- **Schritt 0 (billige Lever) — bestätigt negativ, mit Verschärfung:** E1/F1 heben nichts über A1;
  **C2 (3× Curiosity) drückt Tool-Cuts fast auf null** (2 vs. 97 Median). Intrinsik ist nicht
  unterdosiert; Überdosierung schadet aktiv (Novelty-Jagd verdrängt wiederholtes Schneiden). H2 in
  beiden Richtungen unattraktiv.
- **H1 (Kredit-Horizont): nicht gestützt.** B1 (γ=0.80) ≈ A1 — der Horizont-Knob hat keinen
  messbaren Effekt. Entweder trägt langer Kredit die Kette ohnehin nicht (konsistent mit
  Lebensdauer ≤ ~1000 Ticks und Welt-Ratchet aus Prüfung 3), oder der Effekt ist unter der
  Seed-Varianz begraben. In keinem Fall rechtfertigen die Daten jetzt einen teuren
  Kredit-Backfill-Bau.
- **H3 (Kultur): unmessbar, aber der Kanal ist nachweislich tot** (Prüfung 1). Das ist stärker als
  jedes Batterie-Ergebnis: „D1 ≈ A1 → Kultur deprioritisieren" wäre die falsche Lesart — richtig
  ist „Kultur-Kanal existiert nicht, erst reparieren, dann messen".
- **H4 (Lern-Kopplung): bleibt die führende Hypothese, ist aber NICHT sauber belegt.** A1 ≈ A2 auf
  den finalen IQRs, aber der gepaarte Verlaufs-Vergleich deutet einen kleinen A1-Vorsprung an
  (Prüfung 3). Die Batterie kann H4 nicht von den Konfounden trennen: Respawn-Mühle (Lernfenster
  ≤ ~1000 Ticks, ständiger Reset), chronische Energie-Knappheit (Reward-Dominanz des Überlebens)
  und Welt-Seed-Dominanz (Seed erklärt mehr Varianz als jeder Arm).
- **Ungültig/erledigt:** D1, A3 (Placebos); `discoveries` als Diskriminator (saturiert überall).

**Konfounde, die VOR einem erneuten Batterielauf beseitigt sein müssen:** (K1) toter Nearby-Cache,
(K2) Respawn-Mühle bzw. mindestens ihre vollständige Instrumentierung, (K3) fehlende
Demografie-/Konzentrations-Metriken + zu wenig Seeds gegen die Welt-Seed-Dominanz.

## (c) Priorisierte Empfehlungen — „Was als Nächstes bauen"

### Pflicht: Konfounde beseitigen (vor jeder neuen Messung)

**K1 — Nearby-Cache-Fix in die v2-Lineage bringen (höchste Priorität).**
- *Evidenz:* Prüfung 1 — D1/A3 byte-identisch zu A1/A2; `agents/agent.py:452-455` + `:203-204`
  liefern für Radius 2 für immer `[]`; Social Learning, Trade, soziale Wahrnehmung tot.
- *Erwarteter Effekt:* Kultur-Kanal (M1-Kriterium 2) existiert überhaupt erst; D1/A3 werden
  messbar; Trust/ToM/Trade wieder aktiv. Achtung: Verhalten ALLER künftigen Läufe ändert sich —
  neue Baseline nötig.
- *Umfang:* Klein. Fix existiert bereits auf `core/audit-fixes` (Befund „toter Nearby-Cache") —
  cherry-picken oder minimal neu: Cache pro Tick invalidieren (z. B. `_cached_nearby_agents = None`
  am Tick-Anfang) statt Init auf `[]`.
- *Dateien/Lane:* `artificial_society/agents/agent.py` — **HOT-File, core-lead seriell.**

**K2 — Respawn-Mühle entschärfen ODER vollständig instrumentieren.**
- *Evidenz:* Prüfung 2 — min-pop = 8 überall, ≥ 70–100 % der Nachrücker sind Zufalls-Respawns,
  Population schlägt ≥ 3–5× um; Selektion + Kulturkontinuität strukturell ausgehebelt
  (`simulation.py:38-39,304-310,639-640`).
- *Erwarteter Effekt:* Erst damit misst eine Batterie „lernende Gesellschaft" statt
  „Zufallsagenten-Fluktuation". Zwei Stufen: (a) Minimal (Mess-Fix): Zähler `deaths`, `respawns`,
  `mean_age` je Snapshot in den Runner; (b) Struktur-Fix (Design-Entscheidung, Richtungsfrage an
  den User): Population geburten-getragen machen (Energie-Ökonomie: chronisches ~66/240-Niveau aus
  Prüfung 4 ist die Wurzel — vgl. Vegetations-/Wind-Fix-Historie auf `core/audit-fixes`) und/oder
  Respawn mit Vererbung von einem lebenden Agenten statt Zufallshirn.
- *Umfang:* (a) klein (nur Runner), (b) mittel (Balancing-Iterationen nötig).
- *Dateien/Lane:* (a) `scripts/m1_pilot.py` — infra-Lane, unkritisch. (b) `simulation.py`,
  ggf. `agents/agent.py` — **HOT, core-lead seriell.**

**K3 — Batterie-Messbarkeit härten.**
- *Evidenz:* Prüfung 3/5 — Welt-Seed erklärt mehr Varianz als jeder Arm (Seed 41 überall ≈ 0,
  Seed 42 überall hoch); `discoveries` saturiert; „eine Glücks-Agentin" ist von
  Populationskompetenz nicht unterscheidbar.
- *Erwarteter Effekt:* Ein Re-Run kann Arm-Effekte tatsächlich auflösen: mehr Seeds (≥ 8, gepaart
  über Arme auswerten — der gepaarte A1-vs-A2-Vergleich war das informativste Signal der ganzen
  Batterie), plus per-Snapshot: Tool-Cut-Konzentration über Agenten (z. B. Top-1-Anteil/Gini),
  Alters-Histogramm, kumulierte deaths/respawns.
- *Umfang:* Klein–mittel, nur Runner + Report.
- *Dateien/Lane:* `scripts/m1_pilot.py`, `scripts/m1_report.py` — infra-Lane.

### Fähigkeits-Optimierung (H-getrieben — erst NACH K1–K3 neu messen)

**F1 — H4 (Lern-Kopplung/Kredit) bleibt Leitverdacht, Bau erst nach sauberem Re-Run.**
- *Evidenz:* A1 ≈ A2 final; aber gepaart 4/5 pro A1 im Verlauf — der Effekt ist klein und von
  K1/K2 nicht trennbar. Spec §4 gilt: Kredit-Backfill ist kein Knob, sondern neuer Paket-Code —
  zu teuer, um ihn auf konfundierten Daten zu begründen.
- *Empfehlung:* Re-Run der Kern-Arme (A1/A2/A3/D1, gepaart, ≥ 8 Seeds) nach K1+K2a. Erst wenn
  A1 ≈ A2 dann noch steht, Kredit-/Kopplungs-Reparatur bauen (Reward-Attribution an die
  Knapping-Aktion; `agents/brain.py` — HOT).

**F2 — NICHT bauen (durch die Batterie deprioritisiert):**
- Curiosity-Tuning (H2): C2 (3×) schadet aktiv, C1 (0×) ohne klaren Effekt.
- γ-/Horizont-Tuning (H1): B1 wirkungslos.
- Verb-Bias/Entropie-Lever (E1/F1): kein Hub über A1.

### Vorgeschlagene Reihenfolge

1. K1 (Cache-Fix, core-lead) + K2a/K3 (Runner-Instrumentierung, infra) — parallelisierbar.
2. Kern-Re-Run A1/A2/A3/D1 gepaart (≥ 8 Seeds) → entscheidet H3 (jetzt erstmals messbar) und H4.
3. Abhängig vom Re-Run: K2b (Demografie/Energie-Ökonomie) und/oder F1 (Kredit-Kopplung).

---

*Datenbasis: `battery_results/*.jsonl` (31 Läufe, meta/snap/final-Records); Analyse record-weise
inkl. Byte-Identitäts-Vergleich, Hälften-Inkremente und Δpop-Schranken; Quellcode-Belege aus
Worktree `feat/infra-m1-pilot` @ 9e10eb6.*
