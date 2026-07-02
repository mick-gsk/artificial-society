# Forschungsauftrag: Falsifizierbares Umwelt-Design für nachweisbar offene kumulative Akkumulation (emergent-sim / "Artificial Society")

## Rolle & Modus

Du bist mein wissenschaftlicher Ko-Designer für Open-Ended Evolution / ALife und arbeitest in **Claude Code (Opus 4.8, Ultracode — Multi-Agent-Orchestrierung erlaubt und erwünscht)**. Behandle diese Aufgabe als **wissenschaftliche Untersuchung mit Paper-Ambition (peer-reviewed, Thema: confound-kontrollierte open-ended Innovation)**, nicht als Bauauftrag.

**Reihenfolge ist Pflicht (Diagnose-zuerst):** `Diagnose → Hebel-Priorisierung → Design → Plan`. Entwirf **kein** Umwelt-Feature, bevor die Engpass-Ursache am Code verifiziert ist.

**HARTE GRENZE: KEIN Produktionscode ohne explizite Freigabe.** Liefere Forschung, Design und Experimentplan. Erlaubt sind ausschließlich: Lesen/Analysieren der Codebasis, kleine, klar als Wegwerf gekennzeichnete Mess-/Sondierungs-Skripte zur Hypothesenprüfung sowie das Schreiben der unten spezifizierten Dokumente. Das Ändern der Simulations-Logik ist verboten, bis ich "implementieren" sage.

## Kontext der Codebasis (gegeben — nicht neu erfinden)

- Bestehende **"Artificial Society" / emergent-sim** Codebasis (Python): Agenten in einer getickten Welt; Wahrnehmung, Aktion, Reproduktion, Welt-Update.
- **Performance-Budget ist eine harte Schranke:** `world-update` ist der Engpass; ~8000 Ticks ≈ 9 h CPU; Pilots realistisch auf ~1500 Ticks gedeckelt. Compute läuft primär auf dem GPU-PC via SSH, MacBook nur Fallback. **Eigenheit beachten:** `world-update` ist CPU-seitig ~7–11× schneller als auf GPU — kalkuliere Tick-Budgets entsprechend; ein Mechanismus, der erst nach 10^6 Ticks "vielleicht" greift, ist für diese Untersuchung wertlos.
- **Verifiziere am Code, nicht aus der Erinnerung:** Wo Episoden resetten, ob Umweltmods persistieren, wie Transmission implementiert ist, wo das Erfolgsmaß berechnet wird — durch Lesen/Messen belegen, mit Datei-Pfaden + Zeilen.

## Übergeordnetes, falsifizierbares Ziel

Die naive Leitfrage "Wie schaffe ich eine Umgebung, in der eine dem Menschen überlegene Spezies entsteht?" ist **als Forschungsziel untauglich**: "überlegen" ist undefiniert und unfalsifizierbar, und sobald man "überlegen" fix misst, führt man eine externe Fitnessfunktion ein (Verstoß gegen C1).

**Operationales Ziel, das wir verfolgen:**
> Identifiziere und teste die Bedingungen, unter denen das **gekoppelte System (Lerner + Umwelt)** **nachweisbar offene, kumulative Akkumulation funktionaler Komplexität** zeigt — gemessen **gegen eine compute-gematchte Random-/Rekombinations-Baseline**, mit null-kalibrierten funktionalen Maßen und über die Zeit nicht plateauend.

## Empirische Pilot-Fakten (Erdung — hier beginnt die Diagnose, nicht Nebenbemerkung)

Diese Befunde sind **der eigentliche Grund** für diese Direktive und der Ausgangspunkt der Diagnose:

1. **Ein disembodied Random-Recombiner schlägt das gelernte System** (gelernte max. funktionale Tiefe ~22.9 vs. ~32.3 random; gepaarte Differenz **−9.4**, CI **[−11.3, −7.2]**). → Soziale Transmission liefert aktuell **keinen Treuevorteil** gegenüber blindem Rekombinieren.
2. **Diversity-Collapse 0.127 → 0.029** unter greedy means-ends.

**Diagnostische Schlussfolgerung, die dein gesamtes Vorgehen leitet:** Der Engpass liegt **nachweislich NICHT** am fehlenden expandierbaren Lösungsraum (die Original-Hypothese). Er liegt an **(a) Transmissionstreue / Ratchet**, **(b) Diversitätserhalt** und **(c) hackbaren Erfolgsmaßen** (eine naive DAG-Tiefe ist genau das, was ein blinder Rekombinierer trivial maximiert). Jedes Umwelt-Feature ist zuerst gegen diese drei Engpässe zu begründen, bevor es in Frage kommt. Ein Design, das primär den Raum vergrößert, ignoriert die Pilot-Evidenz und ist abzulehnen.

**Pflicht-Disambiguierung (zwei konfundierte Root-Causes trennen):** Der −9.4-Befund hat zwei verschiedene mögliche Ursachen mit verschiedenen Fixes:
- **(A) Maß-Artefakt:** DAG-Tiefe ist genau das, was blinde Rekombination trivial maximiert → der "Vorteil" ist ein Metrik-Defekt.
- **(B) Echter Treuemangel:** soziale Transmission verliert real Information → Ratchet greift nicht.
Die Diagnose **muss** (A) vs. (B) explizit auseinanderhalten und am Code/Daten prüfen, welche dominiert. Optimiere nicht am falschen Engpass.

## Erkenntnistheoretischer Rahmen (verbindlich für alles Folgende)

**Grundregel — NOTWENDIG ≠ HINREICHEND (überschreibt alles):** Tierra/Avida besitzen endogenen Druck + expandierbaren Raum + niche construction und plateauen **trotzdem**. Kein künstliches System hat je nachhaltige offene Eskalation gezeigt. Dein Default-Prior ist Skepsis, nicht Optimismus. Behandle **jede** Mechanik explizit als **notwendige-aber-nicht-hinreichende** Bedingung.

- **Verbotene Sprache:** "zwangsläufig", "führt zu", "garantiert", "erzeugt automatisch", "stellt sicher", "löst das Problem".
- **Erlaubte Ersatzformulierungen:** "ist plausibel notwendig für X", "entfernt einen bekannten Blocker", "macht X möglich, ohne es zu garantieren"; ob hinreichend, entscheidet das Experiment.

**Jede Designentscheidung ist als testbare Hypothese mit folgendem Vierertupel zu formulieren:**
1. **Mechanismus** — was konkret am gekoppelten System Lerner+Umwelt geändert wird, gemappt auf die Codebasis;
2. **null-kalibrierte Baseline** — die compute-gematchte Random-/Rekombinations-Kontrolle, gegen die gemessen wird (zuerst definieren, *bevor* der Mechanismus festgelegt wird);
3. **vorab definierte, hack-resistente funktionale Erfolgsmetrik**;
4. **explizites Kill-Kriterium** — welches Messergebnis die Hypothese *widerlegt*.

> **Falsifikations-Heuristik:** Wer kein Kill-Kriterium formulieren kann, hat keine Hypothese.

## Korrigierte Designprinzipien (C1–C5) — verbindlich

Die ursprünglichen Arbeitsthesen waren teils falsch (Ergebnis eines adversariell validierten Literatur-Checks). Verwende **ausschließlich** die korrigierten Fassungen:

**C1 — Achse ist STATISCH vs. NICHT-STATIONÄR (an die eigene Population gekoppelt), NICHT "endogen vs. exogen".**
Der Druck muss an die eigene Population gekoppelt und nicht-stationär sein: Koevolution / Self-Play / Red-Queen-Dynamik. Eine statische, fest verdrahtete Fitnessfunktion plateaut. Pflicht: gegen **Disengagement** absichern (z. B. Hall of Fame, expliziter Diversitätsdruck), sonst kollabiert das Arms-Race. "Endogen" allein ist nicht der Punkt — gekoppelte Nicht-Stationarität ist es.

**C2 — Repräsentation darf erreichbare Komplexität nicht deckeln (Ausdrucksmächtigkeit, nicht Rohlänge).**
Bestbelegtes Prinzip (vgl. Soros & Stanley 2014 — exakte Bedingungsnummer/-bezeichnung **vor Verwendung verifizieren**, nicht aus dem Gedächtnis zitieren). **Zwei Original-Teilsätze sind FALSCH und werden umgedreht:**
- (a) FALSCH: "enge/diskrete Kanäle verhindern reiche Sprache." **KORREKT: enge/diskrete Kanäle treiben Kompositionalität** (vgl. Resnick et al. 2020 — verifizieren). Plane diskrete/Bottleneck-Kanäle als *Feature*, nicht als Mangel.
- (b) FALSCH: "fixe Genomlänge = fixer Komplexitätsdeckel." **KORREKT: entscheidend ist die Ausdrucksmächtigkeit der Repräsentation** (indirekte Kodierung / CPPN-artig), nicht rohe Länge.
Zielgröße ist ein **expandierbarer AKTIONS-/Konstruktionsraum** (door-opening states / adjacent possible), nicht ein passiver Zustandsraum. **Caveat: Ausdrucksmächtigkeit ≠ Erreichbarkeit** — ohne Stepping Stones bleibt der Raum praktisch leer. **Einnordnung zur Diagnose:** Der Pilot zeigt, dass mehr Raum *nicht* der Engpass ist — C2 ist relevant, aber **nachgeordnet**.

**C3 — Niche Construction ist ENABLING-Bedingung, NICHT "zentraler Treiber"; richtungsneutral** (kann Komplexität auch *abbauen*).
**Technische Pflichtbedingung, sonst existiert der Mechanismus gar nicht:** Umweltmodifikationen müssen über Generationen **persistent** und **vererbbar/transmittierbar** sein — **kein Episode-Reset**. Der **kulturelle (sozial-transmittierte) Kanal ist wichtiger als der genetische** und getrennt zu behandeln.

**C4 — Kumulative Kultur via Demografie + Hochtreue-Transmission ist der unumstrittene Kern; MLS NICHT als gesicherten Haupthebel verkaufen.**
Multi-Level-/Gruppenselektion ist umstritten und fragil — nicht als Haupthebel framen. Unumstrittener Kern: kumulative Kultur via **Demografie (effektive Populationsgröße + Konnektivität)** plus **Hochtreue-Transmission**. "Koordination schlägt IQ" ist zu entschärfen zu: **"Koordination + hinreichende Individualkognition sind komplementär."** **Drei Dinge nie konflatieren:** (i) kulturell vs. genetisch · (ii) Gruppen- vs. Individualselektion · (iii) Koordination vs. IQ.

**C5 — Open-Endedness ist konstitutiv Eigenschaft des GEKOPPELTEN Systems Lerner+Umwelt; Dichotomie "Umwelt statt Agent" ist falsch.**
(vgl. Hughes et al. 2024 — verifizieren.) Investiere **parallel** in Algorithmus UND Umwelt. "Dem Menschen überlegene Spezies" als Ziel ist gestrichen (siehe oben); reales Ziel = nachweisbar offene kumulative Akkumulation gegen compute-gematchte Random-Baseline.

## Pflicht-Achsen: die quantitativen Hebel

Jeder Hebel muss im Design-Doc als eigene, **mit einer messbaren Größe operationalisierte** Achse auftauchen — oder begründet ausgeschlossen werden. Prioritäten 1–3 folgen direkt aus der Diagnose und kommen **zuerst**:

1. **Transmissionstreue / Ratchet-Effekt** — kumulative Kultur existiert nur **oberhalb einer kritischen Fidelity-Schwelle**. Treue explizit parametrisieren, messen, Schwelle empirisch lokalisieren. (Adressiert Pilot-Fakt 1.)
2. **Effektive Populationsgröße + Konnektivität + aktiver Diversitätserhalt** — QD/Novelty-Druck statt reiner Fitness-Maximierung. (Adressiert Pilot-Fakt 2.)
3. **Null-kalibrierte FUNKTIONALE Erfolgsmaße** — naive Maße (DAG-Tiefe, Genomlänge, Vokabulargröße) sind durch blinde Rekombination hackbar; jedes Maß gegen die Random-Baseline kalibrieren. (Adressiert Pilot-Fakt 1 + 2.)
4. **Koevolutionäre Arms-Races / Red-Queen** als Quelle anhaltenden Drucks + **Anti-Disengagement**-Schutz.
5. **Mutations-/Innovationsrate balanciert gegen Selektionsstärke** (Error-Threshold; **Evolvierbarkeit als eigenes Ziel**).
6. **Compute-/Energie-Budget als harte Schranke** — realistische Tick-/Generationenzahlen (siehe Codebasis-Kontext), explizit eingerechnet inkl. CPU-vs-GPU-Eigenheit.
7. **Erreichbarkeit / Stepping-Stone-Struktur** — operationalisiert als Stepping-Stone-Dichte bzw. Übergangswahrscheinlichkeit zwischen Innovationen; der konzeptuelle Hebel hinter QD/Novelty (Ausdrucksmächtigkeit ≠ Erreichbarkeit). **Sekundär:** Embodiment, intrinsische Motivation, Turnover/Tod, Stigmergie, Major Transitions (Konfliktunterdrückung).

## Explizite Anti-Patterns / Verbote (harte Ausschlüsse)

- **KEINE** Notwendig-als-hinreichend-Falle; kein Mechanismus "erzeugt" Open-Endedness.
- **KEINE** Garantie-/Zwangsläufigkeits-Sprache (siehe verbotene Wörter oben).
- **KEINE** "Umwelt statt Agent"-Dichotomie (C5).
- **KEINE** der zwei falschen C2-Teilsätze (enge Kanäle ⇏ arme Sprache; Genomlänge ⇏ Komplexitätsdeckel) — nur in umgedrehter Form verwenden.
- **KEIN** "dem Menschen überlegene Spezies" als Zielgröße und kein fixes "Überlegenheits"-Maß.
- **KEIN** Episode-Reset dort, wo Persistenz/Vererbung gefordert ist (verletzt C3).
- **KEINE** rein fitness-getriebene Selektion ohne Diversitäts-/Novelty-Druck (verletzt Pilot-Fakt 2).
- **KEINE** naiven, ungekalibrierten Erfolgsmaße ohne null-kalibrierte Kontrolle.
- **KEINE** erfundenen Zitate, DOIs oder Jahreszahlen; jede Quelle verifiziert oder explizit `[UNVERIFIZIERT]`.
- **KEIN** Produktionscode / keine Logik-Änderung ohne Freigabe.

## Arbeitsweise (Ultracode-Hygiene)

- **Multi-Agent-Fan-out:** Parallelisiere unabhängige Stränge, danach synthetisieren/deduplizieren/Widersprüche auflösen:
  (i) **Codebase-Mapping** in getrennten Lese-Strängen — (a) `world-update`/Engpass, (b) Agent/Genom/Policy + Aktionsraum, (c) Reproduktion/Vererbung/Episode-Lifecycle (kritisch für C3-Persistenz), (d) Kommunikations-/Transmissionskanal, (e) bestehende Metriken & Logging/Baselines;
  (ii) **Literatur-/Zitat-Verifikation** (WebSearch/WebFetch);
  (iii) **Metrik-Hackbarkeits-Analyse**;
  (iv) **Root-Cause-Diagnose** des Random-Recombiner-Vorteils und des Diversity-Collapse.
- **Adversarielle Selbst-Verifikation (Pflicht):** Lass pro Designentscheidung der Achsen 1–4 einen Gegen-Agenten aktiv widerlegen: *"Wie hackt eine Random-/Blind-Baseline diese Metrik? Wo verstecke ich unbemerkt eine statische Fitnessfunktion? Wo verwechsle ich notwendig/hinreichend? Welcher Reset zerstört C3?"* Dokumentiere Angriff und Antwort; was das nicht übersteht, fällt raus oder wird als fragil markiert.
- **Null-Baseline-Denken zuerst:** Für jedes Maß/Feature zuerst die compute-gematchte Random-/Blind-Baseline definieren. Würde sie das Maß schlagen → Maß/Feature ist noch nicht fertig.
- **Confound-Kontrolle (Paper-Neuheit):** Pro Experiment explizit benennen, welche Confounds kontrolliert werden (mind. Embodiment, Compute, Maß-Hackbarkeit) und wie.
- **Zitat-Disziplin:** Nur verifizierbare Quellen (WebSearch/WebFetch nutzen). Unverifiziertes explizit als `[UNVERIFIZIERT]` taggen; verifiziert vs. unverifiziert getrennt listen. Lieber ehrliche Lücke als erfundene Referenz.
- **Budget-Realismus:** Jeder Experimentvorschlag nennt geschätzte Ticks/Generationen, Laufzeit, GPU-PC-vs-MacBook-Zuordnung inkl. Job-Submission/Monitoring via SSH und passt ins ~1500-Tick-Pilot- bzw. größere GPU-Budget.

## Deliverables (abnahmefähig — getrennte Markdown-Dateien, Pfade angeben)

Schreibe keinen Produktionscode. Erzeuge:

1. **`DIAGNOSE.md`** — Code-verifizierte Root-Cause-Analyse: *warum* der Random-Recombiner gewinnt (mit expliziter (A)-Maß-Artefakt-vs-(B)-Treuemangel-Trennung) und *warum* Diversität kollabiert. **Inklusive: Definition der Diversitätsmetrik (genotypisch? Verhaltens-/QD-Deskriptor?)** — ein Fix ohne Definition ist nicht falsifizierbar. Datei-Pfade + Zeilen. Klar getrennt: **belegt vs. vermutet**.
2. **`HEBEL-ROADMAP.md`** — die Hebel 1–7 geordnet nach Diagnose-Relevanz × Aufwand × Compute-Budget; die nächsten 3 konkreten Schritte gegen die Codebasis.
3. **`CODEBASE-MAP.md`** — pro priorisiertem Hebel die berührten Module/Dateien/Funktionen (Pfade); explizit die Stelle, die Episode-Reset / Persistenz von Umweltmods regelt (C3-Check), und der `world-update`-Engpass.
4. **`DESIGN.md`** — konkretes, falsifizierbares Umwelt-Design aus C1–C5 (korrigiert). Jede Mechanik als **Vierertupel** (Mechanismus + null-kalibrierte Baseline + funktionale Metrik + Kill-Kriterium), mit Label **"notwendig-nicht-hinreichend"** + Begründung gegen die Diagnose + Bezug zu C1–C5.
5. **`EXPERIMENT-PLAN.md`** — priorisierte Tests (billigste/härteste Falsifikationen zuerst); pro Experiment: Hypothese, manipulierte Achse(n), Kontrollen, **kontrollierte Confounds**, compute-gematchte Baseline, erwartete Falsifikationssignatur, Tick-/Generationen-Budget, GPU-PC-Ausführung via SSH. Ablations zur Trennung der Achsen (insb. Treue, Diversität, kulturell/genetisch).
6. **`METRICS-BASELINE.md`** — pro funktionalem Maß: Definition, **compute-gematchte Random-/Blind-Baseline**, Anti-Hacking-Argument (warum nicht durch blinde Rekombination hackbar, Bezug zu −9.4), Null-Kalibrierungsprozedur, Mess-Verfahren für Ratchet-/Fidelity-Schwelle und Diversität. **Inklusive A/B-Reproduzierbarkeitsprotokoll: Seed-Kontrolle, Anzahl Replikate, vorab festgelegter (gepaarter) Test.**
7. **`RISKS-KILL.md`** — pro Designentscheidung Frühindikator + hartes, **vorab quantifiziertes** Kill-Kriterium (z. B. "schlägt Maß X die Random-Baseline nach N Ticks nicht → Mechanik verwerfen"); generische Fallen (Disengagement, Diversity-Collapse, Metrik-Hacking, versteckte statische Fitness); die dokumentierten Red-Team-Einwände.
8. **`REFERENCES.md`** — verifiziert vs. `[UNVERIFIZIERT]` getrennt. Keine Fabrikation.
9. **`OPEN-QUESTIONS.md`** — am Code/Daten ungeprüfte Annahmen; was ich bestätigen muss.

## Definition of Done (Akzeptanzkriterien)

- [ ] Diagnose ist **code-verifiziert** (Pfade/Zeilen), nicht spekuliert; **belegt vs. vermutet** getrennt; (A)-Maß-Artefakt-vs-(B)-Treuemangel **explizit getrennt**.
- [ ] **Diversitätsmetrik ist definiert** (nicht nur die Zahl 0.127→0.029 zitiert).
- [ ] Jede Designentscheidung ist als Hypothese mit **Mechanismus + null-kalibrierter Baseline + funktionaler Metrik + Kill-Kriterium** formuliert; jede Mechanik trägt das Label **"notwendig-nicht-hinreichend"**.
- [ ] **Null** Garantie-/Zwangsläufigkeitssprache im gesamten Output.
- [ ] C1–C5 in **korrigierter** Fassung; die zwei falschen C2-Teilsätze tauchen nur **umgedreht** auf.
- [ ] Die **drei C4-Achsen** (kulturell/genetisch, Gruppe/Individuum, Koordination/IQ) sind **nicht konflatiert**.
- [ ] Ziel ist **offene kumulative Akkumulation vs. Random-Baseline** — nicht "überlegene Spezies".
- [ ] Alle Pflicht-Achsen (1–7) sind adressiert oder begründet ausgeschlossen; Hebel 1–3 sind **priorisiert vor** dem großen Umwelt-Design.
- [ ] **Jedes** Erfolgsmaß hat eine compute-gematchte Random-/Blind-Baseline + explizites Anti-Hacking-Argument.
- [ ] **C3-Persistenz-/Vererbbarkeits-Check ist am konkreten Reset-Code** durchgeführt.
- [ ] Pro Experiment sind die **kontrollierten Confounds** benannt (mind. Embodiment, Compute, Maß-Hackbarkeit).
- [ ] **A/B-Reproduzierbarkeitsprotokoll** (Seeds, Replikate, gepaarter Test) ist spezifiziert.
- [ ] Mindestens eine **adversarielle Red-Team-Runde pro Designentscheidung der Achsen 1–4** ist dokumentiert.
- [ ] Alle Experimente passen ins Compute-Budget (GPU-PC/SSH, CPU-vs-GPU-Eigenheit, realistische Tick-Zahlen) oder begründen explizit den Mehrbedarf.
- [ ] **Kill-Kriterien vorab quantifiziert.**
- [ ] **Keine erfundenen Zitate**; Zitat-Status ausgewiesen.
- [ ] **Kein Produktionscode** ohne Freigabe; Wegwerf-Mess-Skripte als solche markiert.

## Erster Schritt

Bestätige kurz dein Vorgehen (Fan-out-Plan + welche Module du zuerst mappst), dann starte **parallel** mit Codebase-Mapping, Root-Cause-Diagnose und Zitat-Verifikation. **Beginne mit der Diagnose, bevor du Design/Plan ausarbeitest.** Stelle Rückfragen nur, wenn eine Annahme den gesamten Design-Korridor kippen würde; ansonsten arbeite mit explizit markierten Annahmen weiter und schreibe keinen Produktionscode ohne Freigabe.
