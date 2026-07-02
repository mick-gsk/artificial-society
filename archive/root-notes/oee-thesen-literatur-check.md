# Stimmt das, was du geschrieben hast? – Literatur-Check

> Adversariell validiert (5 Thesen × 2 Fachlinsen [ML/Open-Endedness + Evolutionsbiologie/Kultur] + skeptisches Gegengutachten + Vollständigkeits-Kritiker), Stand 2026-06-30.

## 1. Gesamturteil

Deine vier Thesen sind im Kern keine Hirngespinste, sondern treffen reale, in der Open-Endedness- und Evolutionsbiologie-Literatur belegte **Notwendigkeitsbedingungen** für offene Komplexitätssteigerung – endogen/ko-evolvierender Druck (C1), expandierbarer Repräsentationsraum (C2, fast wörtlich Soros & Stanleys vierte Bedingung), niche construction (C3) und Mehrebenen-/kulturelle Selektion (C4). Das ist die gute Nachricht. Die zentrale Schwäche aller Thesen ist symmetrisch und systematisch: Du verwechselst durchgängig **notwendig mit hinreichend** und formulierst mit "zwangsläufig"/"entscheidend"/"der Hebel" zu stark – kein bekanntes künstliches System hat je nachhaltige offene Eskalation gezeigt, und genau die Negativbefunde (Tierra/Avida haben endogenen Druck UND expandierbaren Raum UND niche construction und stagnieren trotzdem) widerlegen jede Maximal-Lesart. Der eine zentrale Vorbehalt, den du verinnerlichen solltest: Deine Thesen sind eine Liste qualitativer "Schalter", aber offene Akkumulation scheitert real an **quantitativen Kipp-Bedingungen** (Transmissionstreue, Populationsgröße/Diversität, Mutations-/Selektions-Balance) und am Mess-/Null-Baseline-Problem – exakt das, was dein eigener Pilot bereits empirisch gezeigt hat (random-recombiner schlägt das gelernte System).

## 2. These für These

### C1 – Endogener Druck (vs. fest verdrahtete Fitnessfunktion)
**Verdikt: teilweise belegt – hohe Konfidenz** (beide Gutachten + Verifikation konvergent).

**Was stimmt:**
- Eine fixe, gradientensteile Zielfunktion führt nachweislich zu vorzeitiger Konvergenz/Plateau durch *Deception* – das ist der Kernbefund von Novelty Search (Lehman & Stanley 2011).
- Wenn der Druck mit den Agenten ko-evolviert, entsteht offene Eskalation – POET (Wang et al. 2019), Hide-and-Seek-Autocurricula (Baker et al. 2019).
- Statische/parsimonie-getriebene ALife-Systeme produzieren historisch ein Komplexitäts-Plateau (Tierra/Avida; Bedau et al. 2000) – realer Negativbefund für fixe Selektionsregime.

**Was zu stark / fehlt (stärkster Gegen-Einwand):**
- **"Zwangsläufig" ist in beide Richtungen falsifiziert.** Tierra/Avida *haben* endogene/intrinsische Fitness (Selbstreplikation, Konkurrenz um CPU-Zeit, ko-evolvierende Parasiten) und stagnieren dennoch → Endogenität ist **nicht hinreichend**. Umgekehrt zeigen Sims-1994-Koevolution und Red-Queen-Dynamiken Eskalation bei *fixer* Fitnessfunktion, und ein exogen spezifiziertes Novelty-/Empowerment-Kriterium ist fest verdrahtet UND plateau-resistent → Endogenität ist **nicht streng notwendig**.
- **Du benennst die falsche Trennachse.** Die entscheidende Variable ist nicht *endogen vs. exogen*, sondern *statisch vs. nicht-stationär/gekoppelt*. POET ist ein extern implementierter Umweltgenerator – der effektive Druck wird aber durch Kopplung an die Population nicht-stationär.
- Auch endogene Koevolution kann statt zu eskalieren *zyklisch/disengaged* werden ("mediocre stable states", loss of gradient; Ficici & Pollack 1998; Cartlidge & Bullock 2004). Endogenität ist kein Schutz davor.

**Zitat-Warnung:** Standish 2003 vermischt zwei verschiedene Standish-Arbeiten (Parsimonie-Aussage stammt aus arXiv nlin/0604026 ~2006, nicht aus dem 2003er "Open-Ended Artificial Evolution"). "Sims 1994 erzeugt *anhaltende* Eskalation" ist stärker als das Paper zeigt. Taylor et al. 2016 existiert in zwei Varianten (Artificial Life 22(3) und arXiv:1507.07403) – uneinheitlich zitiert.

### C2 – Expandierbarer Repräsentationsraum
**Verdikt: teilweise belegt – hohe Konfidenz** (Downgrade-Versuch auf "übervereinfacht" geprüft und verworfen).

**Was stimmt:**
- Der Kern ist **fast wörtlich Soros & Stanleys vierte Notwendigkeitsbedingung**: "Die potenzielle Komplexität des Phänotyps darf nicht durch seine Repräsentation begrenzt sein" (Soros & Stanley 2014, Chromaria – empirisch: Stagnation, sobald eine der vier Bedingungen fehlt). Das ist der stärkste Beleg in deinem gesamten Thesenpaket.
- Komplexifizierung als laufende Raum-Erweiterung ist etabliert (NEAT; Stanley & Miikkulainen 2002).
- Der Teil "Erfindung als endogene Aktion, nicht externer Würfelwurf" deckt sich mit Soros & Stanleys Autonomie-Bedingung und mit Hughes et al. 2024 ("Open-Endedness Essential for ASI").

**Was zu stark / fehlt (stärkster Gegen-Einwand):**
- **Zwei deiner drei Teilbehauptungen sind empirisch falsch:**
  1. *"Fixe Nachrichtenkanäle ⇒ keine reichere Sprache"* ist umgekehrt: begrenzte Kanalbandbreite **treibt** Kompositionalität (Resnick et al. 2020; Kharitonov & Baroni). Ein fester diskreter Kanal kann eine produktiv-unendliche Bedeutungssprache tragen.
  2. *"Fixe Genomlänge ⇒ fixe maximale Komplexität"* gilt nur für *beschreibbare* Struktur, nicht für funktionale Komplexität: indirekte Kodierung (CPPN/HyperNEAT) und turing-vollständige Substrate erzeugen unbeschränkte Phänotypen aus festem Genom. Die relevante Grenze ist *Ausdrucksmächtigkeit*, nicht rohe Länge.
- **Notwendig ≠ hinreichend:** Baker et al. 2020 (Hide-and-Seek) zeigt sechs Runden eskalierender Innovation bei *fixer* Architektur und festem Aktionsrepertoire. Gegenbeispiel zur *Notwendigkeitsrichtung* – plateaut aber nach sechs Runden, also kein Gegenbeleg gegen die *kumulativ-unbegrenzte* Lesart, die du eigentlich meinst. Halte diese Unterscheidung sauber.
- Übersehene Achse: Du brauchst einen expandierbaren **Aktions-/Konstruktionsraum** ("door-opening states", adjacent possible), nicht nur einen großen passiven Zustandsraum. Und: *Ausdrucksmächtigkeit ≠ Erreichbarkeit* – ein unendlicher Raum kann durch fehlende Stepping Stones faktisch unerreichbar bleiben.

**Zitat-Warnung:** Die HyperNEAT-Zahl ">300.000:1-Kompression" ist nicht belegt (Größenordnung plausibel, exakte Zahl nicht verifiziert). Resnick et al. 2020 bestätigt nur die *untere* Grenze des Bottleneck-Effekts. (Soros & Stanley 2014 selbst: verifiziert, nahezu wörtlich.)

### C3 – Niche Construction
**Verdikt: teilweise belegt – hohe Konfidenz.**

**Was stimmt:**
- Niche Construction Theory ist formal etabliert: Organismen verändern Selektionsdrücke für Nachkommen via *ökologischer Vererbung* (Odling-Smee, Laland & Feldman 2003; Laland et al. 1999 PNAS).
- Die menschliche Analogie (Werkzeuge/Feuer/Landwirtschaft als kulturelle Nischenkonstruktion) ist grundsätzlich tragfähig (Wrangham 2009; Boyd & Richerson; Henrich).
- In-silico-Beleg, dass umgebaute Umwelt Eskalationsstufen erzeugt: Hide-and-Seek-Autocurriculum (Baker et al. 2020).

**Was zu stark / fehlt (stärkster Gegen-Einwand):**
- **"Zentraler Treiber" ist überzogen.** NCT-Autoren selbst positionieren niche construction als *ko-direktiven* Prozess *neben* Selektion, nicht als dominanten Motor. Tierra/Avida haben genau diese Rückkopplung und plateauen trotzdem → notwendige Enabling-Bedingung, kein Garant.
- **Niche construction ist richtungsneutral:** Sie kann via *relaxed selection*, Selbstvergiftung der Umwelt oder evolutionäre Fallen Komplexität auch **abbauen**.
- **Major-Transitions-Theorie zeigt:** Leiter-Sprünge (deine These 4) entstehen primär durch *Konfliktunterdrückung* zwischen Subeinheiten, nicht durch Umweltumbau. Niche construction ist hier bestenfalls Ko-Faktor.
- Implizite Voraussetzungen, die du nicht nennst: Umweltmodifikation muss **persistent und vererbbar** sein (in vielen RL-Setups wird die Umwelt pro Episode resettet → der Mechanismus existiert technisch gar nicht). Der eigentliche menschliche Hebel ist **kulturelle** (sozial-transmittierte), nicht genetische niche construction – höhere Anforderungen (Hochtreue-Transmission, Ratchet).

**Zitat-Warnung (echte Fehlattribution):** "Chli & De Wilde (2004), Niche Construction and the Evolution of Complexity" ist **falsch zugeordnet** – der ALIFE-IX-2004-Beitrag dieses Titels stammt von **Tim Taylor**, nicht von Chli & De Wilde.

### C4 – Kulturelle Evolution + Multi-Level-Selektion ("Koordination schlägt IQ")
**Verdikt: teilweise belegt – hohe Konfidenz.**

**Was stimmt:**
- Der **kulturelle Kern** ist breit belegt: kumulative kulturelle Evolution (nicht genetische) erklärt menschliche ökologische Dominanz – Henrichs "collective brain" (Henrich 2015/2016; Boyd & Richerson; Tomasello-Ratchet).
- Der Mensch ist kein angeborener Apex-Prädator – Dominanz folgt aus Technik/Kultur (siehe aber Warnung unten).
- Kulturelle Gruppenselektion ist eine ernstzunehmende, formal ausgearbeitete Theorie (Konformismus/Prestige-Bias reduziert Within-Group-Varianz; Richerson et al. 2016).
- Koordination/geteilte Intentionalität als kognitiver Sonderweg ist gut gestützt (Tomasello).

**Was zu stark / fehlt (stärkster Gegen-Einwand):**
- **Du konflatierst drei logisch unabhängige Dinge:** (a) kulturell vs. genetisch, (b) Gruppen- vs. Individualselektion (MLS), (c) Koordination vs. Einzel-IQ. Kumulative Kultur erfordert **keine** Gruppenselektion – sie läuft auch unter reiner Individualselektion mit hochtreuem sozialem Lernen.
- **MLS/Gruppenselektion ist umstritten, nicht Konsens.** Inklusive-Fitness-Theoretiker halten sie für mathematisch äquivalent oder heuristisch irreführend (West/Griffin/Gardner 2008; Pinker 2012; Nowak/Tarnita/Wilson 2010 + 137-Autoren-Replik); neuere Arbeiten bestreiten sogar die Äquivalenz (van Veelen et al. 2023). Du präsentierst eine offene Streitfrage als gesicherten "Hebel".
- **"Koordination schlägt Einzel-IQ" ist ein falsches Entweder-Oder.** Evidenz zeigt Komplementarität: kumulative Kultur braucht *zugleich* hinreichende Individualkognition (Imitationstreue, Sprache) UND Konnektivität. Henrichs Argument ist "kollektives Gehirn", nicht "IQ egal".
- **Der wichtigere Hebel ist Demografie:** Populationsgröße + Konnektivität (Henrich, Derex 2013, Powell 2009) sind oft stärkere Prädiktoren kultureller Komplexität als ein Gruppen-Fitnessterm. MLS braucht restriktive Bedingungen (geringe Migration, hohe Between-Group-Varianz, Free-Rider-Unterdrückung) – ohne explizite Gruppen-Reproduktion ist "Selektionseinheit Gruppe" nur Metapher.

**Zitat-Warnung:** Inkonsistente Evidenzbasis bei der Nahrungskette: Bonhommeau et al. 2013 (heutiger trophischer Level ~2.21, "mittlerer Bereich") vs. Ben-Dor et al. 2021 (Homo ~2 Mio. Jahre Hyper-Karnivor/Apex-Prädator; trophischer Level sank erst *spät* mit intensiver Kultur). Beide real, messen aber Verschiedenes. **Deine Figur "aus der Mitte an die Spitze durch Kultur" ist damit unscharf bis verkehrt herum** – nur als Nebenbehauptung nutzen.

### C5 – Überrahmung: "Engpass primär an der Umwelt, nicht an Agenten/Algorithmen"
**Verdikt: teilweise belegt – mittlere Konfidenz** (geringere Konfidenz, weil die Outcome-Behauptung unfalsifizierbar ist).

**Was stimmt:**
- Alle vier mechanistischen Einzelthesen sind real verankerte Notwendigkeitsbedingungen.
- "Umwelt/Curriculum ist ein unterschätzter Hebel" wird von der ML-Frontier geteilt (Hughes et al. 2024).
- Der Mensch dominiert via Kultur/Nische, nicht via skalarem Einzel-IQ.

**Was zu stark / fehlt (stärkster Gegen-Einwand):**
- **Die Dichotomie "Umwelt NICHT Agent" ist empirisch falsch.** In *jedem* demonstrierten Eskalationsfall sind Umwelt und Lernmaschinerie untrennbar gekoppelt (Baker et al. 2020 brauchten massive RL-Skalierung – "Scale plays a critical role"; POET koppelt Generator UND Transfer-Optimierer). Hughes et al. – die Quelle, die du reklamierst – definieren Open-Endedness *konstitutiv* als Eigenschaft des **Gesamtsystems Lerner+Umwelt**.
- **"Dem Menschen deutlich überlegene Spezies" ist undefiniert und unfalsifizierbar.** Sobald du "Überlegenheit" fix misst, hast du eine externe Fitnessfunktion eingeführt und These 1 verletzt. Open-Endedness ist konstitutiv *divergent*, nicht konvergent auf ein vorab definiertes Ziel. Das verifizierte ToLSim-Paper (2026) besteht die OEE-Tests nur teilweise (neue evolutionary activity bleibt null).
- "Erfindung als Policy-Aktion statt Würfelwurf" unterschätzt: Variation (Mutation) *ist* in jeder offenen Evolution ein Würfelwurf; entscheidend ist, dass *Selektion/Beibehaltung* endogen folgt, nicht dass Variation deterministisch ist.
- Methodischer Einwand: "so ist der Mensch entstanden" ist eine **n=1-Induktion**.

**Zitat-Warnung:** Keine Fabrikate. ToLSim (arXiv:2603.01701, real, März 2026), Soros & Stanley 2014, Hughes et al. 2024, Bonhommeau 2013, Baker et al. 2020 alle verifiziert.

## 3. Was im Originaltext fehlt (priorisiert nach Hebelwirkung)

1. **Transmissionstreue & der Ratchet-Effekt (höchste Priorität).** Kumulative Kultur entsteht NUR oberhalb einer kritischen *Treue-Schwelle*. **Direkter Bezug zu deinem Pilot:** Dass der disembodied random-recombiner das gelernte System schlägt (diff −9.4), ist das Symptom, dass deine soziale Transmission *keinen Treue-Vorteil über blindes Rekombinieren* liefert. (Tomasello 1999; Boyd & Richerson; Henrich 2004 Tasmanien.)
2. **Effektive Populationsgröße, Konnektivität & aktiver Diversitätserhalt.** Kumulative Komplexität skaliert mit Größe/Vernetzung des Pools (Tasmanien-Effekt). Ohne expliziten QD/Novelty-Mechanismus statt reiner Fitness-Maximierung kollabiert die Suche prämatur – exakt dein "floor-dominated / diversity collapse" (0.127→0.029 unter greedy means-ends). (Henrich 2004; Powell et al. 2009; Pugh/Soros/Stanley 2016.)
3. **Das Mess-/Selektionsproblem "überlegen wozu?".** Naive Komplexitätsmaße (DAG-Tiefe, #Entdeckungen) sind durch blinde Rekombination *hackbar*; funktionale/irreduzible Maße zeigen das Gegenteil – du hast das bereits erlebt. Ohne null-kalibrierte, compute-gematchte Baselines misst du *Hacking des Maßes statt echte Kompetenz*. (Taylor et al. 2016; Bedau-Packard activity statistics.)
4. **Koevolutionäre Arms-Races / Red-Queen-Dynamik.** Der am besten belegte Mechanismus für *anhaltenden* endogenen Druck ohne fixe Fitnessfunktion (Self-Play: AlphaGo, Hide-and-Seek). Gegensteuern nötig (Hall of Fame, Diversitätsdruck) gegen zyklisches Wettrüsten. (Van Valen 1973; Dawkins & Krebs 1979; Silver et al. 2017.)
5. **Mutations-/Innovationsrate balanciert gegen Selektionsstärke (Error-Threshold).** Expandierbarer Raum nützt nichts, wenn die Innovationsdynamik nicht im Fenster liegt, in dem Selektion Verbesserungen einklinken kann. *Evolvierbarkeit selbst* ist ein Selektionsziel. (Eigen 1971; A. Wagner 2005/2011; Lenski et al. 2003.)
6. **Compute-/Energie-Budget als harte Schranke.** In deiner Memory belegt: world-update dominiert, 8000 ticks ~9h CPU, Pilots auf 1500 ticks gedeckelt. Offene Eskalation braucht ggf. Generationen/Populationen mehrere Größenordnungen über dem Bezahlbaren.
7. **Weitere Lücken (kurz):** Embodiment/Sensomotorik (Aktions-/Wahrnehmungsraum muss mit-expandieren); intrinsische Motivation/Neugier gegen deceptive Plateaus; Turnover/Tod als Quelle des Selektionsdifferentials; räumliche Struktur + Stigmergie (Umwelt als externalisiertes Gedächtnis); Major Transitions allgemein (C4 = Spezialfall, erfordert Konfliktunterdrückung + neues Vererbungssystem).

**Meta-Punkt:** Unbegrenzte offene Evolution wurde künstlich noch *nie* überzeugend demonstriert. Das realistische, null-kalibrierte Zwischenziel ist **nachweisbar offene kumulative Akkumulation**, nicht "dem Menschen überlegene Spezies".

## 4. Konsequenzen für deinen nächsten Schritt

- **C2 (expandierbarer Raum) als Designprinzip weitgehend übernehmen** – deine bestbelegte These. ABER streiche die zwei falschen Teilbehauptungen: "fixe Kanäle ⇒ keine reichere Sprache" (Gegenteil ist wahr) und "Genomlänge = Komplexitätsdeckel" (ersetze durch *Ausdrucksmächtigkeit der Repräsentation*). Formuliere als *expandierbarer Aktions-/Konstruktionsraum*.
- **C1 umformulieren:** Streiche "zwangsläufig" und "endogen vs. exogen". Operationalisierbar ist *nicht-stationärer, an die eigene Population gekoppelter Selektionsgradient* – konkret via Koevolution/Red-Queen/Self-Play.
- **C3 als Enabling-Bedingung führen, nicht "zentraler Treiber".** Umweltmodifikationen müssen *persistent über Generationen* und *vererbbar* sein (kein Episode-Reset). Plane explizit den kulturellen, nicht nur genetischen Kanal.
- **C4 stark relativieren.** MLS/Gruppenselektion nicht als gesicherten Haupthebel verkaufen. Setze auf den unumstrittenen Teil: kumulative Kultur via **Demografie (Populationsgröße + Konnektivität) und Hochtreue-Transmission**. "Koordination schlägt IQ" → "Koordination + hinreichende Individualkognition sind komplementär".
- **C5-Dichotomie aufgeben.** Open-Endedness = Eigenschaft des *gekoppelten Systems Lerner+Umwelt*. Parallel in Agent/Algorithmus UND Umwelt investieren.
- **Vor dem nächsten Pilot: zuerst die drei fehlenden quantitativen Hebel adressieren** (Transmissionstreue, Diversitätserhalt/QD, null-kalibrierte funktionale Maße). Dein Negativbefund liegt nachweislich nicht am fehlenden expandierbaren Raum, sondern daran, dass (a) soziale Transmission keinen Treue-Vorteil über blinde Rekombination erzeugt und (b) dein Maßstab gegen Null-Baselines nicht standhält. Ziel umdefinieren: **"nachweisbar offene kumulative Akkumulation gegen compute-gematchte Random-Baseline"**.

## 5. Schlüsselquellen (kuratiert, dedupliziert; unsichere Zuordnungen markiert)

**Open-Endedness / ALife (Kern):**
- Lehman & Stanley 2011, *Abandoning Objectives: Evolution Through the Search for Novelty Alone*, Evol. Comp. 19(2):189-223.
- Soros & Stanley 2014, *Identifying Necessary Conditions for Open-Ended Evolution through ... Chromaria*, ALIFE 14. (Vier Notwendigkeitsbedingungen, Bedingung 4 = C2; verifiziert nahezu wörtlich.)
- Wang, Lehman, Clune & Stanley 2019, *POET* (arXiv:1901.01753); Enhanced POET, ICML 2020 (arXiv:2003.08536).
- Baker et al. 2019/2020, *Emergent Tool Use From Multi-Agent Autocurricula* (arXiv:1909.07528), ICLR 2020.
- Taylor et al. 2016, *Open-Ended Evolution* — **zwei reale Varianten**: Artificial Life 22(3):408-423 *und* arXiv:1507.07403; Variante vor Verwendung festlegen.
- Bedau et al. 2000, *Open Problems in Artificial Life*, Artificial Life 6(4).
- Stanley, Lehman & Soros 2017, *Open-endedness: The last grand challenge*; Stanley & Lehman 2015, *Why Greatness Cannot Be Planned*.
- Hughes et al. 2024, *Open-Endedness is Essential for Artificial Superhuman Intelligence* (arXiv:2406.04268).
- ToLSim: de Pinho et al. 2026, arXiv:2603.01701 (real, März 2026).
- Ficici & Pollack 1998 (ALIFE VI) + Cartlidge & Bullock 2004, Evol. Comp. 12(2):193-222 (Disengagement / loss of gradient).
- NEAT: Stanley & Miikkulainen 2002. MAP-Elites: Mouret & Clune 2015 (arXiv:1504.04909); Pugh/Soros/Stanley 2016 (QD-Survey).

**Evolutionsbiologie / Kultur:**
- Odling-Smee, Laland & Feldman 2003, *Niche Construction*, Princeton UP; Laland et al. 1999, PNAS 96:10242.
- Maynard Smith & Szathmáry 1995, *The Major Transitions in Evolution*, OUP; Szathmáry 2015, PNAS 112(33):10104-10111.
- Okasha 2006, *Evolution and the Levels of Selection*; Sober & Wilson 1998, *Unto Others* (MLS1 vs. MLS2).
- Henrich 2015/2016, *The Secret of Our Success*; Tomasello 1999 (Ratchet); Boyd & Richerson 1985/2005.
- Richerson et al. 2016, BBS 39:e30 (kulturelle Gruppenselektion – *kontrovers*). Gegenpositionen: West/Griffin/Gardner 2008, J. Evol. Biol. 21; Pinker 2012; Nowak/Tarnita/Wilson 2010, Nature 466 + Abbot et al. 2011 (137-Autoren-Replik); van Veelen et al. 2023, Evol. Human Sciences (Erstautor: Matthijs van Veelen).
- Derex et al. 2013, Nature 503:389-391 (*durch Reanalysen angefochten*); Henrich 2004 (Tasmanien); Powell et al. 2009, Science.
- Trophischer Level: Bonhommeau et al. 2013, PNAS (heute, ~2.21) **vs.** Ben-Dor et al. 2021, AJPA (Pleistozän-Apex-Prädator). **Latenter Widerspruch** – nur als Nebenbehauptung nutzen.

**Quantitative Kipp-/Hilfsmechanismen (deine Lücken):**
- Eigen 1971 (Quasispecies/Error-Threshold); A. Wagner 2005/2011 (Robustness/Evolvability); Lenski et al. 2003, Nature (Avida).
- Van Valen 1973 (Red Queen); Dawkins & Krebs 1979, Proc. R. Soc. B; Silver et al. 2017, Nature (AlphaGo Zero).
- Oudeyer & Kaplan 2007; Schmidhuber 2010; Klyubin/Polani/Nehaniv 2005 (Empowerment).
- Emergent communication: Resnick et al. 2020 (arXiv:1910.11424); Mordatch & Abbeel 2018.
- Theraulaz & Bonabeau 1999 (Stigmergie); Nowak 2006, Science (5 Regeln der Kooperation).

**Ausdrücklich unsichere / fehlerhafte Zuordnungen (vor Zitation prüfen):**
- **FALSCH:** "Chli & De Wilde 2004, Niche Construction and the Evolution of Complexity" → Autor ist **Tim Taylor**.
- Standish 2003 vs. ~2006 (Parsimonie-Aussage aus arXiv nlin/0604026, nicht nlin/0210027).
- HyperNEAT ">300.000:1"-Kompressionszahl: nicht belegt.
- Caldwell & Millen 2008, England 2013, Hernández-Orallo 2017: Jahres-/Relevanz-Zuordnung unsicher.
- Bedau, Snyder & Packard 1998 (Typ-III-Dynamik in Tierra): Konzept korrekt, Autoren-/Jahreskonstellation prüfen.
