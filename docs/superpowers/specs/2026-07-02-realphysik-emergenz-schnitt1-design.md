# Design: Realphysik-Emergenz, Vertikaler Schnitt 1 — „Die erste Klinge"

**Datum:** 2026-07-02
**Status:** Design freigegeben (abschnittsweise mit User validiert), Implementierung noch nicht begonnen
**Kontext:** Der Forschungs-/Paper-Strang ist beendet (User-Entscheidung 2026-07-02). Alles Gebaute bleibt
erhalten. Neues Projektziel: die Simulation so ausbauen, dass Agenten — wie einst die Menschheit —
Fähigkeiten *erlernen und kumulativ weiterentwickeln*, die ihnen niemand einprogrammiert oder gezeigt hat
(Werkzeuge, später Sprache und Bauen). Die Diagnostik der beendeten Forschung dient als Landkarte der
Blocker: (1) Erfindungs-*Auswahl* ist gescriptet statt gelernt, (2) hardcodierte Shaping-Boni statt
Überlebensnutzen, (3) geschlossenes 57-Dim-Interface vs. offener Materialraum, (4) lamarckistische
Wissensvererbung.

## Getroffene Grundsatzentscheidungen (mit User geklärt)

| Frage | Entscheidung |
|---|---|
| Realismus-Grad | **Kalibrierte Physik**: Welt bleibt abstrakt (Eigenschaftsvektoren), aber an realer Physik/Chemie geankert; jede Erfindung hat ein plausibles reales Gegenstück auf Abstraktionsebene |
| Realismus-Scope | **Welt + Körper**: auch reale Körper-Constraints (Tragkraft, Reichweite, Kraft, Ermüdung); Gehirn bleibt abstrakt (neuronales Netz) |
| Physik-Basis | **Fundament tauschen**: Architektur (Eigenschaftsvektoren + DiscoveryRegistry + Reaktionspfade + Systeme) bleibt, Inhalt wird real kalibriert; Golden-Trajectory wird beim Umschalten einmal bewusst neu gesetzt |
| Erste Fähigkeit | **Werkzeuge/Erfinden zuerst** (Sprache/Bauen später — Sprache braucht erst etwas Kommunikationswürdiges) |
| Vorgehen | **Ansatz C, vertikaler Schnitt**: schmale, aber echte Physik + Embodiment + Lern-Kopplung end-to-end, dann schichtweise verbreitern |
| Gehirn-Vererbung | **Genetischer Prior, kein Kopieren**: Kinder erben KEINE gelernten Gewichte; Start von einem genetisch kodierten Prior, der sich nur über Selektion + Mutation entwickelt (darwinistisch) |
| Lernsignal | **Nur intrinsische Neugier** zusätzlich zum Überlebenssignal (Novelty + Causal-Model-Vorhersagefehler); keinerlei Erfolgs-/Entdeckungs-Boni |

## 1. Zielbild & Meilenstein 1

**Meilenstein 1 — „Die erste Klinge":** In einer frischen Welt, ohne einprogrammierte Rezepte, entdecken
Agenten die Kette *Feuerstein schlagen → scharfe Kante → Schneiden bringt deutlich mehr Nahrung/Material*.

Erfolgskriterien (alle drei müssen gleichzeitig gelten):

1. **Gelernt, nicht getriggert.** Die Entscheidung, *was* kombiniert wird, trifft die Policy des Agenten,
   nicht ein Skript. Es gibt keine Entdeckungs-/Technologie-Boni; ein Werkzeug lohnt sich ausschließlich,
   weil es mechanistisch das Überleben verbessert.
2. **Kulturell weitergegeben, nicht vererbt.** Die Technik verbreitet sich über *soziales Lernen zu
   Lebzeiten* (Zusehen, Nachahmen, Lehren — auch Eltern→Kind während gemeinsamer Lebenszeit) und übersteht
   Generationswechsel *nur* durch diese Weitergabe. Der lamarckistische Vererbungspfad für Gelerntes wird
   entfernt (siehe §5).
3. **A/B-belegt.** Eine Population mit Lernen schlägt eine Vergleichspopulation ohne, über mehrere Seeds.
   (Genau der Test, an dem das alte System dreifach gescheitert ist — deshalb wird dieses Kernrisiko
   zuerst und am billigsten getestet.)

## 2. Realitäts-Gate (Anforderung „muss real möglich sein", als Mechanismus)

- **Kalibrierungstabelle** (`docs/physics/kalibrierung.md` o.ä., Format bei Implementierung fixiert):
  Jede Eigenschafts-Dimension, jedes Startmaterial und jeder Prozess bekommt einen Eintrag mit realem
  Anker (z.B. „Feuerstein: Mohshärte ~7, muscheliger Bruch → scharfe Kanten; Quelle").
- **Automatisierter Merge-Check:** Ein Test/CI-Check verweigert Änderungen, die Dimensionen, Materialien
  oder Prozesse ohne Kalibrierungseintrag hinzufügen. Realismus ist damit Merge-Bedingung, nicht
  Absichtserklärung — dauerhaft, auch für spätere Beiträge anderer Agenten/Sessions.
- **Abstraktionsebene (ehrlich benannt):** mesoskopisch. Eigenschaften und Prozess*bedingungen* sind real
  kalibriert, Erhaltungssätze (Masse, Energie) werden nie verletzt; es wird keine Molekulardynamik
  gerechnet und keine Labormessgenauigkeit behauptet. Kein Material/Prozess ohne plausibles reales Vorbild.

## 3. Physik v2 (schmal starten)

**Reale Eigenschafts-Dimensionen** (ersetzen die 12 abstrakten inkl. Fantasy-Dims wie „danger"/„light" als
Materialeigenschaft; endgültige Liste bei Implementierung, Startumfang:) Härte, Dichte,
Zähigkeit/Sprödigkeit, Zugfestigkeit, Schärfe (abgeleitete Größe, kein setzbares Attribut), Brennbarkeit,
Zündtemperatur, Schmelzpunkt, Wärmeleitfähigkeit, Nährwert/Essbarkeit, Toxizität, Wassergehalt. Das Schema
wird als Superset angelegt (spätere Layer ergänzen Werte, nicht Struktur).

**Startmaterialien (Schnitt 1):** Holz, Stein (z.B. Granit), Feuerstein, Pflanzenfaser, Ton, Wasser,
Nahrung pflanzlich/tierisch (inkl. Kadaver). Reale Eigenschaftswerte laut Kalibrierungstabelle. Kein
Startmaterial ist ein Rezept-Endprodukt; die 5 Legacy-Rezepte existieren in der v2-Welt nicht.

**Prozesse brauchen reale Bedingungen** (ersetzt Label-Lookups):
- **Schlagen** (Schnitt 1, vollständig): sprödes Material mit muscheligem Bruch + ausreichende
  Aufprallenergie (∝ Kraft des Agenten, Masse des Schlagsteins) → Fragmente, deren Schärfe sich aus
  Sprödigkeit/Bruchverhalten ergibt. Weiches/zähes Material zersplittert nicht → keine Klinge.
- **Schneiden** (Schnitt 1, vollständig): Ertrag = f(Schärfe & Härte des Werkzeugs vs. Zähigkeit des
  Ziels). Mit bloßer Hand am Kadaver: sehr geringer Ertrag; mit scharfer Kante: deutlich mehr.
- **Feuer / Erhitzen / Kochen** (Layer 2, erst nach bestandenem A/B-Test): Feuer braucht Brennstoff
  (brennbar + trocken) + Zündenergie + Sauerstoff, verbraucht Brennstoff (Erhaltung). Kochen wirkt über
  reale Temperatureffekte (Toxizität ↓, Nährwert-Verfügbarkeit ↑, Verderb ↓), kein „ist_gekocht"-Flag.

**Erhaltungssätze:** Masse der Fragmente = Masse des Ausgangsmaterials; Energieflüsse bilanziert (in der
Linie von Phase 4). Emergente Outputs laufen weiterhin durch die DiscoveryRegistry (neuartige Vektoren →
neue IDs) — die Offenheit des Erfindungsraums bleibt architektonisch erhalten.

**Koexistenz:** Physik v2 lebt hinter einer Welt-Konfiguration (`physics=v2` o.ä.) neben v1. v1-Golden
bleibt bis zum bewussten Umschalten grün; v2 bekommt bei Stabilität eine eigene Golden-Trajectory.

## 4. Embodiment (Körper v1) — erzeugt den Erfindungsdruck

Begrenzte **Tragkraft** (Masse-Budget, Tragen kostet Energie ∝ Masse), begrenztes **Greifen** (zwei
„Hände"/gehaltene Objekte), begrenzte **Reichweite** (nur aktuelle Zelle), **Kraft** (∝ Gene/Alter,
bestimmt Aufprallenergie beim Schlagen), **Ermüdung** (schwere Aktionen erhöhen sie, senkt Wirksamkeit,
erholt sich bei Ruhe), **Verletzbarkeit**. Kalibrierungsziel: „Schneiden mit Kante ≫ mit bloßer Hand"
muss ein messbarer, für die Policy spürbarer Unterschied sein — Werkzeuge lohnen sich genau deshalb,
weil der nackte Körper an realen Grenzen scheitert (Antwort auf den in der Forschung identifizierten
Embodiment-Confound).

## 5. Gehirn-Kopplung — Erfinden wird gelernt

Alle Änderungen flag-gated; bestehende Golden bleibt bis zum bewussten Umschalten grün.

- **(a) Policy wählt das *Was*.** Der gescriptete Suchbaum (`systems/invention.py` /
  `need_driven_invention.py`) wird für die v2-Welt stillgelegt. Die Policy wählt Objekt A, Prozess,
  Objekt B. Auswahl per **Sampling** (Softmax über erwarteten Wert/Informationsgewinn), explizit NICHT
  greedy-argmax (der dokumentierte C+D-Fehlschlag: Diversitätskollaps). Der vorhandene
  **Causal-Model-Prototyp** (`AS_CAUSAL_MODEL`, Branch `feat/research-causal-model-proto`, uncommitted)
  ist der Baustein: agenten-eigenes gelerntes Vorhersagemodell „Eigenschaften von Ergebnis(A, B, Prozess)",
  liefert gerichtete Auswahl + Vorhersagefehler als Neugiersignal.
- **(b) Wahrnehmung über Eigenschaften, nicht IDs.** Gehaltene Objekte und Zellen-Objekte werden dem
  Gehirn als reale Eigenschaftsvektoren präsentiert (Top-k-Fenster, Größe bei Implementierung fixiert).
  Damit generalisiert Gelerntes auf nie gesehene Materialien („scharf + hart schneidet gut") — die
  Voraussetzung für Offenheit über die Startmenge hinaus.
- **(c) Belohnung rein überlebensbasiert + Neugier.** Alle Shaping-Boni (+0.5 Entdeckung, +1.0
  Technologie, +0.3 Heilmittel, +0.4 Makro-Replay u.ä.) entfallen in v2. Wert entsteht ausschließlich
  mechanistisch (mehr Nahrung → Energie → Überlebensbelohnung aus Homöostase). Einziger Zusatzantrieb:
  intrinsische Neugier (Novelty + Vorhersagefehler) — belohnt Erkunden, nicht Erfolg-per-Dekret.

**Kultur-Korrektur (lamarckistische Pfade entfernen):** `spawn_child_from_parent` vererbt heute den
gelernten Wissensgraphen (`inherit_from`, Stärke 0.7), Episodenerinnerungen und Gehirngewichte — real
unmöglich. Neu: **genetisch vererbt** werden nur Körperparameter, Gene/Veranlagungen und der genetische
Gewichts-Prior (evolviert via Selektion + Mutation). **Nicht vererbt:** Wissensgraph, Makros,
Episodenerinnerungen, Trust, gelernte Gewichte. Techniken überleben Generationen ausschließlich über
soziales Lernen zu Lebzeiten (bestehende Fidelity-/Trust-/Lehr-Mechanik bleibt und wird auf
v2-Prozesssequenzen portiert).

## 6. Bau-Reihenfolge (jeder Schritt lauffähig + getestet)

0. **Abhängigkeit:** setzt auf dem Ergebnis der laufenden Konsolidierung auf (Phase-4/5-Merges,
   Golden-Regen durch parallelen Agenten). Bis dahin: kein Implementierungs-Code aus diesem Design.
1. Kalibrierungstabelle + Physik v2 schmal (Dimensionen, Startmaterialien, Prozesse Schlagen+Schneiden).
2. Embodiment v1 (Tragkraft/Greifen/Reichweite/Kraft/Ermüdung), Kalibrierung des Werkzeug-Deltas.
3. Lern-Kopplung (a)–(c) inkl. Causal-Model-Integration und Entfernen der Shaping-Boni (v2-Welt).
4. Kultur-Korrektur (genetischer Prior, lamarckistische Pfade raus).
5. **Meilenstein-1-Pilot:** Emergenz-A/B auf dem GPU-PC. Erst nach Bestehen: Layer 2 (Feuer/Kochen).

## 7. Tests

- **Physik-Unit-Tests (deterministisch, dauerhaft):** u.a. „weiches Material zersplittert nicht",
  „Feuer ohne Brennstoff/Trockenheit/Zündenergie bleibt aus", „Massen-/Energiebilanz stets erfüllt",
  „Schärfe nur als Prozessergebnis erzeugbar".
- **Realitäts-Gate-Check (CI):** neue Dimension/Material/Prozess ohne Kalibrierungseintrag → rot.
- **Emergenz-A/B (Meilenstein-Test, Forschungscharakter):** Lernen ON vs. OFF, mehrere Seeds, GPU-PC.
  Darf rot sein, bis der Mechanismus greift; misst Fortschritt statt Byte-Identität. Golden-Kontrakt
  (v1) bleibt davon unberührt; v2 erhält eigene Golden erst nach Stabilisierung.

## 8. Risiken & Fallbacks

- **Spärliches Signal** (reine Neugier lernt evtl. zu langsam): Fallback ist NICHT ein Erfinden-Bonus,
  sondern real-physiologische Zwischensignale (Sättigung beim Essen, Schmerz) — nur datengetrieben
  nachrüsten, falls der Pilot Nicht-Lernen zeigt.
- **Lebensspanne/Kindheit vs. Kultur-Übertragung:** ohne Gewichts-Kopieren muss die gemeinsame Lebenszeit
  Alt/Jung für Übertragung reichen; ggf. Reifungs-Fenster (lernbeschleunigte Kindheit) als realistische
  Erweiterung. Beantwortet der Meilenstein-1-Pilot.
- **Performance:** Welt-Update ist der dokumentierte Bottleneck (55–84 % Laufzeit, pure-Python); lange
  Läufe auf dem GPU-PC (CPU), Vektorisierung (Perf-Tier 1/2/4) bleibt separater Strang.
- **Rework beim Verbreitern:** bewusst akzeptiert (Preis des vertikalen Schnitts); Schema als Superset
  minimiert Strukturbrüche.

## 9. Bewusst außerhalb dieses Schnitts (YAGNI)

Sprache/Kommunikation (emergiert erst sinnvoll, wenn es Kommunikationswürdiges gibt), Bauen aus realen
Materialien (ersetzt später die abstrakten Camp/Brunnen/Farm-Level), Ökonomie-De-Scripting,
Performance-Vektorisierung, Live-Viz-Rebase. Jeweils eigener Design-/Plan-Zyklus.

## 10. Offene Punkte (bei Implementierungsplanung zu fixieren)

- Endgültige Dimensionsliste + Normierung (Anker-Skalen) der Kalibrierungstabelle.
- Aktionsraum-Mechanik der Objektwahl (Attention/Pointer über Eigenschaftsvektoren) im Detail.
- Top-k-Fenstergrößen der Objekt-Wahrnehmung.
- Format/Ort der Kalibrierungstabelle + Implementierung des CI-Checks.
- Umgang mit v1-Systemen, die auf alte Dimensionen zeigen (`material_reward`, Remedy, Conductivity) in
  der v2-Welt: portieren vs. vorerst deaktivieren — Entscheidung pro System im Implementierungsplan.
