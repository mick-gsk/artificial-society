# Kultur-Korrektur (Schnitt 1, Schritt 4) — Design

**Datum:** 2026-07-03
**Basis:** main @ `3664a41` (Pläne 1–3b gemergt, 310 Tests)
**Übergeordnete Spec:** `docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md` §5 (Kultur-Korrektur), §7 (Tests), §8 (Risiken)

## 1. Ziel & Prinzip

Der lamarckistische Vererbungspfad wird für den v2-Pfad (`physics_v2=True`) entfernt:
Bei der Geburt darf **kein Gelerntes** vom Elter aufs Kind übertragen werden. Vererbt
werden ausschließlich **Gene** (inkl. des etablierten genetischen Musters: Gene modulieren,
*wie stark* gelernt wird — sie übertragen keinen gelernten Inhalt). Techniken überleben
Generationen künftig **allein über soziales Lernen zu Lebzeiten** (bestehende
Fidelity-/Trust-/Lehr-Mechanik, unangetastet).

**User-Entscheidung 2026-07-03 (genetischer Gewichts-Prior):** *Reiner Zufalls-Init, kein
vererbter Prior.* Kinder starten mit frisch (zufällig) initialisiertem Netz. Der „genetische
Gewichts-Prior" der Spec reduziert sich bewusst auf das bereits Vorhandene — das
Plastizitäts-Gen (moduliert die Lernrate) und Gene, die als Input/Bias ins Netz fließen.
Es wird **kein** neuer Gewichts- oder Init-Prior-Mechanismus gebaut. Das ist die strengste
„nichts gewährt"-Lesart; Fallback bei Nicht-Lernen im Pilot ist laut Spec §8 **nicht** ein
Prior, sondern real-physiologische Zwischensignale — nur datengetrieben nachrüsten.

## 2. Golden-Schutz: v2-gegated, v1 byte-identisch

Die Schnitt-1-Spec §7 bindet: „Golden-Kontrakt (v1) bleibt davon unberührt." Deshalb wird
**jeder** Lamarck-Pfad so umgebaut, dass er nur bei `physics_v2=True` übersprungen wird;
der v1-Pfad (`physics_v2=False`) durchläuft exakt die bisherige Logik inkl. identischer
RNG-Nutzung. Konsequenz:

- v1: keinerlei Verhaltens-/RNG-Änderung → Golden-Trajektorie + Headless-Digest byte-identisch.
- v2: hat keine Golden-Garantie (Spec §7: „v2 erhält eigene Golden erst nach Stabilisierung").
  Der durch das Überspringen verschobene v2-RNG-Strom ist zulässig; die 500-Tick-Baseline
  aus Plan 3b ist Diagnostik, kein Kontrakt.

Damit ist die Korrektur konsistent mit dem gesamten Schnitt-1-Vorgehen (v1 unberührt, v2
hinter Flag) und benötigt **keine** Golden-Regeneration.

## 3. Pfad-Kartierung — verifiziert am Code (Design-Review 2026-07-03)

Der Design-Review hat jeden Pfad am realen Reproduktionsfluss instrumentiert. Aufrufkette
(v1 wie v2 identisch): `step()` → `agent.update()` liefert child_genes →
`spawn_child_from_parent` (`simulation.py:197`) → `evolution.make_child` (`evolution.py:7`)
→ `Agent.spawn_child(...)` **ohne `parent=`** (`evolution.py:31-38`).

### 3a. Real feuernde Lamarck-Pfade (zu gaten)

| ID | Ort | Was kopiert wird (gelernt) | Flag-Quelle |
|----|-----|----------------------------|-------------|
| A1 | `simulation.py:219,221` `inherit_weights_from` | PPO-Netzgewichte, strength 0.55 (+0.4× Zweit-Elter) | `self.physics_v2` |
| A7 | `simulation.py:222-226` | CausalMemory-Sequenzen (10, fidelity 0.70) | `self.physics_v2` |
| A8 | `simulation.py:227-230` | remedy_knowledge (prob. 0.70/Krankheit) | `self.physics_v2` |
| A9 | `simulation.py:231-244` | material_inventory-Discoveries (`mat_`, 0.4×) | `self.physics_v2` |
| A10 | `systems/evolution.py:39-46` `make_child` | EpisodicMemory.resource_memory (letzte 3 + 2) | **`parent.physics_v2`** |

**Kritisches Flag-Timing (A10):** In `make_child` ist `child.physics_v2` noch **False** —
`attach_body(child)` (das es auf True setzt und das Brain frisch als v2-Netz baut) läuft
erst **danach** in `spawn_child_from_parent` (`simulation.py:209/215`). A10 muss daher auf
**`parent.physics_v2`** gaten; ein Gate auf `child`/`self` (EvolutionSystem trägt kein
`physics_v2`) liefe leer. Für A1/A7/A8/A9 ist `self.physics_v2` in `simulation.py` direkt
verfügbar. **Kein Signaturbruch, keine Flag-Durchreichung nötig** — `self.physics_v2` bzw.
`parent.physics_v2` genügen überall.

**Reihenfolge-Anforderung:** `attach_body(child)` (frisches v2-Brain, Zufallsinit) muss
weiter **vor** dem — nun gegateten — A1 laufen; A1 wird nur *übersprungen*, die
attach_body-Reihenfolge bleibt unangetastet. Sonst entstünde eine Hintertür (A1 vor
attach_body würde das frische Netz wieder wegwerfen/überschreiben lassen).

### 3b. Tote Pfade (feuern NIE — verriegeln, nicht gaten)

**A3–A6** (KnowledgeGraph, TheoryOfMind, EmotionalMemory, world_memory) stehen im Block
`if parent is not None:` in `Agent.spawn_child` (`agent.py:417-424`). Der **einzige** Aufrufer
`evolution.make_child` übergibt `parent=` nicht → der Block läuft im gesamten Fluss nie
(empirisch: bei einer v2-Geburt feuerte nur A1). **Konsequenz:** nicht gaten (Aufwand ohne
Wirkung); ein Test „v2-Kind erbt kein ToM" wäre trivial grün und bewiese nichts. Stattdessen
ein **Verriegelungs-Test**: `Agent.spawn_child` wird nie mit `parent` aufgerufen — schlüge
jemand künftig `parent=` durch, würde A3–A6 schlagartig live und bräuchte dann ein
`parent.physics_v2`-Gate.

Zwei weitere **latente, uncalled** Vererbungsmethoden (kein Aufrufer, Agenten haben die
Attribute nicht): `StrategySystem.inherit_from` (`systems/strategy.py:133`),
`EpisodicStrategyMemory.inherit_from` (`agents/episodic_strategy.py:98`). Harmlos, aber ein
Wächter-Test hält fest, dass sie bei Geburt nicht feuern.

### 3c. Bleibt bewusst (kein Lamarck)

- **Genetik-Positivliste:** `genetics.py` `inherit_genes` (16 Gene, fitness-gewichtete
  Mittelung + Mutation), `inherit_strength`/`ensure_strength_gene`. Plastizitäts-Gen bleibt
  der einzige Kanal „Veranlagung zum Lernen".
- **Verwandtschafts-Prior:** `evolution.py:49-51` `child.trust[parent.id]=0.4` ist ein
  **fester** Prior-Wert, kein kopiertes gelerntes Trust (anders als A4, das gelernte
  Trust-*Schätzungen gegenüber Dritten* kopierte). Beibehalten; im Plan als bewusst-nicht-
  Lamarck notieren, damit es nicht versehentlich mitgegatet wird.
- **Lebzeit-Kanäle (unangetastet):** `social_learning.py` (Beobachtung, `Brain.imitate_from`
  = **A2**, `CausalMemory.receive_transmitted`, campfire sharing), `KnowledgeGraph.imitate_from`,
  `language.py` (Token-Konvergenz), `remedy.py` (`share_remedy_knowledge`). A1 (Geburt) und
  A2 (Lebzeit) sind getrennte Methoden — das Gaten von A1 lässt A2 unberührt.

## 4. Grenzfall `_broadcast_death_knowledge` (`simulation.py:247-284`) — belassen

Kein Geburtspfad, sondern ein **Todes-Broadcast** von Causal-Memory/Remedy/Discoveries an
**lebende** Nachbarn im Radius (tribe-/verwandtschaftsgefiltert). Entscheidung: **für v2
belassen** — es ist kein Eltern-Kind-Erbgang, sondern horizontaler Transfer unter anwesenden
Zeitgenossen, kategorisch außerhalb des Ziels „lamarckistische *Geburts*-Vererbung"; es kann
keine generationelle Lamarck-Schleife erzeugen.

**Aber ehrliche Etikettierung (Review-Korrektur):** Es ist **nicht** „≈ soziales Lernen".
Der reguläre Sozial-Lern-Kanal verlangt Ko-Präsenz **plus Beobachtung/Trust/Lehre über
Zeit**; der Todes-Broadcast ist eine **einmalige, sofortige Wahrscheinlichkeits-Kopie** aus
einem Sterbeereignis, an dem der Empfänger nicht partizipiert (keine Beobachtung, kein
Trust-Gewicht). Also: **horizontaler, aber beobachtungsfreier Broadcast — schwächer/
unrealistischer als der reguläre Sozial-Lern-Kanal.** Der Plan dokumentiert das als
**bekannte Abweichung** und legt einen **Backlog-Eintrag** für die spätere
Sozial-Lern-Realismus-Runde an (über Fidelity/Trust/Beobachtung routen oder
Empfänger-Beobachtung voraussetzen). Kein Code-Change in diesem Schnitt.

## 5. Test-Strategie

- **v1-Golden byte-identisch:** bestehende Golden-Trajektorie + Headless-Digest müssen
  unverändert grün bleiben (Nachweis, dass das Gating v1 nicht berührt).
- **v2 erbt nichts Gelerntes — nur die REAL feuernden Pfade (A1/A7/A8/A9/A10):** v2-Sim,
  Elter mit gefülltem Brain / CausalMemory / remedy_knowledge / `mat_`-Discoveries /
  resource_memory; Kind gebären; assert: Brain-Gewichte ≠ Eltern-Gewichte (frisch
  initialisiert), `causal_memory` leer/None, `remedy_knowledge` leer, keine `mat_`-Einträge
  aus dem Elter, `memory.resource_memory` leer. **Kein** „erbt kein ToM/Knowledge"-Test —
  das wäre ein trivial grüner Test eines toten Zweigs (siehe §3b).
- **Tote Pfade verriegeln:** Regressions-Test, dass `Agent.spawn_child` ausschließlich ohne
  `parent`-Argument aufgerufen wird (A3–A6); Wächter-Test, dass `StrategySystem.inherit_from`
  / `EpisodicStrategyMemory.inherit_from` bei Geburt nicht feuern.
- **Gene werden weiter vererbt (Positiv-Kontrolle):** dasselbe v2-Kind trägt gemittelte+
  mutierte Gene beider Eltern (inkl. strength) — die Genetik-Vererbung ist unberührt; auch
  der Verwandtschafts-Prior `trust[parent.id]=0.4` bleibt.
- **Soziales Lernen zu Lebzeiten wirkt weiter (Positiv-Kontrolle):** ein v2-Nachbar kann per
  bestehender Mechanik (`imitate_from`) Gewichte angleichen — der Lebzeit-Kanal ist intakt.
- **Erhaltung/Determinismus:** kurzer v2-Lauf mit Geburten bleibt NaN-frei und
  massenerhaltend (wie 3b-Rauchlauf).

## 6. Nicht-Ziele (YAGNI)

- Kein neuer genetischer Init-/Gewichts-Prior-Mechanismus (User-Entscheidung §1).
- Keine Golden-Regeneration (v1 bleibt byte-identisch).
- Kein Umbau der Lebzeit-Lernkanäle (soziales Lernen, Sprache, Imitation bleiben).
- Kein Entfernen der v1-Lamarck-Pfade (v1 ist eingefroren; die Pfade werden nur bei v2
  übersprungen, nicht gelöscht).
