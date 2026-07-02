# Kalibrierungstabelle Physik v2

> GENERIERT aus `artificial_society/environment/physics/calibration.py` via
> `scripts/gen_kalibrierung.py` — nicht von Hand editieren.
> Realitäts-Gate: Spec `docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md` §2.


## Eigenschafts-Dimensionen

| Name | Realer Anker | Quelle |
|---|---|---|
| `brittleness` | Sprödigkeit: 0 = zäh/duktil (Sehne, frisches Holz) … 1 = ideal spröde mit muscheligem Bruch (Glas/Obsidian ≈ 0.95, Feuerstein ≈ 0.9) | Bruchmechanik spröder vs. duktiler Werkstoffe; Lithik (Feuersteinschlagen) |
| `density` | Rohdichte ρ / 5000 kg/m³, geklemmt auf [0,1] (Wasser 1000 → 0.2, Granit 2700 → 0.54) | CRC Handbook / Gesteinskunde-Standardwerte |
| `flammability` | Entflammbarkeit 0..1 qualitativ: 0 = nicht brennbar (Stein), 0.8 = trockenes Holz, 1 = Zunder | Brandverhalten von Naturstoffen (Forst-/Brandschutzliteratur) |
| `grain_fineness` | Gefügefeinheit: 0 = grobkristallin (Granit ≈ 0.15) … 1 = kryptokristallin (Feuerstein ≈ 0.95). Bestimmt muscheligen Bruch und erreichbare Kantenschärfe | Petrologie: Kryptokristallinität von Silex — Grundlage des Feuersteinschlagens |
| `hardness` | Ritzhärte, Mohs-Skala / 10 (Talk 1 → 0.1, Quarz/Feuerstein 7 → 0.7, Diamant 10 → 1.0) | Mohs-Härteskala (Mineralogie-Standard) |
| `ignition_temp` | Zündtemperatur / 1000 °C (Holz ≈ 300 °C → 0.3); 1.0 = praktisch nicht entzündbar | Zündtemperatur-Tabellen (Holz 280–340 °C) |
| `melting_point` | Schmelz-/Sinterpunkt / 2000 °C (Ton sintert ≈ 1000 → 0.5, Quarz ≈ 1670 → 0.84); 1.0 = schmilzt praktisch nicht bzw. zersetzt sich vorher | Keramik-/Petrologie-Standardwerte |
| `moisture` | Wasseranteil 0..1 (Frischfleisch ≈ 0.7, lufttrockenes Holz ≈ 0.12, Wasser = 1.0) | Holzfeuchte-/Lebensmitteltabellen |
| `nutrition` | verwertbare Energie / 400 kcal pro 100 g (mageres Rohfleisch ≈ 140 → 0.35, Beeren ≈ 50 → 0.13) | Nährwerttabellen (USDA) |
| `sharpness` | Kantenschärfe: 0 = stumpf … 1 ≈ frisch geschlagene Obsidianklinge. NUR als Prozessergebnis (Bruch) erzeugbar — kein Startmaterial hat sharpness > 0 | Experimentelle Archäologie: Schärfe geschlagener Steinwerkzeuge |
| `tensile_strength` | Zugfestigkeit / 1000 MPa (Hanffaser ≈ 600 MPa → 0.6, Holz längs ≈ 100 → 0.1, Fels ≈ 10 → 0.01) | Werkstoffkunde-Tabellenwerte |
| `thermal_conductivity` | Wärmeleitfähigkeit / 10 W/(m·K), geklemmt (Granit ≈ 2.8 → 0.28, Holz ≈ 0.15 → 0.02) | CRC Handbook, Wärmeleitfähigkeiten |
| `toxicity` | akute Schädlichkeit bei rohem Verzehr, 0..1 qualitativ (rohes Aas ≈ 0.15 wegen Keimbelastung) | Lebensmittelhygiene: Keimbelastung roher Tierprodukte |

## Startmaterialien

| Name | Realer Anker | Quelle |
|---|---|---|
| `berries` | Beeren: ≈ 50 kcal/100 g, wasserreich, weich; leichte Rest-Toxizität (Wildsammlung) | USDA Nährwerttabelle (Beeren) |
| `carcass` | Tierkadaver: Verbund aus Haut/Sehnen/Fleisch/Knochen — sehr zäh; Nährwert praktisch nur durch Zerteilen (Schneiden) erschließbar | Zoologie/Jagdpraxis: Zerwirken |
| `clay_moist` | Feuchter Ton: weich/plastisch, sintert ab ≈ 1000 °C, sehr feines Gefüge | Keramik-Grundlagen |
| `dry_wood` | Lufttrockenes Holz: ρ ≈ 650, Zündtemp ≈ 300 °C, Zugfestigkeit längs ≈ 100 MPa, zäh (splittert nicht muschelig) | Holztechnik-Tabellenwerte |
| `flint` | Feuerstein/Silex: Mohs ≈ 7, ρ ≈ 2600, kryptokristallin → muscheliger Bruch; klassisches Ausgangsmaterial für Klingen; im Rohzustand NICHT scharf | Petrologie Silex; experimentelle Archäologie Feuersteinschlagen |
| `granite` | Granit: Mohs ≈ 6.5, ρ ≈ 2700 kg/m³, grobkristallin, mäßig spröde — splittert unter starkem Schlag in stumpfe Bruchstücke | Gesteinskunde-Standardwerte (Granit) |
| `plant_fiber` | Bastfaser (Hanf/Lein): Zugfestigkeit ≈ 600 MPa, sehr leicht, gut brennbar | Werkstoffkunde Naturfasern |
| `raw_meat` | Rohes Muskelfleisch: ≈ 140 kcal/100 g, ≈ 70 % Wasser, zäh (niedrige Sprödigkeit), roh leicht keimbelastet | USDA Nährwerttabellen; Lebensmittelhygiene |
| `water` | Wasser: ρ = 1000 kg/m³ — Referenzstoff der Dichte- und Feuchteskala | CRC Handbook |

## Prozesse

| Name | Realer Anker | Quelle |
|---|---|---|
| `cut` | Schneiden: Ertrag steigt mit Kantenschärfe × Härte des Werkzeugs und sinkt mit der Zähigkeit des Ziels; schneidbar sind nur Stoffe, die weicher als das Werkzeug und insgesamt weich sind (Fleisch, Pflanzen, bedingt Holz) — Gestein ist nicht schneidbar, sondern nur schlagbearbeitbar; einen Kadaver mit bloßer Hand zu zerwirken ist nahezu unmöglich, mit Steinklinge effizient | Experimentelle Archäologie: Zerwirken mit Steinklingen |
| `strike` | Hart-Hammer-Perkussion: nur spröde, feinkörnige Gesteine (muscheliger Bruch) liefern scharfe Abschläge; nötige Schlagenergie im Bereich eines kräftigen Handschlags (10–50 J); zähe Stoffe (Holz, Fleisch) zersplittern so nicht | Experimentelle Archäologie: Feuersteinschlagen/Lithik; Bruchmechanik spröder Stoffe |

## Körper-Parameter

| Name | Realer Anker | Quelle |
|---|---|---|
| `carry_capacity` | Dauer-Tragfähigkeit ≈ 30 % des Körpergewichts (Trekking-Richtwert 20–25 %, militärisches Marschgepäck 30–45 % mit Ermüdungsfolgen); skaliert mit Kraft, gedämpft durch Ermüdung | Ergonomie-/Militär-Richtwerte zum Lastentragen |
| `fatigue` | Ermüdung/Erholung, Größenordnungen: ~200 kräftige Schläge bis deutliche Erschöpfung (geübte Steinschläger arbeiten stundenlang); Dauerlast an der Traggrenze über Hunderte Ticks tragbar; Erholung in Ruhe über Dutzende Ticks. [Zeitskala Sim-Tick↔Realzeit bewusst qualitativ, bis die Sim-Integration sie fixiert] | Arbeitsphysiologie (Ermüdung/Erholung beim Lastentragen und repetitiver Arbeit) |
| `strike_energy` | Schlagenergie eines Handschlags mit Werkzeugstein 5–50 J (deckungsgleich mit dem Anker des Prozesses strike: kräftiger Handschlag 10–50 J); skaliert mit Kraft, gedämpft durch Ermüdung | Biomechanik des Hammerschlags; experimentelle Archäologie |
