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
| `carcass` | Tierkadaver: Verbund aus Haut/Sehnen/Fleisch/Knochen — sehr zäh; Nährwert praktisch nur durch Zerteilen (Schneiden) erschließbar. nutrition 0.14 = essbarer Anteil ~40 % (dressed yield) × Rohfleisch 0.35 — die intensive nutrition mittelt über Knochen/Haut/Innereien; toxicity 0.02 = frisches Fleisch nahezu unbedenklich, Gefahr entsteht erst über Verwesung | Zoologie/Jagdpraxis: Zerwirken, Schlachtausbeute (dressed yield ~40 %) |
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
| `body_mass` | Default-Körpermasse 70 kg (erwachsener Mensch, Referenzperson); bestimmt Tragkapazität (~30 % davon) und die Kadaver-Masse beim Tod (Spec B3/B5: 70-kg-Kadaver ≈ 1250 Sim-Energie ≈ 5× MAX_ENERGY) | Anthropometrie: ICRP-Referenzperson ~70 kg |
| `carry_capacity` | Dauer-Tragfähigkeit ≈ 30 % des Körpergewichts (Trekking-Richtwert 20–25 %, militärisches Marschgepäck 30–45 % mit Ermüdungsfolgen); skaliert mit Kraft, gedämpft durch Ermüdung | Ergonomie-/Militär-Richtwerte zum Lastentragen |
| `fatigue` | Ermüdung/Erholung, Größenordnungen: ~200 kräftige Schläge bis deutliche Erschöpfung (geübte Steinschläger arbeiten stundenlang); Dauerlast an der Traggrenze über Hunderte Ticks tragbar; Erholung in Ruhe über Dutzende Ticks. [Zeitskala Sim-Tick↔Realzeit bewusst qualitativ, bis die Sim-Integration sie fixiert] | Arbeitsphysiologie (Ermüdung/Erholung beim Lastentragen und repetitiver Arbeit) |
| `hands` | Zwei Hände, je Hand ein gehaltenes Objekt; Gesamtlast innerhalb der Tragkapazität. Ohne erfundene Behälter ist Transport damit auf 2 Objekte pro Weg begrenzt — der reale Druck, aus dem Behälter/Bündel entstanden sind | Menschliche Anatomie; Archäologie früher Trage-/Behältertechnik |
| `strike_energy` | Schlagenergie eines Handschlags mit Werkzeugstein 5–50 J (deckungsgleich mit dem Anker des Prozesses strike: kräftiger Handschlag 10–50 J); skaliert mit Kraft, gedämpft durch Ermüdung | Biomechanik des Hammerschlags; experimentelle Archäologie |

## Spawn-Parameter (Vorkommen)

| Name | Realer Anker | Quelle |
|---|---|---|
| `clay_moist` | Ufer-Lehm 0.5–5 kg in Sumpf und an Ufern (Nicht-Wasser-Zelle mit Wasser-Nachbar) | Sedimentologie: Ton-/Lehmablagerungen an Gewässerrändern |
| `dry_wood` | Totholz-Äste 0.5–6 kg im Wald | Forstökologie: Totholzaufkommen in Wäldern |
| `flint` | Feuerstein 0.3–4 kg: Knollen im Gebirge (rate_mult 0.5), selten als Kiesel im Grasland (rate_mult 0.05) | Geologie: Feuerstein-Knollen in Kreide/Schotterfluren |
| `granite` | Granit-Gerölle 0.5–8 kg im Gebirge (Lesesteine/Hangschutt) | Geologie: Hangschutt/Lesesteine im Mittelgebirge |
| `initial_density` | Start-Seeding: 3 % der geeigneten Biom-Zellen tragen initial ein Objekt (SPAWN_INITIAL_DENSITY = 0.03, je Quelle skaliert mit rate_mult) | Größenordnung Oberflächen-Vorkommen von Lesesteinen/Totholz; Pilot-feinjustierbar (Spec B2) |
| `plant_fiber` | Gras-/Bastbündel 0.05–0.4 kg in Grasland und Sumpf | Ethnobotanik: Sammelmengen Faserpflanzen |
| `regen_rate` | Regeneration 1e-5 Objekte je Zelle und Tick ≈ 0.0024/Zelle/Tag (240 Ticks/Tag); auf 200×200 mit ~15 % Gebirge ≈ 14 neue Steine/Tag — versiegt nicht, flutet nicht | Auslegungsrechnung Spec B2 (Pilot-feinjustierbar, nie zur Laufzeit pro Agent) |

## Aktions- & Kopplungs-Parameter

| Name | Realer Anker | Quelle |
|---|---|---|
| `bite` | Ein Biss pro Tick: bite = min(0.3 kg, Restmasse); Objekte mit nutrition ≤ 0.02 sind wirkungslos (Steinbeißen = No-op) | Ernährungsphysiologie: Bolus-/Bissgrößen, Größenordnung Mahlzeit ~0.3–1 kg |
| `blade_mass_factor` | Klingen-Massen-Faktor beidseitig begrenzt: clamp(sqrt(m/0.15), 0.2, 1) · clamp((2−m)/1, 0.2, 1) — brauchbare Abschläge ≥ ~150 g (BLADE_MASS_REF), einhändig führbar ≤ ~1 kg (BLADE_HANDLE_MAX); 20-g-Splitter und 4-kg-Brocken schneiden nur mit Faktor ≤ 0.4 bzw. 0.2 (kein 8-kg-Skalpell) | Ethnographie/experimentelle Archäologie der Lithik: Handhabbarkeit von Schneidwerkzeug |
| `cut_work` | Schneidearbeit CUT_WORK_J = 15 + 35·effort Joule pro Schneidevorgang, über den Ermüdungspfad (FATIGUE_PER_JOULE) und die Arbeits-Metabolik verbucht; der Schnitt-Ertrag skaliert mit Effort: yield × (0.5 + 0.5·effort) | Arbeitsphysiologie: repetitives Schneiden/Zerwirken — Zerlegegeschwindigkeit skaliert mit aufgebrachter Kraft |
| `decay` | Verwesung: mass·(1−λ) und nutrition·(1−λ) pro Tick mit λ = ln(2)/(10 Tage · 240 Ticks) ≈ 2.89e-4 (Weichgewebe-Halbwertszeit temperiert ~7–14 Tage); wirkt nur auf feuchte, nahrhafte Stoffe (moisture ≥ 0.5, nutrition > 0 — Stein/Holz verwesen nicht) | Forensische Taphonomie: Weichgewebe-Dekomposition |
| `muscle_efficiency` | Brutto-Wirkungsgrad Skelettmuskel 0.25 (20–25 %); mechanische Arbeit kostet joules/0.25/4184 kcal ≙ ×0.032 Sim-Energie (200 Schläge à 45 J ≈ 0.3 Sim-Energie — bewusst klein, der reale Begrenzer ist die Ermüdung) | Arbeitsphysiologie: Wirkungsgrad Muskelarbeit |
| `sim_energy_per_kcal` | Kopplung kcal↔Sim-Energie: 0.032 Sim-Energie/kcal — eine 1-kg-Fleischmahlzeit (0.35·4000 = 1400 kcal) ergibt ≈ 45 ≙ v1 MEAT_ENERGY; Gate prüft das PRODUKT nutrition-Konvention × Kopplung (± 1) | USDA-Nährwerttabellen + v1-Energieökonomie (MEAT_ENERGY 45, MAX_ENERGY 240) |
| `spoilage` | Verderb: toxicity += 5e-4 pro Tick, gekappt bei 0.6 — rohes Fleisch wird bei Umgebungstemperatur in ~3–5 Tagen gefährlich (Kappe nach ~5 Tagen ≙ 1200 Ticks) | Lebensmittelhygiene: Verderb roher Tierprodukte ungekühlt |
| `toxin_damage` | Toxin-Schaden 20 Health/kg·toxicity: 0.3 kg stark toxischen Materials (0.8) ≈ 5 Health ≙ spürbar, wiederholt tödlich; verdorbenes Fleisch (0.6) ≈ 3.6 Health/Biss | Lebensmittelhygiene/Toxikologie: Dosis-Wirkung roher, verdorbener Tierprodukte |
| `v_max_strike` | Maximale Schlag-Endgeschwindigkeit 14 m/s; gelieferte Energie ≤ ½·m·v² — ein 0.05-kg-Kiesel liefert damit max ~4.9 J (kein Flint-Knacken über Kiesel-Exploit), ein 0.5–2-kg-Schlagstein die vollen 49–50 J | Biomechanik Hammerschlag/Knapping: Endgeschwindigkeit 10–15 m/s |
