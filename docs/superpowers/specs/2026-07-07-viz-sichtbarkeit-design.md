# Design: Sichtbarkeit — vorhandenes Sim-Geschehen in die Live-Viz bringen

**Datum:** 2026-07-07 · **Branch:** `feat/infra-live-viz-v3` (auf `feat/infra-live-viz-v2` @ 052edc6)
**Ziel:** Die Welt wirkt leer/statisch, obwohl die Simulation mehr berechnet als sie zeigt. Wir schließen ausschließlich Sende- und Render-Lücken für **real laufende** Sim-Systeme. Es wird nichts visualisiert, was die Simulation nicht tatsächlich berechnet ("Nichts erfinden"-Constraint, User-Vorgabe).

## 0. Geltungsbereich und Ehrlichkeits-Grenzen

Verifiziert am Code dieses Branches (nicht übernommen aus Annahmen):

| System | Läuft live? | Beleg |
|---|---|---|
| Jahreszeiten | JA — Registry order 10, publiziert `sim._season_state` | `systems/_builtins.py:75`, `_tick_seasons` |
| Wetter (wind/storm_risk/temperature_shift/rain_map) | JA — Registry order 20, `sim._weather_state` | `systems/_builtins.py:76` |
| Umwelt-Events (drought/storm/fire/blight) | JA — `world_regrowth` (order 25) ruft `world.update_environment` | `systems/world_regrowth.py:24` |
| Zell-Felder plant_food, soil_fertility, pollution, usage_pressure, carcasses | JA — `world.F` NumPy-Felder, jeden Tick | `cell_store.py:24-40` |
| Handel | JA — `economy.maybe_trade(self, agents)` im Agenten-Update | `agents/agent.py:1235` |
| Stimmung (mood/mood_label) | JA — `emotional_memory`, bisher nur im Inspector-Detail | `serve/frame.py:433-442` |
| Kräuter | JA — `regrow_herbs`, `collect_herb` | `world.py:8`, `agents/agent.py:19` |

**Explizit AUSSER Scope** (Sim berechnet es nicht → Darstellung wäre Erfinden):

- `systems/world_objects.py` (Lagerfeuer/Werkstatt/Wissensstein): Registry wird **nirgendwo instanziert** — totes Code-Gewebe. Keine Darstellung. (Kandidat für spätere Sim-Vertiefung.)
- Windrichtung: `WeatherSystem.wind` ist eine **richtungslose Stärke** (`weather.py:12`). Die Partikel-Windrichtung bleibt Client-Präsentation; nur die **Stärke** wird datengetrieben.
- Flächenbrand-Ausbreitung, räumliches Regenfeld, Fauna/Beutetiere, Treffer-Effekte im Kampf: existieren im Kern nicht.
- `systems/trade.py` (TradeProposal-Maschinerie): wird nicht getickt; die Handels-Quelle ist ausschließlich `economy.maybe_trade`.

## 1. Architektur-Rahmen

- **Frame-Schema v2 wird abwärtskompatibel erweitert**: nur neue, optionale Keys. Alte Clients ignorieren sie; der neue Client behandelt fehlende Keys als "Feature aus" (defensive Reads wie bisher in `world-render.js`).
- **Backend**: alle Änderungen in `serve/frame.py` + ein minimaler, verhaltensneutraler Beobachtungs-Hook in `systems/economy.py` (append-only Event-Log, capped; ändert keine Entscheidungslogik). Dieser eine Nicht-`serve/`-Edit wird beim Merge dem core-lead geflaggt (wie beim v2-`agent.py`-Edit vereinbart).
- **Frontend**: Svelte/PixiJS wie gehabt (`frontend/src/lib/*`, Build → `serve/static/`).
- **Tests**: pytest `tests/serve/` für jedes neue Frame-Feld (Schema, Caps, Throttling, Abwesenheits-Fall); vitest für Feed-/Marker-Logik. Kein Golden-Impact: Sim-Verhalten bleibt byte-identisch (reines Lesen + Logging).

## 2. Welle A — Draht reparieren (senden, was schon gerendert werden kann)

### A1 Wetter ins Frame
`frame["weather"] = {wind, storm_risk, temp_shift, rain}` aus `sim._weather_state` (gerundet, 4 Floats, jeder Frame — vernachlässigbare Größe). Frontend:
- `particles.js` skaliert Regen-/Rauch-/Staub-Drift mit realer `wind`-Stärke statt des bisherigen Fake-Sinus (Richtung bleibt client-seitig, s. §0).
- Terrain-Shader: vorhandene Vegetations-Sway-Amplitude an `wind` koppeln.
- HUD: kleines Wetter-Chip (Windstärke, Sturmrisiko) neben der Tick-Anzeige.

### A2 Jahreszeit ins Frame
`frame["season"] = {name, phase}` aus `sim._season_state`. Frontend Welle A: Saison-Chip im HUD (Name + Fortschrittsbogen). Der Terrain-Look folgt in Welle B.

### A3 Neue Analyse-Overlays
`_ALLOWED_LAYERS` in `serve/frame.py` um `fertility` (soil_fertility), `pollution`, `pressure` (usage_pressure), `carcasses` erweitert — gleiche `_quantize_layer`-Pipeline, gleiche 10-Tick-Throttle, weiterhin nur auf Anforderung (`layer_names`), also kein Default-Overhead. Frontend: vier neue Einträge in `OVERLAY_NAMES` mit passenden Farbrampen (dataviz-konform, mit den bestehenden 6 konsistent).

### A4 Handels-Ereignisse
- Hook: `economy.maybe_trade` appendet bei erfolgreichem Tausch `{tick, a, b, give_mat, give_qty, get_mat, get_qty, x, y}` an `sim.social_events` (deque, maxlen 64; existiert das Attribut nicht, wird nichts geloggt → alt-kompatibel).
- Frame: `frame["social"] = [...]` nur die Events der letzten Frame-Periode (delta seit letztem gesendetem Tick), cap 16/Frame.
- Frontend: Feed-Typ „Tausch" („#12 tauschte Holz gegen Beeren mit #7"), mit Feed-Cap wie v2; auf der Karte kurzer Austausch-Marker zwischen beiden Agenten (zwei gegenläufige Item-Punkte entlang der Verbindungslinie, ~1s), damit Handel von generischer Kooperation unterscheidbar ist.

## 3. Welle B — Neue Visuals auf realen Daten

### B1 Jahreszeiten-Look
Saisonale Terrain-Tönung im Shader (`terrain-shader.js`): dezente Palette-Verschiebung per `season.name` + `phase`-Überblendung (Frühling satter/heller, Herbst wärmer/brauner, Winter entsättigt/kühler). Keine Schnee-Partikel (Schnee wird nicht simuliert); die Tönung visualisiert den realen `temperature_shift`/`food_factor`-Zustand.

### B2 Stimmungs-Darstellung auf der Karte
Neues optionales Agenten-Feld `md` (mood quantisiert auf int8-Skala, nur gesendet wenn |mood| > 0.15 → spärlich). Frontend: dezente Färbung des vorhandenen Emote-/Figuren-Renders (z.B. Glanz kühl bei negativ, warm bei positiv), nur bei deutlicher Stimmung — keine Dauer-Auren, sonst Rauschen. Inspector zeigt weiterhin das Detail.

### B3 Kräuter unterscheidbar
`_ground_items` klassifiziert Herb-Materialien (aus `HERB_DEFINITIONS`) als eigene ITEM-Klasse; eigenes Sprite (Blatt/Blüte) statt generischem Material-Punkt. Reine Klassifikations-/Sprite-Arbeit auf real liegenden Boden-Items.

### B4 Handel-Polish
Aufbauend auf A4: Inspector/Chronik-Eintrag beim selektierten Agenten („hat mit #7 getauscht"), Chronik nutzt den bestehenden v2-Mechanismus.

## 4. Fehlerfälle & Performance

- Fehlende neue Keys (alter Server/neuer Client oder umgekehrt) → Feature stumm aus; kein Throw. Jeder neue Read defensiv (`frame.weather ?? null`).
- `sim._weather_state`/`_season_state` fehlen (Registry aus, z.B. in Alt-Checkpoints) → Keys werden weggelassen (getattr-Guards), Tests decken den Abwesenheits-Fall.
- Frame-Größenbudget: +~40 Bytes fix (weather/season) + `social` (cap 16 × ~9 Zahlen) + `md` spärlich. Layers unverändert on-demand/throttled.
- `social_events`-Deque capped (64) → kein Speicherwachstum in Langläufen.

## 5. Test- & Abnahme-Plan

1. pytest `tests/serve/`: neue Felder vorhanden/korrekt, Abwesenheits-Fälle, Caps, Kompatibilität (Frame ohne Registry-Systeme).
2. vitest: Feed-Tausch-Einträge (Cap, Formatierung), Wind-Skalierung (particles), Overlay-Registrierung.
3. Bestands-Suite bleibt grün (keine Sim-Verhaltensänderung; der economy-Hook ist reines Logging → bestehende Golden/Tests unberührt).
4. Manuelle Verifikation: lokaler Headless-Run + Dashboard, danach Deploy auf GPU-PC :8000 (bestehender `run-server.cmd`-Weg); Sichtprüfung: Wetter-/Saison-Chip, mind. ein Tausch im Feed, neue Overlays schaltbar, Saison-Tönung im Zeitraffer.
5. Review-Gates: pro Welle Code-Review durch Review-Agenten (Team-Konvention), kein User-Gate außer Richtungsfragen.

## 6. Reihenfolge

Welle A komplett (A1–A4, ein PR-fähiger Stand, auf GPU-PC anschaubar) → Zwischenblick → Welle B (B1–B4). Nach Welle B: Neubewertung, ob die Welt noch leer wirkt; erst dann ggf. Sim-Vertiefung (Flächenbrand, räumliches Wetter, world_objects-Verdrahtung) als eigenes, mit den laufenden Experimenten koordiniertes Vorhaben.
