# Design-Note: GPU-residente Big-World-Engine (Perf, Tier 5)

**Datum:** 2026-07-02 · **Status:** Entscheidung freigegeben (User), Umsetzung **nach Schnitt 1**
(Physik v2). Voller Spec-/Plan-Zyklus folgt dann; diese Note fixiert Ziel, Architekturentscheidung
und Messkontrakt, damit nichts verloren geht.

## Problem

Auch die CPU ist für das Zielbild viel zu langsam. Der gemessene Engpass ist das pure-Python
Welt-Update (55–84 % der Laufzeit, ~33 M `clamp`-Aufrufe / 100 Ticks; `docs/performance-notes.md`).
Die Tier-1/2/4-Optimierungen bringen ~6–10× — für den vom User definierten Zielmaßstab reicht das
nicht.

## Zielmaßstab (User-Entscheidung 2026-07-02)

**Hunderte Agenten (200–1000), Grid 200×200+ (40k+ Zellen), Läufe über 100k+ Ticks** (viele
Generationen). Zielgröße: **Minuten pro Lauf statt Stunden/Tage** — wenige ms/Tick bei voller Welt.

## Baseline (gemessen 2026-07-02, GPU-PC 9800X3D, CPU-Pfad, main @ 7b69f8e)

| pop | grid | Zellen | ms/Tick | Ticks/s | 100k Ticks |
|----:|-----:|-------:|--------:|--------:|-----------:|
| 36  | 60×40   | 2.400  | 137,5   | 7,27 | 3,8 h  |
| 8   | 200×200 | 40.000 | 1.421,6 | 0,70 | 39,5 h |
| 200 | 200×200 | 40.000 | 1.725,3 | 0,58 | 47,9 h |
| 500 | 200×200 | 40.000 | 2.104,8 | 0,48 | **58,5 h** |

Zerlegung: Welt-Update ≈ **1,41 s/Tick** am Zielgrid (populationsunabhängig, skaliert linear mit
Zellzahl — 16,7× Zellen ≈ 17,4× Kosten), Brains ≈ 1,4 ms/Agent. Ein einziger Meilenstein-Lauf im
Zielmaßstab kostet heute **2,5 Tage**; ein A/B-Pilot mit 10 Seeds × 2 Armen wäre unbenutzbar.
Reproduzieren: `scripts/perf_bench.py scale` (CPU-Pfad, `PYTHONHASHSEED=0`).

Damit ist der Messkontrakt verankert: **dieselben Konfigurationen, dasselbe Skript**
(`scripts/perf_bench.py` + Scale-Läufe) werden nach der Engine-Umsetzung erneut gemessen;
das Vorher/Nachher gehört zum Abnahmekriterium.

**Zwischenstand nach Tier 1+2 (2026-07-02, vorgezogen):** Welt-Update vektorisiert
(bit-identisch, Golden grün) + CPU-Default. GPU-PC-Messung: 36 @ 60×40 = 56,8 ms;
8 @ 200×200 = 50,0 ms (28×); 200 @ 200×200 = 352,3 ms; 500 @ 200×200 = **740,0 ms/Tick
(20,6 h/100k)**. Der Weltterm ist eliminiert; was bleibt, ist das Pro-Agent-Planen
(`imagine_rollout`, ~1,4 ms/Agent) — genau der Teil, den diese Engine batcht. Das
≥50×-Abnahmeziel bezieht sich weiter auf die ursprüngliche Baseline oben.

## Entscheidung: Torch-residente Engine (Hybrid)

1. **Welt als Tensoren:** alle Zellfelder als `torch`-Arrays (GPU-resident); Regrowth =
   elementweise Ops, Diffusion = Faltung. Ersetzt die per-Zelle-Python-Schleifen vollständig.
2. **Agenten als Struct-of-Arrays:** Position/Energie/Hormone/Gene als gestapelte Tensoren
   statt N Python-Objekte im heißen Pfad.
3. **Gehirne gebatcht:** ein Pass über alle Agenten pro Tick (`torch.func.functional_call` +
   `torch.vmap` über gestapelte Per-Agent-Gewichte). Die gemessene ~21×-Batching-Effizienz trägt
   auf der GPU erst bei großen Batches — genau der Zielmaßstab.
4. **Hybrid-Schnitt:** seltene, ereignisgetriebene Sozial-Systeme (Stämme, Handel, soziales
   Lernen — gemessen ~8 %) bleiben zunächst Python; nur der heiße Pfad (Welt, Wahrnehmung,
   Physik-Prozesse, Brains) wird tensorisiert.
5. **Warum Torch, nicht JAX/Numba:** bereits Dependency, Brains sind Torch, läuft nativ auf dem
   Windows-GPU-PC (sm_120/cu128 provisioniert), identischer Code fällt auf dem MacBook auf CPU
   zurück (Dev lokal, große Läufe remote — `compute-on-gpu-pc`-Regel).

## Konsequenzen / Leitplanken

- **Determinismus neu verhandelt:** GPU-Parallelität ist nicht byte-identisch zur v1-Welt. Die
  Engine erhält eine **eigene Golden-Baseline** (engine-intern deterministisch: counter-based
  RNG, deterministische Kernel wo nötig); die v1-Golden bleibt für die Referenzwelt bestehen.
- **Reihenfolge:** bewusst **nach Schnitt 1** — die Engine portiert die *v2*-Physik
  (Eigenschaftsvektoren/Prozesse aus `environment/physics/`), nicht die wegfallende v1-Physik.
  Der Meilenstein-1-Pilot (kleiner Maßstab) läuft notfalls noch auf der langsamen Welt.
- **Nie-gescriptet-Prinzip unberührt:** die Engine ändert *wie schnell* die Welt rechnet,
  niemals *was* Agenten können (roadmap §4b).

## Abnahmekriterien (bei Umsetzung)

1. Vorher/Nachher-Benchmark auf dem GPU-PC über die Baseline-Konfigurationen; Ziel:
   **≥50× bei pop=500/200×200**, Richtwert wenige ms/Tick.
2. Engine-eigene Golden-Trajectory grün über 2 Prozessläufe (Determinismus im neuen Regime).
3. Verhaltens-Äquivalenzklasse statt Byte-Gleichheit: Ökologie-Health-Metriken (Population,
   Nahrung/Kapazität, Geburten) der Engine-Welt statistisch vergleichbar mit der Referenzwelt
   bei gleicher Konfiguration.
