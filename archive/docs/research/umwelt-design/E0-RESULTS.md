# E0-RESULTS.md — Embodiment-gematchte Baseline (Befund, 2026-06-30)

> **Status:** lokaler CPU-Vorlauf (MacBook), seeds 1001–1004, N_attempts=12000. **Kein** GPU-Locked-Regime-Lauf; Vergleich gegen die *dokumentierten* Learned-Mittel (nicht ein ko-gelaufener Learned-Arm). Starker Vorbefund, GPU-Konfirmation (ko-gelaufener Learned + embodied-Null + M5-Metrik am Locked-Regime) ausstehend. Code: `artificial_society/research/recombiner.py::run_embodied_recombiner` (getestet, `tests/test_research_metrics.py`).

## Frage (E0 / Hebel 2, „notwendig-nicht-hinreichend")

Ist das berühmte **−9.4 / −8.5 „Lern-Defizit"** ein echter Treue-/Lern-Mangel — oder ein **Embodiment-/Populations-Fragmentierungs-Confound** des Vergleichs *embodied Population* vs. *disembodied Perfekt-Gedächtnis-Monopolist*?

## Methode

`run_embodied_recombiner` = identische blinde Uniform-Kombinationsmaschinerie wie der disembodied Recombiner, aber das `n_attempts`-Budget wird round-robin auf `n_agents` Agenten mit **je eigenem lokalen Pool** verteilt (Fragmentierung); eine Entdeckung wird nur mit Wahrscheinlichkeit `share_fidelity` in den Pool eines zufälligen anderen Agenten geteilt (imperfekte Transmission). **Ein** global-deduplizierter Registry (die Sicht der Metrik, wie `DISCOVERY_REGISTRY`); `discovered_by = Agent-Index`. `n_agents=24, share_fidelity=0.72` spiegelt Pop=24 + sozialen Kanal `FIDELITY_BASE=0.72`. *Dokumentierte Vereinfachung:* keine räumliche/Need-Dynamik; Re-Derivation eines bereits global bekannten Materials wächst den Pool nicht.

## Ergebnis (Mittel über 4 seeds, N=12000)

| Konfiguration | #disc | maxFD | meanFD | DV2 | graded | gradedDW |
|---|---:|---:|---:|---:|---:|---:|
| disembodied P=1 (der Pilot-Null) | 9700 | 30.2 | 14.43 | 24.8 | 28.0 | 191 |
| embodied P=8  f=0.72 | 10356 | 23.0 | 9.90 | 13.5 | 27.4 | 66 |
| **embodied P=24 f=0.72** (Learned-gematcht) | 10198 | **20.0** | **7.51** | 10.8 | 31.4 | 71 |
| embodied P=24 f=0.30 | 10016 | 18.0 | 6.70 | 8.8 | 31.6 | 66 |
| embodied P=24 f=0.00 (volle Fragmentierung) | 9725 | 17.5 | 6.17 | 9.5 | 32.7 | 79 |
| **LEARNED (dokumentiert, Pilot/Schritt-A)** | — | **~22.9** | **~7.4** | **~2** | — | — |

## Interpretation

1. **Der Defizit ist überwiegend ein Embodiment-Confound (BELEGT, lokal).** Embodiment-Matching (P=24, f=0.72) senkt `mean_functional_depth` **14.43 → 7.51** — praktisch **auf** das dokumentierte Learned-Mittel (7.4). `max_functional_depth` fällt **30.2 → 20.0**; der Learned-Arm (22.9) liegt damit sogar **leicht darüber**. Auf den Tiefen-Metriken ist ein **fairer** (embodiment-gematchter) blinder Null **≈ Learned-Arm** — der −9.4 wurde gegen einen **unfairen** (disembodied-Monopolist) Null gemessen.
2. **Residuum auf DV2 (offene Frage).** DV2 fällt durch Fragmentierung 24.8 → ~10.8, **nicht** auf den Learned-Floor (~2). Entweder echtes Rest-Defizit *oder* die DV2-Floor-Sättigung des embodied Arms (M5-Caveat) — **nicht trennbar ohne M5-Metrik auf einem ko-gelaufenen Learned-Arm.**
3. **Metrik-Befund (M5(a)).** Die **un-gewichtete** `graded`-Metrik ist **flach ~28–33** über alle Fragmentierungsgrade (sie hebt vom Floor ab, ist aber **tiefenblind** — die Magnitude sättigt nahe der Task-Decke, weil alle Konfigs ~10k Entdeckungen haben). Die **tiefen-gewichtete** `gradedDW` diskriminiert dagegen scharf (191 → 66–79) und ist die taugliche Gate-DV; die un-gewichtete dient nur als Sättigungs-Diagnostik.

## Folgen für das Ziel

- Die richtige, **null-kalibrierte, compute-*und*-embodiment-gematchte** Baseline ist `run_embodied_recombiner(P≈Pop, f≈FIDELITY_BASE)`, **nicht** der disembodied Recombiner.
- Gegen diese faire Baseline ist der Learned-Arm auf den Tiefen-Metriken **bei Parität bis leicht darüber** — die Dramatik „random schlägt learned" ist großteils Confound. Ob der Learned-Arm *strikt über* dem fairen Null liegt (das eigentliche Ziel), ist **noch nicht entschieden** und braucht den GPU-Konfirmationslauf (ko-gelaufener Learned + embodied-Null + `gradedUW/DW` + DV2 am Locked-Regime).
- **Kill-Bezug:** E0 widerlegt nicht das Ziel, aber es **verschiebt die Messlatte** von „32 schlagen" auf „~7.5 meanFD / ~20 maxFD / fairen DV2 schlagen" — und macht M1 (selektive Generierung) erst gegen *diese* Latte sinnvoll bewertbar.

## Nächster Schritt (GPU-gated)

`gradedUW/DW` + `run_embodied_recombiner` in die Gate-Pipeline (`analyze_gate`) verdrahten → ein Locked-Regime-Pilot (4–12 seeds × 1500 ticks) ko-läuft Learned + embodied-Null und wertet alle DVs gepaart aus. Erst danach ist „Learned > fairer Null?" falsifizierbar beantwortbar.
