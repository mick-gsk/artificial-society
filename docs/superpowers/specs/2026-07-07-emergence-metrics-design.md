# Emergence-Metrics System — Design

**Date:** 2026-07-07 · **Lane:** systems (no hot files) · **Status:** draft → tailored review → implement

## 1. Problem & Goal

The project goal (Roadmap §4b) is *learned* tool use, building, and proto-language — but no run can currently answer whether any of that is emerging: there is no instrumentation. This system is a **passive, headless-friendly observer** that turns raw sim state into emergence indicators every sample interval. It must work identically for v1, v2, and rssm arms so any experiment (RSSM confirmatory, M1 learning-ON/OFF, culture coupling) can use the same instrument.

## 2. Non-Goals

- No behavior change of any kind: read-only, zero RNG draws, no sim-state mutation (golden-safe — the registry auto-constructs every system on every sim, including v1 golden runs).
- No serve/dashboard wiring in v1 of this system (the tracker exposes the same shapes the stats system uses; the infra lane can add HISTORY_KEYS later).
- No provenance tagging inside `CausalMemory` (that would touch shared culture code); transmitted-vs-independent is *approximated* by spread + `CultureTracker.discovery_tick` timing, stated as a proxy.

## 3. Architecture

One new module: `artificial_society/systems/emergence_metrics.py`, mirroring `systems/rssm_learning.py`'s shape: class `EmergenceMetrics` + module-level `_tick(sim, tick)` + `@register(name="emergence_metrics", order=68, tick=_tick)`.

- **Order 68**: after technology (50, fresh `capability_map`) and rssm training (65), before stats (70) — the "stats ticks last" contract test stays untouched.
- **Sampling**: `SAMPLE_EVERY = 25` ticks (module constant); `_tick` returns immediately otherwise. All work is pure reads + appends to the instance's own lists.
- **State**: `series: dict[str, list[tuple[int, float]]]` (capped last 400 samples per key), `last: dict[str, float]`, `events: list[dict]` (capped 500, e.g. "new structure", "token converged"), plus internal previous-snapshot fields for deltas.
- **API**: `sim.emergence_metrics.last / .series / .summary()`; `dump_jsonl(path)` writes one line per sample (for headless A/B runners).

## 4. Metrics (per sample)

**Tools (guarded `sim.physics_v2` or rssm-with-v2-objects; else keys absent):**
- `tool_holders` — alive agents with a held object (`a.hands.held`) whose `effective_sharpness(obj) > 0.3` or heat-emission prop > 0.2.
- `verb_fired_{verb}` / `verb_success_rate` — per-interval deltas of the **nested** counters `sim.world.objects.metrics["verbs_fired"][verb]` and `["verbs_failed"][verb]` (built lazily in `agents/agent.py:1008-1012`; success = fired − failed; guard missing keys → 0). Flat per-verb keys do NOT exist; the flat metrics (`fragments_total`, `cuts_with_tool`, …) may be reported as-is where useful.
- `inventions` — delta of `len(layer.discovery.entries)`.

**Building:**
- `structures_total` / `structures_new` — sum over world cells of `cell["structures"]` flags (camp/farm/well); new = delta. (v2 composite objects: out of scope, noted.)

**Proto-language (read-only re-implementation — do NOT call `TokenWorld.check_convergence`, it emits events):**
- `lang_tokens_active` — tokens with `use_count ≥ 3` in ≥ 2 agents' `token_memory.associations`.
- `lang_max_share` — max over tokens of (users with `use_count ≥ 3`) / alive agents.
- `lang_converged` — count of tokens with share ≥ 0.5 AND mean pairwise context-cosine > 0.7 (mirrors `check_convergence`'s thresholds without emitting).

**Culture / technology:**
- `culture_distinct_seqs` — |union of `causal_memory.sequences` keys across alive agents|.
- `culture_max_spread` — max # agents sharing one sequence key; `culture_shared_seqs` — # keys known by ≥ 3 agents.
- `novel_causal_keys` — per-interval delta of the union size (invention/learning rate).
- `tech_capabilities` — `len(sim.technology.capability_map)`. (Technology is registered ACTIVE (`_builtins.py:79`, tick order 50) and the map is rebuilt every tick — the `getattr` guard is defensive only and in practice never triggers.)

**Behavior:**
- `behavior_entropy` — **discrete verb-frequency entropy** over the per-interval `verbs_fired` deltas (Shannon entropy of the normalized verb histogram; defined for every arm that fires verbs, omitted when no verbs fired in the interval). NOT derived from `_rssm_last_action` (a continuous tensor — entropy undefined without arbitrary binning) and **no hot-file writes to create new attributes**.

## 5. Determinism & safety rules

- Construction and tick draw **zero** RNG; no `random`/`np.random`/`torch` calls at all.
- All reads defensive (`getattr(..., None)` guards): the system must not crash on v1 sims lacking v2 fields, dormant technology, or agents without `token_memory`/`causal_memory`.
- Full pytest incl. golden must stay green with the module present (it auto-registers via `discover()`).

## 6. Tests (TDD)

1. Registration: sim constructs → `sim.emergence_metrics` exists; order 68 < stats; golden + `torch.get_rng_state()` unchanged by a sampled tick.
2. Synthetic counting: inject fake `causal_memory.sequences` across agents → `culture_max_spread`/`culture_shared_seqs` exact; fake `token_memory` associations → `lang_*` exact (incl. the no-emit guarantee: `TOKEN_WORLD.world_log` length unchanged).
3. v2 path: tiny physics_v2 sim, grasp a sharp object onto an agent → `tool_holders == 1`; verb-counter delta logic with a hand-built metrics dict.
4. Sampling interval + caps: series only grows at multiples of `SAMPLE_EVERY`; caps enforced.
5. `dump_jsonl` round-trip.

## 7. Open questions (for the reviewer)

- Thresholds: share 0.5 / cosine 0.7 mirror `check_convergence` literals; sharpness 0.3 mirrors tool-quality scale; **spread ≥3 is a novel pin with no existing anchor** (documented as such). All are module constants.
- Is per-25-tick full-cell scanning acceptable at 200×200 (worst case ~40k cells / sample)? If not: incremental structure counting or sample every 100 ticks at large grids.
- Anything emergence-relevant and cheaply observable that's missing?
