"""Passive, headless-friendly emergence observer (systems lane, registry order=68).

The project goal (Roadmap §4b) is *learned* tool use, building, and proto-language
— but no run could previously answer whether any of that is emerging: there was no
instrumentation. This module turns raw sim state into emergence indicators every
``SAMPLE_EVERY`` ticks, identically for v1, v2, and rssm arms, so any experiment
(RSSM confirmatory, M1 learning-ON/OFF, culture coupling) can use the same
instrument.

Mirrors ``systems/rssm_learning.py``'s shape: class + module-level ``_tick`` +
``@register``. Ordered at 68: after technology (50, fresh ``capability_map``) and
rssm training (65), strictly before stats (70) — ``test_phase2_systems.py``'s
``test_tick_order_stats_last_and_disease_before_economy`` pins stats as the last
system ticked every tick; this must stay true.

Non-goals (see design doc §2): no behaviour change of any kind — read-only, zero
RNG draws, no sim-state mutation. ``systems.registry.discover()`` auto-imports this
module into every sim, including v1 golden runs, so construction and tick must be
side-effect-free by construction, not by convention.
"""

from __future__ import annotations

import json
import math

import numpy as np

from artificial_society.environment.physics.processes import effective_sharpness
from artificial_society.environment.physics.props import IDX2
from artificial_society.systems.registry import register

# Sampling cadence (module constant, not per-instance state — never touched by RNG).
SAMPLE_EVERY = 25

# Caps (design §3): keep memory bounded for long headless A/B runs.
SERIES_CAP = 400
EVENTS_CAP = 500

# Thresholds (module constants; provenance in design doc §7).
TOOL_SHARPNESS_MIN = 0.3
# PROP_DIMS_V2 (environment/physics/props.py) has no "heat_emission" dimension
# today — this guard mirrors the technology getattr guard elsewhere in the
# codebase: defensive only, never triggers in practice, but costs nothing and
# lights up for free if a future fire-tool prop is added under this name.
HEAT_EMISSION_PROP = "heat_emission"
HEAT_EMISSION_MIN = 0.2
LANG_USE_COUNT_MIN = 3
LANG_CONVERGE_SHARE = 0.5
LANG_CONVERGE_COSINE = 0.7
CULTURE_SPREAD_MIN = 3  # novel pin, no existing anchor (design doc §7)


def _shannon_entropy(counts: dict) -> float | None:
    """Entropy of a normalized histogram of positive integer counts, or None if
    every count is zero/absent (entropy undefined for an empty distribution)."""
    total = sum(c for c in counts.values() if c > 0)
    if total <= 0:
        return None
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


class EmergenceMetrics:
    """State holder; all behaviour lives in the registered tick hook below."""

    def __init__(self) -> None:
        self.series: dict[str, list] = {}
        self.last: dict[str, float] = {}
        self.events: list = []
        self.samples: list = []

        # Previous-snapshot fields, for interval deltas.
        self._prev_verbs_fired: dict = {}
        self._prev_verbs_failed: dict = {}
        self._prev_discovery_count: int = 0
        self._prev_structures_total: float = 0.0
        self._prev_culture_keys: set = set()
        self._prev_lang_converged: int = 0

    # -- bookkeeping ----------------------------------------------------------
    def _record(self, tick: int, key: str, value: float) -> None:
        self.last[key] = value
        bucket = self.series.setdefault(key, [])
        bucket.append((tick, value))
        if len(bucket) > SERIES_CAP:
            del bucket[: len(bucket) - SERIES_CAP]

    def _emit_event(self, tick: int, kind: str, **extra) -> None:
        self.events.append({"tick": tick, "type": kind, **extra})
        if len(self.events) > EVENTS_CAP:
            del self.events[: len(self.events) - EVENTS_CAP]

    def summary(self) -> dict:
        """Snapshot of the most recently recorded value per tracked key."""
        return dict(self.last)

    def dump_jsonl(self, path: str) -> None:
        """Write one JSON line per recorded sample (for headless A/B runners)."""
        with open(path, "w") as fh:
            for row in self.samples:
                fh.write(json.dumps(row) + "\n")

    # -- sampling ---------------------------------------------------------
    def sample(self, sim, tick: int) -> None:
        """Take one sample of every metric. Pure reads only — no RNG, no mutation
        of anything outside ``self``."""
        agents = getattr(sim, "agents", None) or []
        alive = [a for a in agents if getattr(a, "alive", False)]
        world = getattr(sim, "world", None)
        objects_layer = getattr(world, "objects", None) if world is not None else None

        # Embodiment (physics v2 hands/body) is opt-in per agent; agents on the
        # v1 arm never get a Hands instance (agents/agent.py:220-225), so this
        # doubles as the "physics_v2 or rssm-with-v2-objects" guard without
        # depending on the exact flag name of whichever arm is running.
        has_embodiment = any(getattr(a, "hands", None) is not None for a in alive)

        fired_now, failed_now = self._verb_counters(objects_layer)

        if has_embodiment and objects_layer is not None:
            self._sample_tools(tick, alive, objects_layer, fired_now, failed_now)

        self._sample_structures(tick, world)
        self._sample_language(tick, alive)
        self._sample_culture(tick, alive)
        self._sample_technology(tick, sim)
        self._sample_behavior(tick, fired_now)

        self._prev_verbs_fired = fired_now
        self._prev_verbs_failed = failed_now

        snapshot = dict(self.last)
        snapshot["tick"] = tick
        self.samples.append(snapshot)
        if len(self.samples) > SERIES_CAP:
            del self.samples[: len(self.samples) - SERIES_CAP]

    # -- per-category helpers --------------------------------------------
    def _verb_counters(self, objects_layer) -> tuple:
        metrics = getattr(objects_layer, "metrics", None) if objects_layer is not None else None
        if metrics is None:
            return {}, {}
        fired = dict(metrics.get("verbs_fired", {}))
        failed = dict(metrics.get("verbs_failed", {}))
        return fired, failed

    def _sample_tools(self, tick, alive, objects_layer, fired_now, failed_now) -> None:
        heat_idx = IDX2.get(HEAT_EMISSION_PROP)

        tool_holders = 0
        for a in alive:
            hands = getattr(a, "hands", None)
            held = getattr(hands, "held", None) if hands is not None else None
            if not held:
                continue
            for obj in held:
                props = getattr(obj, "props", None)
                sharp = effective_sharpness(obj)
                heat_ok = (
                    heat_idx is not None
                    and props is not None
                    and float(props[heat_idx]) > HEAT_EMISSION_MIN
                )
                if sharp > TOOL_SHARPNESS_MIN or heat_ok:
                    tool_holders += 1
                    break
        self._record(tick, "tool_holders", float(tool_holders))

        verb_keys = set(fired_now) | set(self._prev_verbs_fired)
        total_fired_delta = 0
        total_failed_delta = 0
        for verb in verb_keys:
            fired_delta = fired_now.get(verb, 0) - self._prev_verbs_fired.get(verb, 0)
            failed_delta = failed_now.get(verb, 0) - self._prev_verbs_failed.get(verb, 0)
            if fired_delta:
                self._record(tick, f"verb_fired_{verb}", float(fired_delta))
            total_fired_delta += max(0, fired_delta)
            total_failed_delta += max(0, failed_delta)
        if total_fired_delta > 0:
            success = (total_fired_delta - total_failed_delta) / total_fired_delta
            self._record(tick, "verb_success_rate", float(success))

        discovery = getattr(objects_layer, "discovery", None)
        entries = getattr(discovery, "entries", None)
        n_discovery = len(entries) if entries is not None else self._prev_discovery_count
        inventions = n_discovery - self._prev_discovery_count
        self._prev_discovery_count = n_discovery
        self._record(tick, "inventions", float(inventions))

    def _sample_structures(self, tick, world) -> None:
        if world is None:
            return
        cells = getattr(world, "cells", None)
        if cells is None:
            return
        total = 0.0
        for row in cells:
            for cell in row:
                structs = cell["structures"]
                total += sum(1 for v in structs.values() if v)
        new = total - self._prev_structures_total
        self._prev_structures_total = total
        self._record(tick, "structures_total", total)
        self._record(tick, "structures_new", new)
        if new > 0:
            self._emit_event(tick, "new_structure", count=new)

    def _sample_language(self, tick, alive) -> None:
        memories = [getattr(a, "token_memory", None) for a in alive]
        memories = [m for m in memories if m is not None]
        if not memories:
            return  # feature inapplicable to this arm — keys stay absent

        n_alive = len(memories)
        all_token_ids: set = set()
        for m in memories:
            all_token_ids.update(m.associations.keys())

        active_count = 0
        max_share = 0.0
        converged = 0
        for token_id in all_token_ids:
            users = [
                m
                for m in memories
                if token_id in m.associations
                and m.associations[token_id].use_count >= LANG_USE_COUNT_MIN
            ]
            if len(users) >= 2:
                active_count += 1
            share = len(users) / n_alive if n_alive else 0.0
            max_share = max(max_share, share)
            if len(users) >= 2 and share >= LANG_CONVERGE_SHARE:
                mean_ctxs = [
                    m.mean_context(token_id) for m in users if m.mean_context(token_id) is not None
                ]
                if len(mean_ctxs) >= 2:
                    sims = []
                    for i in range(len(mean_ctxs)):
                        for j in range(i + 1, len(mean_ctxs)):
                            va, vb = mean_ctxs[i], mean_ctxs[j]
                            denom = float(np.linalg.norm(va)) * float(np.linalg.norm(vb)) + 1e-8
                            sims.append(float(np.dot(va, vb)) / denom)
                    avg_sim = sum(sims) / len(sims) if sims else 0.0
                    if avg_sim > LANG_CONVERGE_COSINE:
                        converged += 1

        self._record(tick, "lang_tokens_active", float(active_count))
        self._record(tick, "lang_max_share", float(max_share))
        self._record(tick, "lang_converged", float(converged))
        if converged > self._prev_lang_converged:
            self._emit_event(tick, "token_converged", count=converged - self._prev_lang_converged)
        self._prev_lang_converged = converged

    def _sample_culture(self, tick, alive) -> None:
        counts: dict = {}
        seq_keys: set = set()
        any_causal_memory = False
        for a in alive:
            cm = getattr(a, "causal_memory", None)
            if cm is None:
                continue
            any_causal_memory = True
            for key in cm.sequences:
                seq_keys.add(key)
                counts[key] = counts.get(key, 0) + 1
        if not any_causal_memory:
            return  # feature inapplicable — keys stay absent

        distinct = len(seq_keys)
        max_spread = max(counts.values()) if counts else 0
        shared = sum(1 for c in counts.values() if c >= CULTURE_SPREAD_MIN)
        novel = len(seq_keys - self._prev_culture_keys)
        self._prev_culture_keys = seq_keys

        self._record(tick, "culture_distinct_seqs", float(distinct))
        self._record(tick, "culture_max_spread", float(max_spread))
        self._record(tick, "culture_shared_seqs", float(shared))
        self._record(tick, "novel_causal_keys", float(novel))

    def _sample_technology(self, tick, sim) -> None:
        technology = getattr(sim, "technology", None)
        capability_map = getattr(technology, "capability_map", None)
        if capability_map is None:
            return
        self._record(tick, "tech_capabilities", float(len(capability_map)))

    def _sample_behavior(self, tick, fired_now) -> None:
        deltas = {
            verb: fired_now.get(verb, 0) - self._prev_verbs_fired.get(verb, 0)
            for verb in set(fired_now) | set(self._prev_verbs_fired)
        }
        entropy = _shannon_entropy(deltas)
        if entropy is not None:
            self._record(tick, "behavior_entropy", float(entropy))


def _tick(sim, tick: int) -> None:
    if tick % SAMPLE_EVERY != 0:
        return
    sim.emergence_metrics.sample(sim, tick)


@register(name="emergence_metrics", order=68, tick=_tick)
def _build(sim):
    return EmergenceMetrics()
