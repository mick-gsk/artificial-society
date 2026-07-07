"""Team C — measurement/harness tests for scripts/m1_pilot.py + m1_report.py.

Covers the population-collapse verification instrumentation:
  * path-independent mortality counting (the v1 `remove_dead` blind spot fix),
  * the new per-snapshot Allee/demography fields,
  * the `--respawn-mode` crutch wiring,
  * m1_report surfacing the new fields (incl. graceful skip on old JSONL).

Pure harness scope: no edits to agent.py / resources.py / simulation.py.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import m1_pilot  # noqa: E402
import m1_report  # noqa: E402
import pytest  # noqa: E402

from artificial_society.simulation import Simulation  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_state():
    """run_one patches module globals (simulation.MIN_POPULATION/RESPAWN_COUNT/
    RESPAWN_MODE/CHECKPOINT_INTERVAL) and wraps Agent._nearest_compatible_mate on
    the class. Save/restore around every test so this module can't contaminate
    the determinism/golden tests that share the pytest process."""
    import artificial_society.simulation as sim_mod
    from artificial_society.agents.agent import Agent

    names = ["MIN_POPULATION", "RESPAWN_COUNT", "RESPAWN_MODE", "CHECKPOINT_INTERVAL"]
    saved = {n: getattr(sim_mod, n, None) for n in names}
    had = {n: hasattr(sim_mod, n) for n in names}
    saved_partner = Agent._nearest_compatible_partner
    saved_partner_orig = m1_pilot._PARTNER_SEEK_ORIG
    try:
        yield
    finally:
        for n in names:
            if had[n]:
                setattr(sim_mod, n, saved[n])
            elif hasattr(sim_mod, n):
                delattr(sim_mod, n)
        Agent._nearest_compatible_partner = saved_partner
        m1_pilot._PARTNER_SEEK_ORIG = saved_partner_orig


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _build_sim(physics_v2: bool):
    return Simulation(
        headless=True,
        load_checkpoint=False,
        physics_v2=physics_v2,
        grid_w=12,
        grid_h=10,
        initial_population=8,
        seed=3,
    )


# --------------------------------------------------------------------------
# 1. Path-independent mortality — the core deliverable.
# --------------------------------------------------------------------------


def test_mortality_pathindependent_counts_v1_death():
    """v1 kills an agent -> path-independent `deaths` counts it, even though the
    legacy `remove_dead` wrapper is blind (pre-filter simulation.py:626)."""
    sim = _build_sim(physics_v2=False)
    mortality, removedead = m1_pilot._instrument_mortality(sim)
    sim.agents[0].alive = False  # deterministic forced death
    sim.step()
    assert mortality["count"] >= 1, "path-independent counter must see the v1 death"
    # The whole point: the old remove_dead-based counter is blind in v1.
    assert removedead["count"] == 0, "v1 pre-filter starves the remove_dead wrapper"


def test_mortality_agrees_in_v2():
    """v2 has no pre-filter, so both counters see the same death (agreement)."""
    sim = _build_sim(physics_v2=True)
    mortality, removedead = m1_pilot._instrument_mortality(sim)
    sim.agents[0].alive = False
    sim.step()
    assert mortality["count"] >= 1
    assert mortality["count"] == removedead["count"], "v2: both paths must agree"


def test_mortality_no_false_positive_on_births_or_respawns():
    """New agents (births/respawns append fresh ids) must NOT register as deaths:
    set-difference of rosters is immune to additions."""
    sim = _build_sim(physics_v2=True)
    mortality, _ = m1_pilot._instrument_mortality(sim)
    sim.step()  # nobody forced to die on the first tick of a fresh sim
    assert mortality["count"] == 0


# --------------------------------------------------------------------------
# 2. New per-snapshot Allee/demography fields.
# --------------------------------------------------------------------------

_NEW_SNAP_KEYS = [
    "deaths",
    "deaths_removedead",
    "mate_seek_fired",
    "sex_m",
    "sex_f",
    "fertile_m",
    "fertile_f",
    "sex_ratio_fertile",
    "pairwise_cheb_mean",
    "pairwise_cheb_median",
    "nearest_partner_min",
    "nearest_partner_mean",
]


def _run_tiny(tmp, **kw):
    return m1_pilot.run_one("learn", 1, 6, 12, 10, 8, 3, tmp, **kw)


def test_snapshot_records_have_new_fields():
    with tempfile.TemporaryDirectory() as tmp:
        path = _run_tiny(tmp, physics="v1")
        recs = _read(path)
    snaps = [r for r in recs if r.get("record") in ("snap", "final")]
    assert snaps, "expected snap/final records"
    for r in snaps:
        for k in _NEW_SNAP_KEYS:
            assert k in r, f"missing new field {k!r} in {r['record']} record"


def test_pairwise_and_sex_counts_are_consistent():
    """pop == sex_m + sex_f (all agents have a sex); pairwise distance defined
    for pop >= 2."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _run_tiny(tmp, physics="v2")
        final = _read(path)[-1]
    assert final["sex_m"] + final["sex_f"] == final["pop"]
    if final["pop"] >= 2:
        assert final["pairwise_cheb_mean"] is not None
        assert final["pairwise_cheb_median"] is not None


def test_demography_fields_direct_allee_signal():
    """`_demography_fields` returns a null nearest-partner distance when one
    fertile sex is absent (the Allee corner), and a real distance otherwise."""
    sim = _build_sim(physics_v2=True)
    fields = m1_pilot._demography_fields(sim)
    # Fresh founders are age 0 (< MIN_REPRODUCTION_AGE) -> no fertile agents yet.
    assert fields["fertile_m"] == 0 and fields["fertile_f"] == 0
    assert fields["nearest_partner_min"] is None
    assert fields["sex_ratio_fertile"] is None
    # Pairwise distance is still defined (>= 2 living agents).
    assert fields["pairwise_cheb_mean"] is not None


# --------------------------------------------------------------------------
# 3. --respawn-mode crutch wiring.
# --------------------------------------------------------------------------


def test_respawn_mode_off_disables_crutch_and_is_recorded():
    import artificial_society.simulation as sim_mod

    with tempfile.TemporaryDirectory() as tmp:
        path = _run_tiny(tmp, physics="v1", respawn_mode="off")
        recs = _read(path)
    meta = recs[0]
    final = recs[-1]
    assert meta["respawn_mode"] == "off"
    assert final["respawns"] == 0, "off-mode must never fire emergency_respawn"
    assert sim_mod.MIN_POPULATION == 0, "off-mode patches MIN_POPULATION to 0"


def test_respawn_mode_records_hook_global():
    """scatter/inherit are recorded in meta AND set the Team-B hook global
    simulation.RESPAWN_MODE (one-line integration point)."""
    import artificial_society.simulation as sim_mod

    for mode in ("scatter", "inherit"):
        with tempfile.TemporaryDirectory() as tmp:
            path = _run_tiny(tmp, physics="v1", respawn_mode=mode)
            meta = _read(path)[0]
        assert meta["respawn_mode"] == mode
        assert mode == sim_mod.RESPAWN_MODE


# --------------------------------------------------------------------------
# 4. mate_seek_fired counter mechanism (deterministic, no sim state needed).
# --------------------------------------------------------------------------


def test_partner_seek_counter_counts_pursued_targets():
    from artificial_society.agents.agent import Agent

    saved_orig = m1_pilot._PARTNER_SEEK_ORIG
    saved_method = Agent._nearest_compatible_partner
    try:
        # Frozen stub original: returns a target (pursued) then None (no target).
        m1_pilot._PARTNER_SEEK_ORIG = lambda self, agents: agents  # truthy -> counted
        m1_pilot._PARTNER_SEEK_STATE["count"] = 0
        m1_pilot._install_partner_seek_counter()
        dummy = object.__new__(Agent)
        Agent._nearest_compatible_partner(dummy, (1, 1))  # non-None -> +1
        assert m1_pilot._PARTNER_SEEK_STATE["count"] == 1
        m1_pilot._PARTNER_SEEK_ORIG = lambda self, agents: None
        m1_pilot._install_partner_seek_counter()
        Agent._nearest_compatible_partner(dummy, None)  # None -> no increment
        assert m1_pilot._PARTNER_SEEK_STATE["count"] == 1
    finally:
        m1_pilot._PARTNER_SEEK_ORIG = saved_orig
        Agent._nearest_compatible_partner = saved_method


# --------------------------------------------------------------------------
# 5. m1_report surfaces the Allee section (and skips old JSONL gracefully).
# --------------------------------------------------------------------------


def _final_with_fields(**over):
    base = {
        "record": "final",
        "pop": 20,
        "fertile_m": 5,
        "fertile_f": 6,
        "sex_ratio_fertile": 0.83,
        "pairwise_cheb_mean": 12.0,
        "nearest_partner_mean": 9.0,
        "mate_seek_fired": 40,
        "deaths": 30,
        "deaths_removedead": 0,  # v1-blind
    }
    base.update(over)
    return base


def test_report_allee_check_prints_new_fields():
    arms = {"learn": [_final_with_fields()]}
    buf = io.StringIO()
    with redirect_stdout(buf):
        m1_report._print_allee_check(arms)
    out = buf.getvalue()
    assert "Allee-/Demografie-Diagnose" in out
    assert "near_partner" in out
    # v1 blindness hint fires when deaths>0 but deaths_removedead==0.
    assert "remove_dead-Zählung blind" in out


def test_report_allee_check_graceful_on_old_jsonl():
    """Finals without any Team-C field must not crash the report."""
    arms = {"learn": [{"record": "final", "pop": 10, "respawns": 4}]}
    buf = io.StringIO()
    with redirect_stdout(buf):
        m1_report._print_allee_check(arms)
    out = buf.getvalue()
    assert "übersprungen" in out


def test_report_full_run_on_produced_jsonl():
    """End-to-end: a real m1_pilot run feeds m1_report.main without crashing."""
    with tempfile.TemporaryDirectory() as tmp:
        _run_tiny(tmp, physics="v1")
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["m1_report.py", "--dir", tmp]
        try:
            with redirect_stdout(buf):
                m1_report.main()
        finally:
            sys.argv = argv
    assert "Allee-/Demografie-Diagnose" in buf.getvalue()
