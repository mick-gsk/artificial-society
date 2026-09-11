"""Demografie-Stabilisierung (Design 2026-07-06): C1 Gründer-Cooldown-Staffelung
(Harness) + C2 Fruchtbarkeits-Gate auf erneuerbare Versorgung-pro-Mund
(``plant_renewal_ema``, nur physics_v2).

Bindet die drei Review-Auflagen fest, die diese Datei betreffen:
  * C1 zieht den initialen ``reproduction_cooldown`` aus DEMSELBEN globalen
    ``random``-Strom wie der Schwester-``age``-Draw, nur in ``_age_structure_founders``.
  * C2 ist STRIKT ``physics_v2``-gegatet: v1 behält das Bestand-Gate
    (``_local_food_per_capita < REPRODUCTION_MIN_FOOD_PER_CAPITA``) byte-für-byte
    und liest ``plant_renewal_ema`` NIE (Golden-Schutz).
  * Der C2-Floor darf nicht in Erholungs-Unterdrückung kalibriert werden:
    Niedrigdichte/erholter Fluss ⇒ Gate OFFEN.
"""

from __future__ import annotations

import os
import random as _random
import sys

import pytest

from artificial_society.simulation import Simulation

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import m1_pilot  # noqa: E402

import artificial_society.agents.agent as agent_mod  # noqa: E402
from artificial_society.agents.agent import (  # noqa: E402
    REPRODUCTION_COOLDOWN,
    REPRODUCTION_ENERGY,
)

_V2 = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _build(seed=42, physics_v2=True, pop=8, **kw):
    params = {**_V2, "initial_population": pop, "seed": seed, "physics_v2": physics_v2, **kw}
    return Simulation(**params)


# ---------------------------------------------------------------------------
# C1 — Gründer-Cooldown-Staffelung (Harness _age_structure_founders)
# ---------------------------------------------------------------------------


def test_c1_cooldowns_streuen_ueber_bereich():
    """Mit --age-structured-founders sind die initialen Cooldowns über
    [0, REPRODUCTION_COOLDOWN) gestreut (nicht alle identisch/0), damit der
    synchrone t0-Konzeptions-Puls zerlegt wird."""
    sim = _build(seed=7, pop=30)
    m1_pilot._age_structure_founders(sim)
    cds = [a.reproduction_cooldown for a in sim.agents]
    assert all(0 <= c < REPRODUCTION_COOLDOWN for c in cds), cds
    assert max(cds) > 0, "Cooldowns müssen gestaffelt sein, nicht alle 0"
    assert len(set(cds)) > 1, "Cooldowns müssen streuen, nicht alle identisch"


def test_c1_determinismus_gleicher_seed_gleiche_folge():
    """Gleicher Seed ⇒ identische Cooldown-Folge (globaler seed_all-Strom)."""

    def _cds():
        sim = _build(seed=99, pop=24)
        m1_pilot._age_structure_founders(sim)
        return [a.reproduction_cooldown for a in sim.agents]

    assert _cds() == _cds()


def test_c1_ohne_flag_alle_cooldowns_null():
    """Ohne den Flag (kein _age_structure_founders-Aufruf) starten alle Gründer
    mit reproduction_cooldown == 0 — Default/v1-Pfad unverändert."""
    sim = _build(seed=7, pop=30)
    assert all(a.reproduction_cooldown == 0 for a in sim.agents)


def test_c1_zieht_aus_globalem_strom_verschiebt_ihn():
    """C1 zieht aus dem globalen random-Strom (kein separates RNG-Objekt): ein
    reiner age-only-Draw (nur `age`, kein Cooldown) über denselben Strom lässt
    danach eine ANDERE Folge übrig als der age+cooldown-Draw — Beleg, dass der
    Cooldown-Draw denselben Strom nutzt und ihn (erwartet) verschiebt."""
    sim_a = _build(seed=555, pop=20)
    _random.seed(1234)
    m1_pilot._age_structure_founders(sim_a)
    after_full = _random.random()

    _random.seed(1234)
    for a in sim_a.agents:
        _random.randint(0, 1999)  # nur der age-Draw, KEIN Cooldown-Draw
    after_age_only = _random.random()

    assert after_full != after_age_only


# ---------------------------------------------------------------------------
# C2 — Fruchtbarkeits-Gate auf plant_renewal_ema (nur physics_v2)
# ---------------------------------------------------------------------------


def _make_fertile_pair(sim):
    """Zwei ko-lokierte, fortpflanzungsfähige Agenten (F + M); alle anderen
    weit weg, damit sie den Mund-/Mate-Scan nicht stören."""
    f = sim.agents[0]
    m = sim.agents[1]
    for a in (f, m):
        a.alive = True
        a.age = 1000
        a.energy = REPRODUCTION_ENERGY + 140.0
        a.reproduction_cooldown = 0
        a.pregnant = False
        a._cached_nearby_agents = None
    f.sex = "f"
    m.sex = "m"
    f.pos = (10, 7)
    m.pos = (10, 7)
    for other in sim.agents[2:]:
        other.pos = (0, 0)
        other._cached_nearby_agents = None
    return f, m


def test_c2_hohe_dichte_gate_schliesst():
    """v2: erneuerbarer Fluss ≈0 ⇒ renewal/Münder < Floor ⇒ Gate schließt,
    keine Konzeption trotz gültigem Paar."""
    sim = _build(physics_v2=True)
    f, m = _make_fertile_pair(sim)
    sim.world.F["plant_renewal_ema"][:] = 0.0
    assert (
        f._local_renewal_per_capita(sim.world, sim.agents)
        < agent_mod.REPRODUCTION_MIN_RENEWAL_PER_CAPITA
    )
    assert f._try_reproduce(sim.world, sim.agents) is None
    assert f.pregnant is False


def test_c2_niedrige_dichte_gate_offen():
    """BINDEND (Review M-1/M-5): bei erholtem Fluss + wenigen Mündern öffnet das
    Gate mit dem DEFAULT-Floor ⇒ Konzeption möglich. Sperrt den Floor gegen
    Erholungs-Unterdrückung."""
    sim = _build(physics_v2=True)
    f, m = _make_fertile_pair(sim)
    sim.world.F["plant_renewal_ema"][:] = 1.0  # erholter Fluss, klar > jeder sane Floor
    assert (
        f._local_renewal_per_capita(sim.world, sim.agents)
        >= agent_mod.REPRODUCTION_MIN_RENEWAL_PER_CAPITA
    )
    f._try_reproduce(sim.world, sim.agents)
    assert f.pregnant is True, "erholtes Regime muss konzipieren können"


def test_c2_t0_signal_nicht_annaehernd_null():
    """t0-Robustheit: bei voller Welt ist der post-headroom-Fluss ≈0, aber der
    EMA wird mit dem PRE-headroom-Potenzial geseedet ⇒ Signal NICHT ≈0."""
    sim = _build(physics_v2=True, grid_w=24, grid_h=18)
    sim.step()  # tick==0 regrow seedet plant_renewal_ema mit pre-headroom plant_gain
    ema = sim.world.F["plant_renewal_ema"]
    assert ema.max() > 0.01, "t0-Seed muss den EMA klar über 0 heben"


def test_c2_v1_neutralitaet_liest_ema_nie():
    """KRITISCH (Review C-1/M-3): mit physics_v2=False läuft weiter das
    Bestand-Gate; plant_renewal_ema wird NIE gelesen. Spy: _local_renewal_per_capita
    wirft — der v1-Pfad darf es nicht aufrufen."""
    sim = _build(physics_v2=False)
    f, m = _make_fertile_pair(sim)
    assert not f.physics_v2

    from artificial_society.agents.agent import Agent

    orig = Agent._local_renewal_per_capita

    def _boom(self, world, agents):
        raise AssertionError("v1-Pfad darf plant_renewal_ema/renewal-Gate NIE lesen")

    Agent._local_renewal_per_capita = _boom
    try:
        # Bestand-Gate offen (viel food) ⇒ v1 konzipiert, ohne den Renewal-Spy zu berühren.
        sim.world.F["food"][:] = 200.0
        f._try_reproduce(sim.world, sim.agents)
        assert f.pregnant is True
    finally:
        Agent._local_renewal_per_capita = orig


def test_c2_v1_gate_bleibt_bestand_gate():
    """v1: leerer Bestand (food=0) schließt das Gate, selbst wenn der EMA hoch
    stünde — Beleg, dass v1 den Bestand liest, nicht den EMA."""
    sim = _build(physics_v2=False)
    f, m = _make_fertile_pair(sim)
    sim.world.F["food"][:] = 0.0
    sim.world.F["plant_renewal_ema"][:] = 999.0  # würde das Renewal-Gate weit öffnen
    assert f._try_reproduce(sim.world, sim.agents) is None
    assert f.pregnant is False


def test_c2_determinismus_gleicher_seed():
    """Gleicher Seed ⇒ identische renewal-per-capita-Trajektorie über mehrere Ticks."""

    def _run():
        sim = _build(seed=321, physics_v2=True, grid_w=24, grid_h=18)
        vals = []
        for _ in range(8):
            sim.step()
            vals.append(round(float(sim.world.F["plant_renewal_ema"].sum()), 6))
        return vals

    assert _run() == _run()
