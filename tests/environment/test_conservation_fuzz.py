"""D1-Massen-Ledger-Fuzzer: 50 Seeds × 200 zufällige Aktionen, Invariante nach JEDER Aktion.

Invariante: boden + hände + ledger[eaten] + ledger[decayed] == ledger[spawned] + ledger[from_carcass].
Sim-frei: ein geskripteter Körper (Body+Hands) treibt die do_*-Funktionen direkt (Spec Plan-Schnitt 3a).
"""

from __future__ import annotations

import math
import random

from artificial_society.environment.phys_objects import ObjectLayer, tick_decay
from artificial_society.environment.physics.actions import (
    do_cut,
    do_eat,
    do_grasp,
    do_release,
    do_strike,
    enforce_carry_budget,
)
from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG, Body, Hands
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.props import IDX2

_GRID = 8
_MATERIALIEN = ("granite", "flint", "dry_wood", "plant_fiber", "clay_moist", "carcass", "raw_meat")
_AKTIONEN = ("spawn", "grasp", "release", "strike", "cut", "eat", "decay", "overload", "die")


def _assert_invariante(layer: ObjectLayer, hands: Hands, seed: int, schritt: int, aktion: str):
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9), (
        f"Ledger-Leck: seed={seed} schritt={schritt} aktion={aktion}: {lhs} != {rhs}"
    )


def _fuzz_ein_seed(seed: int, n_aktionen: int = 200) -> None:
    rng = random.Random(seed)
    layer = ObjectLayer(_GRID, _GRID, rng=rng)
    body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.5)
    hands = Hands()

    for schritt in range(n_aktionen):
        pos = (rng.randrange(_GRID), rng.randrange(_GRID))
        aktion = rng.choice(_AKTIONEN)
        boden_nah = [o for o, _ in layer.objects_near(pos, 1)]
        boden_hier = layer.objects_at(pos)

        if aktion == "spawn":
            layer.add(
                make_object(rng.choice(_MATERIALIEN), rng.uniform(0.05, 8.0)),
                (rng.randrange(_GRID), rng.randrange(_GRID)),
                source="spawned",
            )
        elif aktion == "grasp" and boden_nah:
            do_grasp(body, hands, layer, pos, rng.choice(boden_nah))
        elif aktion == "release" and hands.held:
            do_release(body, hands, layer, pos, rng.choice(hands.held))
        elif aktion == "strike" and hands.held:
            striker = rng.choice(hands.held)
            ziele = boden_hier + [o for o in hands.held if o is not striker]
            if ziele:
                do_strike(body, hands, layer, pos, striker, rng.choice(ziele), rng.random(), rng)
        elif aktion == "cut":
            klinge = rng.choice(hands.held + [None]) if hands.held else None
            ziele = boden_hier + [o for o in hands.held if o is not klinge]
            if ziele:
                do_cut(body, hands, layer, pos, klinge, rng.choice(ziele), rng.random())
        elif aktion == "eat":
            ziele = boden_hier + list(hands.held)
            if ziele:
                do_eat(body, hands, layer, pos, rng.choice(ziele))
        elif aktion == "decay":
            tick_decay(layer)
        elif aktion == "overload":
            body.fatigue = min(1.0, body.fatigue + 0.5)
            enforce_carry_budget(body, hands, layer, pos)
        elif aktion == "die":
            # Agent stirbt mit (bis zu) 2 gehaltenen Objekten: Hände fallen zu
            # Boden (neutral), Körper wird als Kadaver gemünzt (from_carcass) —
            # exakt der remove_dead()-Pfad, sim-frei nachgestellt.
            for obj in list(hands.held):
                hands.release(obj)
                layer.add(obj, pos)
            layer.add(make_object("carcass", body.body_mass), pos, source="from_carcass")
            body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.5)  # „nächster" Agent
            hands = Hands()

        _assert_invariante(layer, hands, seed, schritt, aktion)


def test_ledger_fuzzer_50_seeds_x_200_aktionen():
    for seed in range(50):
        _fuzz_ein_seed(seed)


def test_ledger_smoke_beispielkette():
    """Schneller Smoke-Check der Kern-Kette: spawn → grasp → strike → cut → eat → decay."""
    rng = random.Random(99)
    layer = ObjectLayer(_GRID, _GRID, rng=rng)
    body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=0.7)
    hands = Hands()
    hammer = make_object("granite", 1.0)
    flint = make_object("flint", 0.8)
    kadaver = make_object("carcass", 25.0)
    layer.add(hammer, (4, 4), source="spawned")
    layer.add(flint, (4, 4), source="spawned")
    layer.add(kadaver, (4, 4), source="from_carcass")

    do_grasp(body, hands, layer, (4, 4), hammer)
    schlag = do_strike(body, hands, layer, (4, 4), hammer, flint, 1.0, rng)
    assert schlag.fragments
    klinge = max(schlag.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))
    do_release(body, hands, layer, (4, 4), hammer)
    do_grasp(body, hands, layer, (4, 4), klinge)
    schnitt = do_cut(body, hands, layer, (4, 4), klinge, kadaver, 0.8)
    assert schnitt.extracted is not None
    assert do_eat(body, hands, layer, (4, 4), schnitt.extracted).ok
    tick_decay(layer)
    _assert_invariante(layer, hands, 99, -1, "beispielkette")
