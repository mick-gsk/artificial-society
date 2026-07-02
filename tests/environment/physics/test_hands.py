"""Hände v1: zwei Hände, Masse-Budget — der Transport-Engpass vor der Behälter-Erfindung."""

from __future__ import annotations

import pytest

from artificial_society.environment.physics.body import MAX_HELD, Body, Hands
from artificial_society.environment.physics.objects import make_object


def _body():
    return Body(body_mass=70.0, strength=0.7)  # Kapazität 18.48 kg


def test_whole_carcass_exceeds_capacity():
    # DER Erfindungsdruck: ein 25-kg-Kadaver ist als Ganzes untragbar.
    hands = Hands()
    assert not hands.can_grasp(make_object("carcass", 25.0), _body())


def test_light_pieces_are_graspable_up_to_two_hands():
    hands = Hands()
    body = _body()
    assert hands.grasp(make_object("raw_meat", 3.0), body)
    assert hands.grasp(make_object("flint", 0.8), body)
    assert hands.carried_mass_kg() == pytest.approx(3.8)
    # Dritte Hand gibt es nicht — auch für ein federleichtes Objekt.
    assert not hands.can_grasp(make_object("plant_fiber", 0.05), body)
    assert MAX_HELD == 2


def test_mass_budget_counts_total_load():
    hands = Hands()
    body = _body()
    assert hands.grasp(make_object("granite", 15.0), body)
    # Zweite Hand frei, aber 15 + 5 > 18.48 → nein.
    assert not hands.can_grasp(make_object("granite", 5.0), body)


def test_release_frees_the_hand():
    hands = Hands()
    body = _body()
    stone = make_object("granite", 15.0)
    assert hands.grasp(stone, body)
    hands.release(stone)
    assert hands.carried_mass_kg() == 0.0
    assert hands.grasp(make_object("carcass", 3.0), body)


def test_release_unheld_object_raises():
    with pytest.raises(ValueError):
        Hands().release(make_object("flint", 0.8))
