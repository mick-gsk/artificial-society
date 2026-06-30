"""
Task C.3 — Remove inline reward pumps from invention calls in agents/agent.py.

Asserts via source inspection that:
- flat "reward += 0.5" is gone
- flat "reward += 1.0" is gone
- flat "reward += 0.3" (bare, i.e. the old cooking bonus) is gone
- the curiosity-intrinsic term "reward += 0.3 * intrinsic" is still present
"""

import inspect
import importlib

import artificial_society.agents.agent as agent_mod


def _get_src() -> str:
    return inspect.getsource(agent_mod)


def test_no_flat_need_reward():
    src = _get_src()
    assert "reward += 0.5" not in src, (
        "Flat need-invention reward 'reward += 0.5' must be removed (C.3)"
    )


def test_no_flat_try_invention_reward():
    src = _get_src()
    assert "reward += 1.0" not in src, (
        "Flat try-invention reward 'reward += 1.0' must be removed (C.3)"
    )


def test_no_flat_cook_bonus():
    src = _get_src()
    # The old flat cooking bonus was "reward += 0.3\n" (bare scalar).
    # The intrinsic term "reward += 0.3 * intrinsic" must remain, so we check
    # for the bare form only: "reward += 0.3" NOT followed by " * intrinsic".
    import re
    matches = re.findall(r"reward \+= 0\.3[^\s*]", src)
    # Allow only the line-ending form (nothing after 0.3 on that token)
    bare = [m for m in matches if not m.startswith("reward += 0.3 *")]
    assert not bare, (
        f"Bare flat cooking reward 'reward += 0.3' must be removed (C.3); found: {bare}"
    )


def test_curiosity_intrinsic_still_present():
    src = _get_src()
    assert "reward += 0.3 * intrinsic" in src, (
        "Curiosity-intrinsic term 'reward += 0.3 * intrinsic' must NOT be removed (C.3)"
    )
