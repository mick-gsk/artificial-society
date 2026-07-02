"""`last_action_target` — records WHO an attack/cooperate action targeted.

The viz layer needs to draw action lines (attack -> victim, cooperate -> partner).
`last_action_mode` already records WHAT an agent did each tick; this field records
WHO it was directed at, following the exact same reset-each-tick pattern.
"""

from __future__ import annotations

from artificial_society.simulation import Simulation


def _sim(pop=4):
    return Simulation(
        headless=True,
        load_checkpoint=False,
        seed=42,
        grid_w=10,
        grid_h=8,
        initial_population=pop,
    )


def test_attack_records_victim_id():
    sim = _sim()
    attacker, victim = sim.agents[0], sim.agents[1]
    for a in sim.agents:
        a.alive = True
    attacker.pos = (3, 3)
    victim.pos = (3, 3)
    # Force the attack to fire deterministically: aggression high enough that
    # threshold = max(0.05, aggression - 0.3) = 1.0, so `random.random() > 1.0`
    # (random.random() is always in [0, 1)) is unconditionally False — no seed
    # needed. Trust below 0 makes victim a valid target.
    attacker.genes["aggression"] = 1.3
    attacker.trust[victim.id] = -1.0
    for other in sim.agents[2:]:
        other.pos = (0, 0)  # keep out of attack range so victim is the only target

    reward = attacker._attack(sim.agents, {})

    assert reward != 0.0, "attack did not execute — test setup failed to force it"
    assert attacker.last_action_target == victim.id


def test_cooperate_records_partner_id():
    sim = _sim()
    donor, recipient = sim.agents[0], sim.agents[1]
    for a in sim.agents:
        a.alive = True
    donor.pos = (4, 4)
    recipient.pos = (4, 4)
    for other in sim.agents[2:]:
        other.pos = (0, 0)  # keep out of cooperate proximity radius
    donor.energy = 200.0  # >= COOP_SHARE_THRESHOLD_DONOR
    recipient.energy = 30.0  # < COOP_SHARE_THRESHOLD_RECV

    reward = donor._cooperate(sim.agents, {}, tick=0)

    assert reward > 0.0, "cooperate share did not execute — test setup failed to force it"
    assert donor.last_action_target == recipient.id


def test_last_action_target_resets_when_no_action_next_tick():
    sim = _sim()
    attacker, victim = sim.agents[0], sim.agents[1]
    for a in sim.agents:
        a.alive = True
    attacker.pos = (3, 3)
    victim.pos = (3, 3)
    attacker.genes["aggression"] = 1.3
    attacker.trust[victim.id] = -1.0
    for other in sim.agents[2:]:
        other.pos = (0, 0)

    attacker._attack(sim.agents, {})
    assert attacker.last_action_target == victim.id

    # `_attack` alone does not reset a stale target from an earlier call — the
    # reset lives in Agent.update, mirroring the unconditional `mode = "idle"`
    # reset at the start of each tick's action dispatch. Simulate a stale value
    # left over from a prior tick and confirm a fresh update() tick clears it
    # unconditionally, before any action for *this* tick is even decided. Move
    # the victim far away and zero aggression/trust so this tick's brain-driven
    # action list cannot possibly re-trigger an attack or cooperate that would
    # set a fresh (non-None) target and mask a broken reset.
    attacker.last_action_target = victim.id
    attacker.genes["aggression"] = 0.0
    attacker.trust.clear()
    victim.pos = (9, 7)
    attacker.reproduction_cooldown = 999  # also suppress reproduction side effects

    attacker.update(sim.world, sim.agents, tick=1)

    assert attacker.last_action_target is None
