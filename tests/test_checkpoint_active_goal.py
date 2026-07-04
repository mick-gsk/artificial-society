"""SubGoal.done_fn must be picklable (audit fix 22).

GoalPlanner.suggest_goals stored a local closure as SubGoal.done_fn. SubGoals
live on the agent's goal_stack and are pickled with every checkpoint, so any
mid-run save with an active suggested goal crashed with "Can't pickle local
object". done_fn is now a module-level, picklable NeedFulfilled instance.
"""

from __future__ import annotations

from artificial_society import simulation as sim_mod
from artificial_society.simulation import Simulation
from artificial_society.systems.goal_stack import NeedFulfilled, SubGoal

SMALL = dict(headless=True, grid_w=12, grid_h=8, initial_population=4)


def test_checkpoint_saves_with_active_suggested_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(tmp_path / "ckpt.pkl"))
    sim = Simulation(load_checkpoint=False, **SMALL)
    sim.agents[0].goal_stack.push(
        SubGoal(
            action="rub",
            reward_pred=0.5,
            max_ticks=20,
            label="need_test",
            done_fn=NeedFulfilled(0),
        )
    )

    sim._save_checkpoint()  # used to raise "Can't pickle local object"

    fresh = Simulation(load_checkpoint=True, **SMALL)
    assert not fresh.agents[0].goal_stack.is_empty(), "goal must survive the round-trip"
