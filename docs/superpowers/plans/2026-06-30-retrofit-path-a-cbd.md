# Retrofit Path-A (C→B→D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make invention policy-coupled and value-driven so the learned/social arm advances task-functional frontiers (DV2 `accumulated_useful_depth`) instead of churning — flipping the Stage-0a gate from −21.75 (discovery-matched) toward > 0.

**Architecture:** Three coupled mechanisms in binding order. **C** kills the flat novelty-pump reward and replaces it with realized homeostatic value + a *marginal* gain over a transmitted, tribe-level functional frontier. **B** makes the brain decide *when* to invent via a learned, separately-credited Bernoulli "research" head (the existing `research_drive` dim 6 is currently computed and discarded). **D** makes the agent decide *what* to combine via an imagined means-ends search over a pure, uncounted `_combine_pure()` twin, executing only the argmax. A short Phase 0 first fixes the `discovered_by`/`tick`/`uses` instrumentation gap Schritt A uncovered, so DV1/DV3 become trustworthy. The behaviour change is intentional → the golden trajectory goes RED and is regenerated once.

**Tech Stack:** Python 3.9, NumPy, PyTorch (PPO brain), pytest. `../venv/bin/python`. Determinism via `artificial_society.rng.seed_all`.

## Global Constraints

- **Determinism is sacred.** All randomness routes through `artificial_society.rng.seed_all` (it seeds global `random`, NumPy, and torch together). Never call `random.seed`/`np.random.seed` directly. Never edit a determinism test to make it pass.
- **Hot files = frozen contract, routed via `core-lead` (serial branch).** Hot: `simulation.py`, `agents/agent.py`, `agents/brain.py`, `environment/materials.py`, `systems/registry.py`, `systems/_builtins.py`. **Non-hot (systems lane):** `systems/invention.py`, `systems/need_driven_invention.py`, `agents/endocrine.py`, `research/*`.
- **Golden regen is a deliberate, manual act**, run once after the behaviour-changing tasks land, never to "make a test pass":
  ```bash
  PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy ../venv/bin/python -c \
    "import json; from tests._util import compute_trajectory; \
     json.dump(compute_trajectory(), open('tests/golden_trajectory.json','w'), indent=0)"
  ```
- **Gate before pushing:** `bash scripts/check.sh` (ruff on changed files + full pytest incl. determinism contract). A RED golden after the C/B/D tasks is expected; flag it for core-lead, do not silence it.
- **Compute-match integrity (D):** imagined/predicted combines MUST NOT go through the instrumented `combine_vectors` (research/instrument.py patches it by function identity and feeds the executed-call count to the recombiner's budget). Imagined evals call `_combine_pure` (a distinct function object) and read vectors via a non-incrementing lookup.
- **Branch:** `core/retrofit-path-a-cbd` (hot files present → core lane). All tests run with `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.
- **Target metric:** DV2 `accumulated_useful_depth` from `research/metrics.py` (validated in Schritt A). Re-pilot at the locked pre-registration regime (n=12 × 1500 ticks, grid 30×20, pop 24, CPU, `CUDA_VISIBLE_DEVICES=-1`).

---

## File Structure

**Phase 0 — Instrumentation (make DV1/DV3 trustworthy):**
- `environment/materials.py` (HOT) — split `get_vector` (counted, increments `uses`) from a pure lookup; add an explicit `record_use`; keep `register` storing `discovered_by`/`tick`.
- `systems/invention.py`, `systems/need_driven_invention.py` — call `record_use` at real homeostatic-relief sites; ensure real `agent.id` reaches `register`.
- `research/run_single.py`, `research/export.py` — export the (now real) `discovered_by`/`tick`; add per-recipe adopter set.

**Phase C — value reward (kill the novelty pump):**
- `systems/invention.py`, `systems/need_driven_invention.py` — replace flat `NEW_DISCOVERY_BONUS` with `material_reward + RATCHET_GAIN · marginal`; add a transmitted tribe-level frontier (`_lineage_frontier.py`).
- `systems/_lineage_frontier.py` (NEW) — module-level, run-resettable per-tribe functional frontier (the social ratchet object).
- `agents/agent.py` (HOT) — remove inline `+0.5/+1.0/+0.3` bonuses and redundant `apply_discovery` pumps.
- `tests/systems/test_invention_rewards.py` — rewrite for the value-gated economics.

**Phase B — learned WHEN (research head):**
- `agents/brain.py` (HOT) — rescale `research_drive` to [0,1]; add a separate Bernoulli "invent" head with its own log-prob + advantage term.
- `agents/agent.py` (HOT) — gate invention on the learned head instead of `tick%3`/cooldown/RNG.
- `tests/agents/test_research_head.py` (NEW).

**Phase D — learned WHAT (means-ends search):**
- `environment/materials.py` (HOT) — factor reaction rules into `_combine_impl(..., deterministic)`; `combine_vectors` (counted) and `_combine_pure` (uncounted twin).
- `systems/invention.py`, `systems/need_driven_invention.py` — replace `random.choice` input/action selection with an imagined means-ends search using `_combine_pure`; add a value-of-information term.
- `tests/systems/test_means_ends.py` (NEW).

**Phase V — verify (golden regen + re-pilot + gate):**
- `tests/golden_trajectory.json`, `tests/test_headless.py` — regenerate / update once.
- `research/run_pilot.py`, `research/analyze_gate.py` — re-run the gate on DV2.

---

## Phase 0 — Instrumentation

### Task 0.1: Pure vector lookup + explicit `record_use` (split the `uses` pollution)

**Files:**
- Modify: `artificial_society/environment/materials.py:288-293` (`get_vector`), add `record_use` + `peek_vector`
- Test: `tests/environment/test_registry_uses.py` (Create)

**Interfaces:**
- Produces: `DiscoveryRegistry.peek_vector(mat_id) -> np.ndarray` (no side effect); `DiscoveryRegistry.record_use(mat_id) -> None` (increments `uses`); `get_vector` retained as a thin `peek_vector` + `record_use` for backward-compat callers that genuinely consume the material.

- [ ] **Step 1: Write the failing test**
```python
# tests/environment/test_registry_uses.py
import numpy as np
from artificial_society.environment.materials import DiscoveryRegistry

def test_peek_does_not_increment_but_record_use_does():
    reg = DiscoveryRegistry()
    mid = reg.register(np.ones(12, dtype=np.float32), discoverer_id=7, tick=3, recipe=("strike", "a", "b"))
    e = next(x for x in reg.entries if x["id"] == mid)
    assert e["uses"] == 0
    reg.peek_vector(mid); reg.peek_vector(mid)
    assert e["uses"] == 0          # peek is side-effect free
    reg.record_use(mid)
    assert e["uses"] == 1          # explicit use is counted
    assert e["discovered_by"] == 7 and e["tick"] == 3
```

- [ ] **Step 2: Run test to verify it fails**
Run: `SDL_VIDEODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_registry_uses.py -v`
Expected: FAIL (`peek_vector`/`record_use` not defined).

- [ ] **Step 3: Implement in `materials.py`** (replace `get_vector` body, add two methods)
```python
    def peek_vector(self, mat_id: str) -> np.ndarray:
        """Side-effect-free vector lookup (does NOT count as a use)."""
        for entry in self.entries:
            if entry["id"] == mat_id:
                return entry["vector"].copy()
        return np.zeros(N_PROPS, dtype=np.float32)

    def record_use(self, mat_id: str) -> None:
        """Mark a material as genuinely used (homeostatic relief / consumption)."""
        for entry in self.entries:
            if entry["id"] == mat_id:
                entry["uses"] += 1
                return

    def get_vector(self, mat_id: str) -> np.ndarray:
        """Backward-compat: lookup + count as a use (legacy consumers)."""
        self.record_use(mat_id)
        return self.peek_vector(mat_id)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `SDL_VIDEODRIVER=dummy ../venv/bin/python -m pytest tests/environment/test_registry_uses.py -v` → PASS

- [ ] **Step 5: Audit `get_vector` call-sites — switch non-consuming reads to `peek_vector`.**
Run: `grep -rn "get_vector" artificial_society/ | grep -v "def get_vector"`
For each: if the call is a *read for computation* (e.g. building combine inputs, rendering, stats) → change to `peek_vector`. If it is a genuine *consumption* (agent eats/uses the material for homeostatic relief) → leave as `get_vector` or change to explicit `record_use` + `peek_vector`. Document each decision in the commit body.

> ⚠️ This changes per-tick behaviour (uses counts shift) → contributes to the golden going RED (regenerated in Phase V). That is expected and intended.

- [ ] **Step 6: Commit**
```bash
git add artificial_society/environment/materials.py tests/environment/test_registry_uses.py
git commit -m "core(materials): split peek_vector/record_use from get_vector (de-pollute uses)"
```

### Task 0.2: Real `discovered_by`/`tick` in the export + adopter set

**Files:**
- Read first: `artificial_society/research/run_single.py`, `research/export.py:21-33` (`entry_to_json`)
- Modify: `research/run_single.py` (export `discovered_by`/`tick` verbatim from the registry entry; add a per-recipe adopter set keyed by agent id)
- Test: `tests/test_research_export_fields.py` (Create)

**Interfaces:**
- Consumes: `DiscoveryRegistry.entries` entries with real `discovered_by` (agent id) + `tick`, from Task 0.1's preserved `register`.
- Produces: exported entries carry `discovered_by >= 0`, real `tick`, and `adopters` (list of distinct agent ids that re-used the recipe) — what `transmitted_frontier_advances` (DV3) needs.

- [ ] **Step 1: Diagnose the Schritt-A gap.** Run a 1-seed × 200-tick learned export and check whether `register` receives a real `agent.id`:
```bash
SDL_VIDEODRIVER=dummy CUDA_VISIBLE_DEVICES=-1 ../venv/bin/python -c "
from artificial_society.research.run_single import run_learned
entries, *_ = run_learned(seed=1001, ticks=200, grid_w=30, grid_h=20, pop=24)
import collections
db = collections.Counter(e.get('discovered_by', -1) >= 0 for e in entries)
tk = collections.Counter(e.get('tick', 0) > 0 for e in entries)
print('discovered_by>=0:', db, ' tick>0:', tk)"
```
Expected (current bug): mostly `False`. Identify whether the cause is (a) `entry_to_json` reading the wrong key, (b) `agent.id` being -1/unset at the `register` call, or (c) `run_single` overwriting the field. Fix the root cause (the field already exists in `register`).

- [ ] **Step 2: Write the failing test**
```python
# tests/test_research_export_fields.py
from artificial_society.research.run_single import run_learned

def test_learned_export_carries_real_attribution():
    entries, *_ = run_learned(seed=1001, ticks=200, grid_w=30, grid_h=20, pop=24)
    assert any(e.get("discovered_by", -1) >= 0 for e in entries), "no agent attribution"
    assert any(e.get("tick", 0) > 0 for e in entries), "no real tick"
```

- [ ] **Step 3: Run to verify it fails**, then **fix the root cause** in `run_single.py`/`export.py` so the real `discovered_by`/`tick` survive to the export; add an `adopters` list per entry (collect distinct agent ids passed to `record_use`, exported alongside `uses`).

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Re-validate DV1/DV3 are now meaningful** on a fresh tiny pair via `analyze_gate --compare-dvs` (the `transmitted_frontier_advances` row should leave "instrumentation-blocked").

- [ ] **Step 6: Commit**
```bash
git add artificial_society/research/run_single.py artificial_society/research/export.py tests/test_research_export_fields.py
git commit -m "research(stage0a): export real discovered_by/tick + adopter set (unblock DV1/DV3)"
```

---

## Phase C — Value reward (kill the novelty pump)

### Task C.1: Transmitted tribe-level functional frontier (the social ratchet)

**Files:**
- Create: `artificial_society/systems/_lineage_frontier.py`
- Test: `tests/systems/test_lineage_frontier.py`

**Interfaces:**
- Produces:
  - `frontier_value(tribe_id: int) -> float` (current best realized functional value for that tribe; 0.0 if none)
  - `update_frontier(tribe_id: int, value: float) -> float` returns `marginal = max(0.0, value - prev)`, then raises the frontier to `max(prev, value)`
  - `reset_frontiers() -> None` (call at run start, next to `DISCOVERY_REGISTRY.reset()`)
- Rationale: keying the frontier by `tribe_id` makes it a **shared, transmitted social object** (all tribe members read/raise the same ratchet) without touching hot tribe files; `marginal` is continuous and unbounded in count (concave only via `material_reward`'s own shape), avoiding [0,1]-clip saturation.

- [ ] **Step 1: Write the failing test**
```python
# tests/systems/test_lineage_frontier.py
from artificial_society.systems import _lineage_frontier as lf

def test_frontier_marginal_and_monotone():
    lf.reset_frontiers()
    assert lf.frontier_value(1) == 0.0
    assert lf.update_frontier(1, 0.8) == 0.8       # first advance = full value
    assert lf.update_frontier(1, 0.5) == 0.0       # below frontier = no marginal
    assert round(lf.update_frontier(1, 1.1), 6) == 0.3  # only the increment counts
    assert lf.frontier_value(1) == 1.1
    assert lf.frontier_value(2) == 0.0             # tribes are independent
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**
```python
# artificial_society/systems/_lineage_frontier.py
"""Transmitted, tribe-level functional frontier — the social ratchet for Path-A.

Keyed by tribe id so every tribe member shares one monotone 'best realized
functional value' object; an invention's reward credit is the *marginal* gain it
adds over that shared frontier (guided variation, Boyd & Richerson 1985). Module
-level + run-resettable to match the DISCOVERY_REGISTRY singleton lifecycle.
"""
from __future__ import annotations

_FRONTIER: dict[int, float] = {}


def reset_frontiers() -> None:
    _FRONTIER.clear()


def frontier_value(tribe_id: int) -> float:
    return _FRONTIER.get(int(tribe_id), 0.0)


def update_frontier(tribe_id: int, value: float) -> float:
    tid = int(tribe_id)
    prev = _FRONTIER.get(tid, 0.0)
    marginal = max(0.0, float(value) - prev)
    if value > prev:
        _FRONTIER[tid] = float(value)
    return marginal
```

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Wire `reset_frontiers()` into the run start** — call it wherever `DISCOVERY_REGISTRY.reset()` is called (`research/run_single.py`, and any sim reset path). Grep: `grep -rn "DISCOVERY_REGISTRY.reset" artificial_society/`.

- [ ] **Step 6: Commit**
```bash
git add artificial_society/systems/_lineage_frontier.py tests/systems/test_lineage_frontier.py
git commit -m "systems(invention): transmitted tribe-level functional frontier (ratchet object)"
```

### Task C.2: Value-gated reward in `invention.py` (replace the flat bonus)

**Files:**
- Modify: `systems/invention.py:85` (`NEW_DISCOVERY_BONUS` → `RATCHET_GAIN`), `:145-175` (`agent_try_invention` reward), `:219-243` (`agent_try_cook`)
- Modify: `systems/need_driven_invention.py:378-420`
- Rewrite: `tests/systems/test_invention_rewards.py`

**Interfaces:**
- Consumes: `_lineage_frontier.update_frontier`, `material_reward(new_vec, agent_state)` (materials.py:560), `agent.tribe_id` (or the agent's tribe accessor — confirm the attribute name during execution; fall back to `agent.id` lineage if no tribe).
- Produces: invention functions return `legacy_reward + material_reward + RATCHET_GAIN * marginal` (no flat novelty bonus, no rediscovery floor inflation).

- [ ] **Step 1: Rewrite the reward test (the spec for C)**
```python
# tests/systems/test_invention_rewards.py  (replaces novelty-pump assertions)
from artificial_society.systems import invention as inv
from artificial_society.systems import _lineage_frontier as lf

def test_no_flat_novelty_pump_constant():
    # the flat, value-blind bonus is gone
    assert not hasattr(inv, "NEW_DISCOVERY_BONUS")
    assert hasattr(inv, "RATCHET_GAIN")

def test_marginal_value_is_the_reward_signal(monkeypatch):
    lf.reset_frontiers()
    # a higher-value discovery for a tribe yields a positive marginal-scaled reward;
    # repeating a no-better discovery yields ~0 marginal (only homeostatic base remains)
    m1 = lf.update_frontier(5, 0.9)
    m2 = lf.update_frontier(5, 0.9)
    assert m1 > 0 and m2 == 0.0
```

- [ ] **Step 2: Run to verify it fails** (imports `NEW_DISCOVERY_BONUS`-era constants).

- [ ] **Step 3: Implement the reward rewrite.** In `invention.py`:
  - Replace `NEW_DISCOVERY_BONUS = 4.5` with `RATCHET_GAIN = 4.5  # scales marginal functional gain over the tribe frontier` (freeze pre-pilot).
  - In `agent_try_invention` (after `mat_id = DISCOVERY_REGISTRY.register(...)`), replace the `is_new_discovery` flat-bonus block (`:162-166`) with:
```python
    value = material_reward(new_vec, agent_state)        # realized homeostatic value
    tribe_id = getattr(agent, "tribe_id", agent.id)
    marginal = update_frontier(tribe_id, value)          # gain over the shared ratchet
    emergent_reward = value + RATCHET_GAIN * marginal    # no flat novelty bonus
```
  - Remove the `REDISCOVERY_REWARD` floor (`max(emergent_reward, REDISCOVERY_REWARD)`); re-discoveries now simply earn their (small) realized value + 0 marginal.
  - Apply the same pattern in `agent_try_cook` (`:232-239`) and delete its debug `print`.
  - Keep `endocrine.apply_discovery` but gate it on `marginal > 0` (dopamine for genuine progress, not novelty): `if marginal > 0 and hasattr(agent, "endocrine"): agent.endocrine.apply_discovery(min(1.0, RATCHET_GAIN * marginal))`.
  - In `need_driven_invention.py`, mirror: `base_reward = material_reward(...)`, add need-fulfilment as today, then `marginal = update_frontier(tribe_id, base_reward + fulfillment_normalized*NEED_FULFILLMENT_BONUS)`; reward the marginal; drop `NEW_DISCOVERY_BONUS`/`REDISCOVERY_REWARD`.
  - `import` `update_frontier` from `systems._lineage_frontier` in both modules.

- [ ] **Step 4: Run to verify the reward test passes.**
Run: `SDL_VIDEODRIVER=dummy ../venv/bin/python -m pytest tests/systems/test_invention_rewards.py -v` → PASS

- [ ] **Step 5: Commit**
```bash
git add artificial_society/systems/invention.py artificial_society/systems/need_driven_invention.py tests/systems/test_invention_rewards.py
git commit -m "systems(invention): value-gated marginal reward (kill the flat novelty pump)"
```

### Task C.3: Remove inline reward pumps in `agent.py` (HOT)

**Files:**
- Modify: `agents/agent.py:1170-1191` (need-driven `+0.5`+`apply_discovery(1.0)`; exploratory `+1.0`+`apply_discovery(1.0)`; cooking `+0.3`)

**Interfaces:**
- Consumes: the float reward already returned by `agent_try_invention`/`agent_invent_from_need`/`agent_try_cook` (now value-gated from C.2).
- Produces: `reward += <the function's own returned value>` only — no hardcoded constants, no second `apply_discovery` pump (the endocrine response now lives inside the invention functions).

- [ ] **Step 1: Write a lane test asserting the new wiring**
```python
# tests/agents/test_invention_reward_wiring.py
import inspect
from artificial_society.agents import agent as agent_mod

def test_no_hardcoded_invention_bonuses_in_agent_update():
    src = inspect.getsource(agent_mod)
    # the flat inline bonuses are gone; reward comes from the invention fn return
    assert "reward += 0.5" not in src
    assert "reward += 1.0" not in src
    assert "reward += 0.3\n" not in src  # cooking flat bonus (keep intrinsic line intact)
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Edit `agent.py`** — replace the three blocks so reward accrues from the returned value and the redundant `apply_discovery` calls are deleted:
```python
        if self._need_inv_cooldown <= 0:
            compute_need_vector(self, current_cell)
            reward += agent_invent_from_need(self, world, *self.pos, tick)
            self._need_inv_cooldown = NEED_INVENTION_INTERVAL
        else:
            self._need_inv_cooldown -= 1
        # exploratory + cooking gating is replaced in Phase B; for now keep schedule:
        if tick % 3 == 0 and random.random() < inv_prob:
            reward += agent_try_invention(self, world, *self.pos)
        if tick % 4 == 0 and random.random() < 0.18:
            reward += agent_try_cook(self, world, *self.pos)
```
(Leave `reward += 0.3 * intrinsic` at `:1202` untouched — that is the curiosity-intrinsic term, not a novelty pump.)

- [ ] **Step 4: Run the wiring test + full suite (golden expected RED).**
Run: `SDL_VIDEODRIVER=dummy ../venv/bin/python -m pytest tests/agents/test_invention_reward_wiring.py -v` → PASS
Run: `SDL_VIDEODRIVER=dummy ../venv/bin/python -m pytest tests/test_regression_golden.py -q` → **expected FAIL** (intentional behaviour change). Do not fix by editing the test.

- [ ] **Step 5: Commit (note the intentional RED golden)**
```bash
git add artificial_society/agents/agent.py tests/agents/test_invention_reward_wiring.py
git commit -m "core(agent): route invention reward from value-gated fns, drop inline pumps [golden RED: intentional]"
```

---

## Phase B — Learned WHEN (separate research head)

### Task B.1: Rescale `research_drive` to [0,1] and expose it

**Files:**
- Modify: `agents/brain.py:372` (`research_drive` extraction)
- Test: `tests/agents/test_research_head.py`

**Interfaces:**
- Produces: `brain.act(...)["research_drive"]` ∈ [0,1] (was tanh ∈ [-1,1]); `RESEARCH_DRIVE_THRESHOLD = 0.4` now interpreted on [0,1].

- [ ] **Step 1: Write the failing test**
```python
# tests/agents/test_research_head.py
import torch
from artificial_society.agents.brain import Brain  # confirm class name during execution

def test_research_drive_in_unit_interval():
    torch.manual_seed(0)
    b = Brain()  # use the real constructor signature
    out = b.act(b.example_features(), b.initial_hidden())  # use real helpers/fixtures
    assert 0.0 <= out["research_drive"] <= 1.0
```
*(During execution, adapt to the real `Brain` constructor + feature/hidden fixtures used elsewhere in `tests/agents/`.)*

- [ ] **Step 2: Run to verify it fails** (current range is [-1,1]).

- [ ] **Step 3: Implement at `brain.py:372`**
```python
        research_drive = (0.5 * (float(action_list[6]) + 1.0)) if len(action_list) > 6 else 0.0
```

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**
```bash
git add artificial_society/agents/brain.py tests/agents/test_research_head.py
git commit -m "core(brain): rescale research_drive to [0,1] for the invent gate"
```

### Task B.2: Separate Bernoulli "invent" head with its own credit path

**Files:**
- Modify: `agents/brain.py:139-140` (heads), `:209-230` (forward/act + log-prob), `:447-464` (PPO update)
- Test: extend `tests/agents/test_research_head.py`

**Interfaces:**
- Produces:
  - a new head `self.invent_logit = nn.Linear(hidden_size, 1)`; `act(...)` returns `"invent_prob"` (sigmoid), a sampled boolean `"invent"` (Bernoulli, RNG via the globally-seeded torch), and its own `"invent_log_prob"`.
  - the PPO update adds the Bernoulli head's policy-gradient term (`-(invent_log_prob_new − invent_log_prob_old).exp()`-ratio · advantage, PPO-clipped) to `actor_loss`, so the invent decision gets credit *isolated* from the 7-dim Gaussian's summed log-prob.
- Rationale: the existing single Gaussian sums log-probs across all 7 dims (`brain.py:230`), so a shared scalar can't isolate "when to invent" credit. A separate Bernoulli head with its own log-prob and the same GAE advantage gives the decision a real gradient (design §B forced fix).

- [ ] **Step 1: Write failing tests** — `invent_prob` ∈ [0,1]; `act` returns `invent` (bool) + `invent_log_prob`; the stored transition carries the invent log-prob; two seeded `act` calls are reproducible (determinism).

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.** Add the head in `__init__`; in the forward/act path compute `invent_prob = torch.sigmoid(self.invent_logit(next_hidden))`, sample `invent ~ Bernoulli(invent_prob)` (torch, seeded), store `invent_log_prob = Bernoulli(invent_prob).log_prob(invent)`. In the replay buffer carry `invent`, `invent_log_prob`. In `maybe_train`/PPO update recompute `invent_log_prob_new` from the stored obs and add a second clipped surrogate term using the *same* normalized `advantages`. Keep `research_drive` (dim 6) as an auxiliary signal but make the **invent head** the decision (dim 6 can feed the head's input or be retired — pick one and document).

- [ ] **Step 4: Run tests + a 60-tick determinism smoke** (`act` reproducible under fixed seed).

- [ ] **Step 5: Commit**
```bash
git add artificial_society/agents/brain.py tests/agents/test_research_head.py
git commit -m "core(brain): separate Bernoulli invent head with isolated PPO credit"
```

### Task B.3: Gate invention on the learned head (replace the fixed schedule)

**Files:**
- Modify: `agents/agent.py:1073-1091` (read `invent`/`research_drive`), `:1170-1191` (replace `tick%3`/cooldown/RNG gate)

**Interfaces:**
- Consumes: `brain_step["invent"]` (bool) + `brain_step["research_drive"]` ∈ [0,1] from B.2/B.1.
- Produces: invention attempted iff `brain_step["invent"]` (with a deterministic bootstrap floor for the first ~N ticks + a small seeded ε-exploration so the head has signal to learn from), replacing `tick%3 and random()<inv_prob` and the cooldown counter.

- [ ] **Step 1: Write a lane test** asserting the fixed schedule is gone and the decision reads the brain:
```python
# tests/agents/test_invent_gate.py
import inspect
from artificial_society.agents import agent as agent_mod
def test_invention_gate_is_policy_driven():
    src = inspect.getsource(agent_mod)
    assert "tick % 3 == 0 and random.random()" not in src
    assert 'brain_step["invent"]' in src
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — extract `invent = brain_step["invent"]`; gate the exploratory/need-driven invention on `invent` (plus a bootstrap floor: `invent or tick < INVENT_BOOTSTRAP_TICKS or random.random() < INVENT_EPSILON`, constants frozen pre-pilot, RNG already seeded by `seed_all`). Remove the cooldown counter and `tick%3` gate.

- [ ] **Step 4: Run the gate test + full suite (golden still RED).**

- [ ] **Step 5: Commit**
```bash
git add artificial_society/agents/agent.py tests/agents/test_invent_gate.py
git commit -m "core(agent): policy-gate invention on the learned invent head [golden RED: intentional]"
```

---

## Phase D — Learned WHAT (imagined means-ends search)

### Task D.1: Pure, uncounted `_combine_pure` twin

**Files:**
- Modify: `environment/materials.py:430-476` (factor into `_combine_impl(..., deterministic)`; keep `combine_vectors`; add `_combine_pure`)
- Test: `tests/environment/test_combine_pure.py`

**Interfaces:**
- Produces:
  - `_combine_impl(vec_a, vec_b, action, env, *, deterministic: bool) -> np.ndarray | None` — the reaction rules; when `deterministic`, the `rub` ignition branch uses its *expected* outcome (no `random.random()`/`random.uniform`).
  - `combine_vectors(vec_a, vec_b, action, env)` = `_combine_impl(..., deterministic=False)` — **unchanged identity/behaviour** (instrument still patches this).
  - `_combine_pure(vec_a, vec_b, action, env)` = `_combine_impl(..., deterministic=True)` — a **distinct function object** the instrument's identity check never matches.
- Rationale: imagined evals must be deterministic *and* uncounted (Global Constraint). `combine_vectors` keeps identical behaviour, so it does not by itself move the golden.

- [ ] **Step 1: Write the failing test**
```python
# tests/environment/test_combine_pure.py
import numpy as np
from artificial_society.environment import materials as M

def test_combine_pure_is_deterministic_and_distinct_object():
    assert M._combine_pure is not M.combine_vectors
    a = M.MATERIALS["dry_grass"]; b = M.MATERIALS["flint"]; env = {"moisture": 0.5}
    r1 = M._combine_pure(a, b, "rub", env)
    r2 = M._combine_pure(a, b, "rub", env)
    assert (r1 is None and r2 is None) or np.allclose(r1, r2)  # no RNG → identical
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — move the body of `combine_vectors` into `_combine_impl(...)`; in the `rub` branch replace the stochastic ignition (lines 452-455) with a deterministic expectation when `deterministic=True` (e.g. gate on `ignite_p > 0.15` and set `heat_emission = 0.5 + 0.15`, `light_emission = 0.3 + 0.1` — the means of the former `uniform` draws; under `deterministic=False` keep the exact current random code so `combine_vectors` is byte-identical). Define the two public wrappers.

- [ ] **Step 4: Run the test + confirm `combine_vectors` behaviour is unchanged** (a quick equality check against a saved sample under fixed seed; the golden for `combine_vectors` proper should NOT shift from this task alone).

- [ ] **Step 5: Commit**
```bash
git add artificial_society/environment/materials.py tests/environment/test_combine_pure.py
git commit -m "core(materials): _combine_pure deterministic twin (uncounted, for imagined search)"
```

### Task D.2: Means-ends candidate search in `invention.py`

**Files:**
- Create: `systems/_means_ends.py` (the search; non-hot)
- Modify: `systems/invention.py:126-129` + `_choose_action_by_need:408-410`; `systems/need_driven_invention.py:207-217`
- Test: `tests/systems/test_means_ends.py`

**Interfaces:**
- Consumes: `_combine_pure` (materials), `DiscoveryRegistry.peek_vector` (no use-count), `material_reward`, `_lineage_frontier.frontier_value`, `PRIMITIVE_ACTIONS`.
- Produces:
  - `select_best_combine(available, agent_state, env, tribe_id, rng, *, top_n=6, voi_weight=0.3) -> tuple[str, str|None, str]` returning `(mat_a, mat_b, action)`:
    1. score each candidate input by `material_reward(peek_vector)`; keep top-N need-relevant inputs.
    2. for each `(a, b, action)` over the top-N × `PRIMITIVE_ACTIONS`, predict `pred = _combine_pure(...)`; **skip None**.
    3. score `= material_reward(pred, agent_state) + voi_weight · novelty(pred)` where `novelty = distance of pred to the nearest known registry/base vector` (value-of-information term, design §D forced fix against greedy collapse), plus `RATCHET_GAIN · max(0, material_reward(pred) − frontier_value(tribe_id))`.
    4. return the argmax (seeded softmax via `rng` for a little stochasticity); exclude the `rub` ignition branch from scoring (use its deterministic expectation).
- Rationale: replaces blind `random.choice`; imagined evals use `_combine_pure` only → not compute-counted. Execution still calls the real `combine_vectors` once.

- [ ] **Step 1: Write the failing test**
```python
# tests/systems/test_means_ends.py
import random, numpy as np
from artificial_society.systems._means_ends import select_best_combine
from artificial_society.environment.materials import MATERIALS

def test_picks_a_valid_triple_without_counting_combines():
    from artificial_society.research.instrument import count_combine_calls
    avail = ["dry_grass", "flint", "stone", "raw_meat"]
    state = {"energy": 0.2, "cold": True, "dark": False}
    with count_combine_calls() as cc:
        a, b, action = select_best_combine(avail, state, {"moisture": 0.5}, tribe_id=1, rng=random.Random(0))
    assert a in avail and action  # a valid plan
    assert cc.n == 0              # imagined search did NOT touch the counted combine_vectors
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `_means_ends.select_best_combine` per the interface; then in `invention.py` replace the `random.choice` input selection (`:126-129`) and the action choice in `_choose_action_by_need` (`:408-410`) with one `select_best_combine(...)` call (keep a seeded exploration floor). Mirror in `need_driven_invention.py` (`:207-217`). Pass the agent's `rng`-safe randomness (global `random`, already seeded).

- [ ] **Step 4: Run the means-ends test + reward + invention suites.**

- [ ] **Step 5: Commit**
```bash
git add artificial_society/systems/_means_ends.py artificial_society/systems/invention.py artificial_society/systems/need_driven_invention.py tests/systems/test_means_ends.py
git commit -m "systems(invention): imagined means-ends selection via _combine_pure (uncounted)"
```

---

## Phase V — Verify (golden regen, re-pilot, gate)

### Task V.1: Regenerate the golden + headless digest (core-lead, once)

**Files:**
- Modify: `tests/golden_trajectory.json`, `tests/test_headless.py` (digest expectations)

- [ ] **Step 1: Confirm the only red tests are the intended determinism ones** (golden + headless), not logic bugs:
Run: `bash scripts/check.sh` and read the failures.
- [ ] **Step 2: Regenerate the golden** with the Global-Constraints command.
- [ ] **Step 3: Update `tests/test_headless.py`** digest expectations only if the change touched initial-world generation — invention changes likely do NOT shift the *initial* digest (it samples biomes + initial agent genes, computed before any tick), so verify the headless test is actually red before editing it.
- [ ] **Step 4: Full suite green.**
Run: `bash scripts/check.sh` → exit 0.
- [ ] **Step 5: Commit**
```bash
git add tests/golden_trajectory.json tests/test_headless.py
git commit -m "core: regenerate golden trajectory after C->B->D behaviour change (intentional)"
```

### Task V.2: Re-pilot + gate on DV2

**Files:**
- Use: `research/run_pilot.py`, `research/analyze_gate.py` (no code change unless a flag is missing)

- [ ] **Step 1: Smoke** (1 seed × 200 ticks) both arms; confirm exports carry real `discovered_by`/`tick`/`adopters` and `analyze_gate --compare-dvs` computes DV1/DV3 (no longer blocked).
- [ ] **Step 2: Re-pilot** at the locked regime (GPU PC, CPU):
```bash
CUDA_VISIBLE_DEVICES=-1 ../venv/bin/python -m artificial_society.research.run_pilot --ticks 1500 --workers 8
```
- [ ] **Step 3: Gate** on the validated DV2 + the full compare:
```bash
../venv/bin/python -m artificial_society.research.analyze_gate --outdir <out> --dv accumulated_useful_depth
../venv/bin/python -m artificial_society.research.analyze_gate --outdir <out> --compare-dvs
```
- [ ] **Step 4: Record the verdict** vs the pre-registered rule (target: move discovery-matched DV2 from −21.75 toward > 0 → BORDERLINE_LEAN_A → PATH_A). Write `docs/research/retrofit-repilot-<date>.md` with the paired-diff CIs, `discoveries-per-attempt`, and `knockout required_frac` (should rise if the ratchet works).
- [ ] **Step 5: Ablations** (design §3.7): full / value-blind / low-fidelity / teaching-off — only if Step 4 is ≥ BORDERLINE.

---

## Optional Phase E — Payoff-biased transmission (stretch; only after a non-negative re-pilot)

Per design §E (subordinate, only meaningful after A+D land and `uses` is clean): store `output_value` on `CausalMemory.record`; payoff/prestige-weight `sample_for_transmission`; replace the global `FIDELITY_BASE=0.72` resample with local 12-dim nearest-neighbour perturbation (fidelity above the Lewis-Laland threshold, < 1.0); payoff-rank vertical inheritance. Defer until V.2 shows learned ≥ random per-attempt; otherwise this amplifies a ratchet that does not yet exist. (Requires a separate exploration of `agents/culture.py` / social-learning APIs — not mapped in this plan.)

---

## Honest risk register (carried from the design)

- **The flip is not guaranteed.** Even perfectly executed, C→B→D may not lift discovery-matched DV2 (−21.75) above 0; first target is BORDERLINE_LEAN_A.
- **Compute-match self-destruct** if imagined evals leak into `combine_vectors` — D.1's distinct `_combine_pure` object + D.2's `cc.n == 0` test guard this.
- **PPO credit** for "when to invent" is false without B.2's separate head — do not ship B.3 without B.2.
- **Goodhart / diversity collapse** from a greedy need-maximizer — D.2's VoI term mitigates; watch `n_functional_clusters` in the re-pilot.
- **Ratchet saturation** under [0,1] clipping — C keeps `marginal` continuous/unbounded in count via `material_reward`'s own shape.
- **Golden churn** — every behaviour task is committed with `[golden RED: intentional]`; the golden is regenerated exactly once in V.1 by core-lead.
