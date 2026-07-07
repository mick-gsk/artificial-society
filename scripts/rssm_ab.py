#!/usr/bin/env python
"""3-arm RSSM A/B runner + RMST analysis (spec §10.2). Run on the GPU-PC.

run:     venv/bin/python scripts/rssm_ab.py run --arm B --seed 1 --ticks 20000 --out runs/ab1
analyze: venv/bin/python scripts/rssm_ab.py analyze runs/ab1 --tau 2000 --transient 1500
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def rmst(lifespans, censored, tau):
    """Kaplan-Meier restricted mean survival time to horizon tau."""
    events = sorted(zip(lifespans, censored))
    n, surv, prev_t, area = len(events), 1.0, 0.0, 0.0
    at_risk = n
    for t, cens in events:
        t_c = min(t, tau)
        area += surv * (t_c - prev_t)
        prev_t = t_c
        if t >= tau:
            break
        if not cens:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
    area += surv * max(0.0, tau - prev_t)
    return area


def wilcoxon_exact(deltas):
    """Exact two-sided signed-rank p for N<=12 (enumeration). Zero deltas dropped."""
    import itertools

    d = [x for x in deltas if x != 0]
    n = len(d)
    # Sequential (not midrank) tie ranks are intentional: this ranking is only
    # used internally to build the enumerated reference distribution `ws` below,
    # which is generated with the SAME sequential scheme, so it stays internally
    # consistent. RMST deltas are continuous-valued, so exact ties are practically
    # impossible anyway.
    ranks = {
        i: r + 1 for r, (i, _) in enumerate(sorted(enumerate(map(abs, d)), key=lambda kv: kv[1]))
    }
    w_obs = sum(ranks[i] for i, x in enumerate(d) if x > 0)
    total = list(ranks.values())
    ws = [sum(c) for k in range(n + 1) for c in itertools.combinations(total, k)]
    mean_w = sum(total) / 2
    extreme = sum(1 for w in ws if abs(w - mean_w) >= abs(w_obs - mean_w))
    return min(1.0, extreme / len(ws))


def _cast_like(default, raw):
    """Cast a CLI string `raw` to the runtime type of `default` (int/float/else str)."""
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def parse_cfg_overrides(pairs):
    """Parse repeated `--cfg key=value` into a dict, typed off RSSMConfig's own fields."""
    from artificial_society.agents.rssm.config import RSSMConfig

    template = RSSMConfig()
    valid = {f.name for f in dataclasses.fields(RSSMConfig)}
    overrides = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--cfg expects key=value, got {pair!r}")
        if key not in valid:
            raise ValueError(f"unknown RSSMConfig field: {key!r}")
        overrides[key] = _cast_like(getattr(template, key), raw)
    return overrides


def parse_v1_consts(pairs):
    """Parse repeated `--v1-const KEY=value` into a dict, typed off the current module attr."""
    import artificial_society.agents.brain as brain_mod

    consts = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--v1-const expects KEY=value, got {pair!r}")
        if not hasattr(brain_mod, key):
            raise ValueError(f"unknown artificial_society.agents.brain constant: {key!r}")
        consts[key] = _cast_like(getattr(brain_mod, key), raw)
    return consts


def _arm_sim(arm, seed, cpu, cfg_overrides=None, v1_consts=None):
    from artificial_society.agents.rssm.config import RSSMConfig
    from artificial_society.simulation import Simulation

    kw = dict(
        headless=True,
        load_checkpoint=False,
        seed=seed,
        grid_w=60,
        grid_h=40,
        initial_population=36,
    )
    if arm == "A":
        if cfg_overrides:
            raise ValueError("--cfg overrides only apply to arms B/C, not arm A")
        if v1_consts:
            # Runtime-patch module constants BEFORE Simulation construction — brain.py
            # reads these as module globals inside function bodies at call time (not
            # baked into defaults at import), so setattr here takes effect for the
            # whole run. Exception: PLAN_CANDIDATES, which is only read as a bound
            # default argument value (`plan_action(..., n_candidates=PLAN_CANDIDATES)`)
            # and the sole call site never passes n_candidates explicitly — patching
            # it here has NO effect on the actual planning call.
            import artificial_society.agents.brain as brain_mod

            for key, val in v1_consts.items():
                setattr(brain_mod, key, val)
        # Pin explicitly (review fix, Important 5): a bare Simulation(**kw) falls
        # back to AS_BRAIN_ARCH from the environment, so an ambient env var left
        # set from a prior rssm shell session would silently promote arm A to
        # the rssm substrate — pin v1 unconditionally.
        return Simulation(brain_arch="v1", **kw)
    if v1_consts:
        raise ValueError("--v1-const overrides only apply to arm A, not arms B/C")
    dev = "cpu" if cpu else "cuda"
    if not cpu:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") != "-1", (
            "unset CUDA_VISIBLE_DEVICES=-1 for arms B/C (spec §8)"
        )
        import torch

        assert torch.cuda.is_available(), "arms B/C need CUDA (or pass --cpu for smoke only)"
    cfg = dataclasses.replace(
        RSSMConfig(), train_device=dev, policy_mode=("actor" if arm == "B" else "mpc")
    )
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)
    return Simulation(brain_arch="rssm", rssm_config=cfg, **kw)


def cmd_run(a):
    # Harness runs never autosave (review fix, Important 4): CHECKPOINT_INTERVAL
    # is a module-level global read fresh at tick time (simulation.py::step), so
    # patching it before construction disables autosaves for the whole run —
    # concurrent A/B/C runs would otherwise race on the shared CHECKPOINT_PATH.
    import artificial_society.simulation as sim_mod

    sim_mod.CHECKPOINT_INTERVAL = 0
    cfg_overrides = parse_cfg_overrides(getattr(a, "cfg", None))
    v1_consts = parse_v1_consts(getattr(a, "v1_const", None))
    sim = _arm_sim(a.arm, a.seed, a.cpu, cfg_overrides=cfg_overrides, v1_consts=v1_consts)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = getattr(a, "tag", None)
    stub = f"{a.arm}_{tag}_s{a.seed}" if tag else f"{a.arm}_s{a.seed}"
    known = {}  # id -> (birth_tick, origin)
    agent_ticks = 0  # cumulative sum(len(sim.agents)) — arm A's transitions-axis proxy

    def scan(ev, tick):
        ids = set()
        for ag in sim.agents:
            ids.add(ag.id)
            if ag.id not in known:
                known[ag.id] = (
                    getattr(ag, "birth_tick", tick),
                    getattr(ag, "spawn_origin", "initial"),
                )
                ev.write(
                    json.dumps(
                        {"e": "birth", "id": ag.id, "t": known[ag.id][0], "origin": known[ag.id][1]}
                    )
                    + "\n"
                )
        for aid in [k for k in known if k not in ids and known[k] is not None]:
            ev.write(json.dumps({"e": "death", "id": aid, "t": tick}) + "\n")
            known[aid] = None

    with open(out / f"{stub}.events.jsonl", "w") as ev:
        scan(ev, 0)
        for t in range(1, a.ticks + 1):
            sim.step()
            agent_ticks += len(sim.agents)  # all arms, every tick (review fix, Important 3)
            scan(ev, t)
            if t % 500 == 0:
                wm_u = sim.rssm_learner.wm.wm_updates if sim.rssm_learner else 0
                ev.write(
                    json.dumps(
                        {
                            "e": "log",
                            "t": t,
                            "alive": len(sim.agents),
                            # transitions-matched primary axis (spec §10.2, review
                            # Important-3): cumulative stored transitions for B/C
                            # (monotone learner counter, unlike len(replay) which
                            # shrinks on FIFO eviction); cumulative agent-ticks for
                            # A (no learner, no per-transition counter to read).
                            "transitions_stored": sim.rssm_learner.transitions_stored
                            if sim.rssm_learner
                            else None,
                            "agent_ticks": agent_ticks,
                            "wm_updates": wm_u,
                        }
                    )
                    + "\n"
                )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    import torch

    # Provenance must detect RSSMConfig drift (review fix, Triage 8): hash the
    # actual resolved RSSMConfig for rssm arms (B/C) alongside the CLI args — a
    # bare args-hash can't tell two runs with different hyperparameters apart.
    # Arm A has no RSSMConfig at all; a stable marker keeps the hash's shape
    # (and its "did the config change" semantics) consistent across arms. The
    # v1-const overrides are folded into arm A's marker so a tuning round on
    # arm A also invalidates the hash (cfg_overrides for B/C are already baked
    # into sim.rssm_learner.cfg by _arm_sim, so they need no separate entry).
    cfg_dict = (
        dataclasses.asdict(sim.rssm_learner.cfg)
        if sim.rssm_learner is not None
        else {"brain_arch": "v1", "v1_consts": v1_consts or {}}
    )
    cfg_hash = hashlib.sha1(
        repr((sorted(vars(a).items()), sorted(cfg_dict.items()))).encode()
    ).hexdigest()[:12]
    (out / f"{stub}.summary.json").write_text(
        json.dumps(
            {
                "arm": a.arm,
                "tag": tag,
                "seed": a.seed,
                "ticks": a.ticks,
                "git": sha,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cfg_overrides": cfg_overrides or {},
                "v1_consts": v1_consts or {},
                "cfg_hash": cfg_hash,
            }
        )
    )


def cmd_analyze(a):
    runs = {}
    for f in pathlib.Path(a.dir).glob("*_s*.events.jsonl"):
        # Filename is `<arm>_s<seed>` (no tag, backward compat) or
        # `<arm>_<tag>_s<seed>` (tagged run). rsplit on the LAST "_s" in both
        # cases yields (arm[_tag], seed); the arm-or-arm_tag half becomes the
        # "arm" key below, so a tagged variant renders as its own row (e.g.
        # "B_k4") instead of collapsing into plain "B".
        arm, seed = f.stem.split(".")[0].rsplit("_s", 1)
        births, deaths, last_t = {}, {}, 0
        for line in f.open():
            r = json.loads(line)
            last_t = max(last_t, r.get("t", 0))
            if r["e"] == "birth":
                births[r["id"]] = r
            elif r["e"] == "death":
                deaths[r["id"]] = r["t"]
        # Censoring time = the run's true --ticks, not the max observed event
        # timestamp (a run can end with no events near the tail, understating
        # survival for still-alive agents). Fall back to max-event-timestamp
        # only if the sibling summary.json is missing.
        summary_path = f.parent / f"{arm}_s{seed}.summary.json"
        if summary_path.exists():
            last_t = json.loads(summary_path.read_text())["ticks"]
        cohort = [
            (i, b) for i, b in births.items() if b["origin"] == "birth" and b["t"] >= a.transient
        ]
        lifespans = [(deaths.get(i, last_t) - b["t"]) for i, b in cohort]
        censored = [i not in deaths for i, _ in cohort]
        if lifespans:
            runs.setdefault(arm, {})[int(seed)] = rmst(lifespans, censored, a.tau)
    for arm in sorted(runs):
        vals = list(runs[arm].values())
        print(
            f"arm {arm}: n={len(vals)} RMST median={sorted(vals)[len(vals) // 2]:.1f} "
            f"values={[f'{v:.1f}' for v in vals]}"
        )
    for x, y in (("B", "A"), ("B", "C")):
        if x in runs and y in runs:
            seeds = sorted(set(runs[x]) & set(runs[y]))
            deltas = [runs[x][s] - runs[y][s] for s in seeds]
            if deltas:
                print(
                    f"{x}-vs-{y}: paired deltas={[f'{d:.1f}' for d in deltas]} "
                    f"wilcoxon_p={wilcoxon_exact(deltas):.4f}"
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["A", "B", "C"], required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--ticks", type=int, default=20000)
    r.add_argument("--out", required=True)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--tag", default=None, help="output files become <arm>_<tag>_s<seed>.*")
    r.add_argument(
        "--cfg",
        action="append",
        default=[],
        metavar="key=value",
        help="RSSMConfig field override (repeatable, arms B/C only)",
    )
    r.add_argument(
        "--v1-const",
        dest="v1_const",
        action="append",
        default=[],
        metavar="KEY=value",
        help="artificial_society.agents.brain module constant override (repeatable, arm A only)",
    )
    an = sub.add_parser("analyze")
    an.add_argument("dir")
    an.add_argument("--tau", type=int, default=2000)
    an.add_argument("--transient", type=int, default=1500)
    args = p.parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)
