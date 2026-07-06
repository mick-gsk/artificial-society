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
    ranks = {
        i: r + 1 for r, (i, _) in enumerate(sorted(enumerate(map(abs, d)), key=lambda kv: kv[1]))
    }
    w_obs = sum(ranks[i] for i, x in enumerate(d) if x > 0)
    total = list(ranks.values())
    ws = [sum(c) for k in range(n + 1) for c in itertools.combinations(total, k)]
    mean_w = sum(total) / 2
    extreme = sum(1 for w in ws if abs(w - mean_w) >= abs(w_obs - mean_w))
    return min(1.0, extreme / len(ws))


def _arm_sim(arm, seed, cpu):
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
        return Simulation(**kw)
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
    return Simulation(brain_arch="rssm", rssm_config=cfg, **kw)


def cmd_run(a):
    sim = _arm_sim(a.arm, a.seed, a.cpu)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    known = {}  # id -> (birth_tick, origin)

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

    with open(out / f"{a.arm}_s{a.seed}.events.jsonl", "w") as ev:
        scan(ev, 0)
        for t in range(1, a.ticks + 1):
            sim.step()
            scan(ev, t)
            if t % 500 == 0:
                wm_u = sim.rssm_learner.wm.wm_updates if sim.rssm_learner else 0
                ev.write(
                    json.dumps(
                        {
                            "e": "log",
                            "t": t,
                            "alive": len(sim.agents),
                            "transitions": len(sim.rssm_learner.replay)
                            if sim.rssm_learner
                            else t * len(sim.agents),
                            "wm_updates": wm_u,
                        }
                    )
                    + "\n"
                )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    import torch

    (out / f"{a.arm}_s{a.seed}.summary.json").write_text(
        json.dumps(
            {
                "arm": a.arm,
                "seed": a.seed,
                "ticks": a.ticks,
                "git": sha,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cfg_hash": hashlib.sha1(repr(sorted(vars(a).items())).encode()).hexdigest()[:12],
            }
        )
    )


def cmd_analyze(a):
    runs = {}
    for f in pathlib.Path(a.dir).glob("*_s*.events.jsonl"):
        arm, seed = f.stem.split(".")[0].split("_s")
        births, deaths, last_t = {}, {}, 0
        for line in f.open():
            r = json.loads(line)
            last_t = max(last_t, r.get("t", 0))
            if r["e"] == "birth":
                births[r["id"]] = r
            elif r["e"] == "death":
                deaths[r["id"]] = r["t"]
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
    an = sub.add_parser("analyze")
    an.add_argument("dir")
    an.add_argument("--tau", type=int, default=2000)
    an.add_argument("--transient", type=int, default=1500)
    args = p.parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)
