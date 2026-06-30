"""Path-A A/B pilot — learned-OFF vs learned-ON (M1) vs embodiment-matched null.

The E0 result showed the disembodied recombiner is an unfair null (it confounds the
−9.4 deficit with embodiment). This driver evaluates M1 (policy-coupled selective
generation, ``AS_PATHA_M1``) against the FAIR baseline ``run_embodied_recombiner``
(population-fragmented, imperfect-sharing) at the locked regime, on the non-floored
depth-weighted graded metric (the discriminating gate DV) plus meanFD / DV2 / maxFD.

Per seed: learned-OFF and learned-ON each run in an isolated subprocess (the global
``DISCOVERY_REGISTRY`` singleton must not bleed between runs); the embodied null runs
in-process (it keeps its own registry) and is compute-matched to the ON arm's measured
attempt count. The fair-null gate compares ON vs the embodied null with paired
bootstrap CIs (reusing ``analyze_gate._bootstrap_mean_ci`` / the pre-registered rule).

Usage (locked regime, GPU-PC via SSH):

    python -m artificial_society.research.run_pathA_pilot \
        --seeds 1001 1002 1003 1004 --ticks 1500 --grid-w 30 --grid-h 20 --pop 24 \
        --outdir docs/research/umwelt-design/pilot_m1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from artificial_society.research import metrics
from artificial_society.research.analyze_gate import _bootstrap_mean_ci
from artificial_society.research.export import load_run
from artificial_society.research.recombiner import run_embodied_recombiner

DV_KEYS = ("gdw", "meanfd", "dv2", "maxfd")


def _subproc_env(m1: bool) -> dict:
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "CUDA_VISIBLE_DEVICES": "-1",  # GPU FP16 path is bugged; CPU is faster for tiny nets
    }
    if m1:
        env["AS_PATHA_M1"] = "1"
    else:
        env.pop("AS_PATHA_M1", None)
    return env


def _run_learned(seed, ticks, grid_w, grid_h, pop, out, m1):
    r = subprocess.run(
        [
            sys.executable, "-m", "artificial_society.research.run_single",
            "--arm", "learned", "--seed", str(seed), "--ticks", str(ticks),
            "--grid-w", str(grid_w), "--grid-h", str(grid_h), "--pop", str(pop),
            "--out", out,
        ],
        env=_subproc_env(m1), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"learned(m1={m1}) seed {seed} failed: {r.stderr[-500:]}")
    return load_run(out)


def _dvs(entries) -> dict:
    if not entries:
        return {"maxfd": 0, "meanfd": 0.0, "dv2": 0, "gdw": 0.0, "n": 0}
    struct = metrics.structural_depths(entries)
    fd = np.array(list(metrics.functional_depths(entries, struct).values()), dtype=float)
    return {
        "n": len(entries),
        "maxfd": int(fd.max()),
        "meanfd": float(fd.mean()),
        "dv2": int(metrics.accumulated_useful_depth(entries, struct=struct)["accumulated_useful_depth"]),
        "gdw": float(metrics.graded_useful_advance(entries, struct=struct, depth_weight=True)["graded_useful_advance"]),
    }


def _gate(a_vals, b_vals) -> dict:
    """Pre-registered gate on a paired DV: arm A (learned-ON) vs arm B (fair null)."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    am, alo, ahi = _bootstrap_mean_ci(a, seed=1)
    bm, blo, bhi = _bootstrap_mean_ci(b, seed=2)
    dm, dlo, dhi = _bootstrap_mean_ci(a - b, seed=3)
    separated = alo > bhi
    positive = dlo > 0
    verdict = "PATH_A" if (separated and positive) else ("BORDERLINE_LEAN_A" if positive else "PATH_B_OR_RETROFIT")
    return {
        "verdict": verdict,
        "on": [am, alo, ahi], "null": [bm, blo, bhi], "diff": [dm, dlo, dhi],
        "ci_separated": bool(separated), "paired_ci_positive": bool(positive),
    }


def run(seeds, ticks, grid_w, grid_h, pop, share_fidelity, outdir):
    os.makedirs(outdir, exist_ok=True)
    print(f"[pathA] {len(seeds)} seeds x {ticks} ticks | grid {grid_w}x{grid_h} pop {pop} "
          f"| null=embodied(P={pop}, f={share_fidelity}) -> {outdir}", flush=True)
    rows = {"OFF": [], "ON": [], "NULL": []}
    for s in seeds:
        off = _dvs(_run_learned(s, ticks, grid_w, grid_h, pop, os.path.join(outdir, f"off_{s}.json"), m1=False)["entries"])
        on_run = _run_learned(s, ticks, grid_w, grid_h, pop, os.path.join(outdir, f"on_{s}.json"), m1=True)
        on = _dvs(on_run["entries"])
        att = max(1, int(on_run["meta"]["n_attempts"]))  # compute-match the null to the ON arm
        null_entries, _ = run_embodied_recombiner(seed=s, n_attempts=att, n_agents=pop, share_fidelity=share_fidelity)
        null = _dvs(null_entries)
        for k, d in (("OFF", off), ("ON", on), ("NULL", null)):
            rows[k].append(d)
        print(f"[pathA] seed {s} (null att={att}): "
              f"OFF gdw={off['gdw']:.0f}/meanFD={off['meanfd']:.2f} | "
              f"ON gdw={on['gdw']:.0f}/meanFD={on['meanfd']:.2f} | "
              f"NULL gdw={null['gdw']:.0f}/meanFD={null['meanfd']:.2f}", flush=True)

    gates = {k: _gate([r[k] for r in rows["ON"]], [r[k] for r in rows["NULL"]]) for k in DV_KEYS}
    summary = {"seeds": seeds, "ticks": ticks, "pop": pop, "share_fidelity": share_fidelity,
               "rows": rows, "fair_null_gate_ON_vs_NULL": gates}
    with open(os.path.join(outdir, "pathA_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== FAIR-NULL GATE — learned-ON (M1) vs embodiment-matched null ===")
    for k in DV_KEYS:
        g = gates[k]
        print(f"  {k:7s}: ON={g['on'][0]:.2f}[{g['on'][1]:.2f},{g['on'][2]:.2f}] "
              f"NULL={g['null'][0]:.2f}[{g['null'][1]:.2f},{g['null'][2]:.2f}] "
              f"diff={g['diff'][0]:.2f}[{g['diff'][1]:.2f},{g['diff'][2]:.2f}] -> {g['verdict']}")
    print(f"\n[pathA] summary -> {os.path.join(outdir, 'pathA_summary.json')}", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Path-A pilot: M1 learned vs embodiment-matched null.")
    p.add_argument("--seeds", type=int, nargs="*", default=[1001, 1002, 1003, 1004])
    p.add_argument("--ticks", type=int, default=1500)
    p.add_argument("--grid-w", type=int, default=30)
    p.add_argument("--grid-h", type=int, default=20)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--share-fidelity", type=float, default=0.72)
    p.add_argument("--outdir", type=str, required=True)
    args = p.parse_args(argv)
    run(args.seeds, args.ticks, args.grid_w, args.grid_h, args.pop, args.share_fidelity, args.outdir)


if __name__ == "__main__":
    main()
