# Archive — abandoned research apparatus

This folder preserves the **open-ended-innovation research effort** (2026-06),
which was abandoned in favor of developing the simulation itself. The living
simulation (world, agents, brains, all society systems) is unaffected — nothing
in here is imported by the active codebase.

Paths mirror where the files originally lived, so provenance is obvious:

| Archived | Was | What it is |
|---|---|---|
| `artificial_society/research/` | package in the source tree | pilot runners, functional-depth metrics, random-recombiner null model, gate analysis |
| `artificial_society/systems/causal_model.py` | flag-gated system (`AS_CAUSAL_MODEL`) | causal-model research prototype (never on by default) |
| `artificial_society/systems/reproduction_metrics.py` | unused module | dead debug tooling (no imports anywhere) |
| `docs/research/` | docs tree | experiment plan, umwelt-design docs, pilot data + figures, go/no-go verdicts |
| `docs/superpowers/specs/2026-06-28-…` | spec | the original research design spec |
| `tests/` | pytest tree | research-only tests (smoke, metrics, path-A; flag-gated) |

## Outcome, in one paragraph

The confound-controlled pilot (2026-06-29) found the learned system produced
*less* functional depth than a random-recombination null (22.9 vs 32.3, paired
diff −9.4 [−11.3, −7.2]). Path B was explored, Path A re-opened via the
umwelt-design directive, and the effort was then abandoned (2026-07-02). The
useful takeaway for future work: the binding constraint was the **learning
machinery and generation–policy coupling**, not the number of world mechanics.

## Full history

- Branch tip with final WIP: tag `archive/wip-causal-proto` (= last state of
  `feat/research-causal-model-proto`, incl. the causal-model wiring into
  `invention.py` that was never merged)
- Deleted research branches are preserved as `archive/branch-<name>` tags

Nothing here is collected by pytest (`testpaths = ["tests"]`) or packaged by
setuptools (`include = ["artificial_society*"]` from the repo root).
