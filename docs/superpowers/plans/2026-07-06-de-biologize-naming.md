# De-Biologize Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename biology-suggesting identifiers in `artificial-society` to neutral agent-/systems vocabulary without changing any simulation behaviour.

**Architecture:** A small token-accurate rewrite tool (`scripts/debio_rename.py`) renames Python *identifier word-parts* per a concept map, leaving strings and comments untouched — so `genes`/`GeneStore` are renamed while `generation`/`genesis`/`eigener` are not. The rename runs one concept-cluster at a time, each gated by the full test suite + the numeric golden trajectory, committed atomically. Coupled string literals (e.g. `getattr(obj, "cortisol")`, action-mode `'mate'`) are handled inside the cluster that renames their bound identifier, with a frame-schema diff proving the wire contract is unchanged. Module files are renamed last, with import + manifest fixups.

**Tech Stack:** Python 3, `tokenize`, `pytest`, `git worktree`.

## Global Constraints

- **Behaviour invariance is the top constraint.** No control-flow, numeric, or algorithmic change. The golden trajectory `tests/golden_trajectory.json` (`[tick, count, Σenergy, Σhealth]`, numeric, no field names) MUST stay byte-identical after every task. The determinism suite (`tests/test_regression_golden.py` + the `tests/test_headless.py` state digest) is part of the full run — never edit a determinism test to make it pass; a red digest means behaviour changed.
- **Interpreter:** Python 3.9.6 at `../venv/bin/python` (relative to the worktree root; verified to hold numpy/torch/fastapi/pytest). Every `python`/`pytest`/`ruff` in this plan runs through it. Do NOT use a bare `python`.
- **Lint gate (per task, before commit):** the repo's auto-format PostToolUse hook does NOT fire on files the rename tool writes directly, but CI lints changed Python files. So after each cluster, run on the changed files:
  ```bash
  CH=$(git diff --name-only -- '*.py'); [ -n "$CH" ] && ../venv/bin/python -m ruff check --fix $CH && ../venv/bin/python -m ruff format $CH
  ```
  A rename can push a line past the length limit (e.g. `photosynthesis`→`resource_regrowth`); `ruff format` rewraps it (behaviour-neutral).
- **Kept terms — never rename** (physics-legit / game-standard / borderline-not-chosen): `energy, health, age, grow, spawn, die, death, alive, brain, body, food, eat, plant, plant_food, selection`. Also out of scope: `biome`.
- **Known kept wire/field strings** (string literals the token tool leaves untouched — must STAY, and the leak audit treats them as intentional): the `world.F` field key `"disease"`, `"plant_food"`; the `/api/status` stats keys emitted as literals in `visualization/statistics.py` (`"avg_sick"`, `"world_disease"`, `"avg_children"`, `"avg_plant"`, `"avg_meat"`, …). Their *values* come from renamed attributes, but the *keys* are literals → wire stays byte-stable automatically.
- **External contracts stay byte-stable:** the `/api/status` + `/api/history` JSON key-shape (dashboard reads it), config/manifest keys, CLI flags. Depth is internal identifiers + comments/docstrings only. (The rich live-viz websocket frame lives on `feat/infra-live-viz-v2`, not on `main` — out of scope here.)
- **Hot/frozen files WILL be edited.** Per `CLAUDE.md`, `agents/agent.py` (46 bio-term lines), `simulation.py` (10), `world.py` (3), `agents/brain.py` (4) are frozen-contract files normally routed through `core-lead`. This rename is a deliberate **cross-lane global refactor** and touches them by necessity — it is a core-lead-level change, not a domain-lane edit. Coordinate before merging (other actor branches: `experiment/rssm-dreamer-brain`, `feat/infra-m1-pilot`, `feat/infra-live-viz-v2` will need to rebase).
- **Old `.pkl` checkpoints may break** (regenerated). No `__setstate__` shim. Delete any stray root `checkpoint.pkl` in the worktree before headless runs (it auto-loads; `Simulation(load_checkpoint=False)` — as the golden uses — ignores it, but keep the tree clean).
- **File scope for renames:** tracked Python files excluding `archive/`. Canonical file list command (used throughout):
  ```bash
  git ls-files '*.py' | grep -v '^archive/'
  ```
- **Working dir:** `/Users/moritzbecker/projekt/as-debio`, branch `chore/de-biologize-naming` (off `main @ 773577b`). All commands run from the repo root there.
- **Per-task gate (applies to every rename task):** lint gate (above) → `../venv/bin/python -m pytest -q` (all pass) → golden `../venv/bin/python -m pytest tests/test_regression_golden.py -q` (pass). Only then commit.
- **Spec:** `docs/superpowers/specs/2026-07-06-de-biologize-naming-design.md` (frozen mapping table).

---

### Task 0: Establish green baseline & capture wire/golden fingerprints

**Files:**
- Create (scratch, git-ignored): `.debio/` working dir for maps + baseline snapshots.

**Interfaces:**
- Produces: `.debio/wire_keys_baseline.txt` (wire-key fingerprint), confirmed green suite — consumed by every later verification step.

- [ ] **Step 1: Confirm clean worktree on the right branch**

Run:
```bash
git -C /Users/moritzbecker/projekt/as-debio rev-parse --abbrev-ref HEAD
git -C /Users/moritzbecker/projekt/as-debio status -sb
```
Expected: `chore/de-biologize-naming`, clean (only the committed spec + this plan).

- [ ] **Step 2: Create the scratch working dir and git-ignore it**

Run:
```bash
mkdir -p .debio
grep -qxF '.debio/' .gitignore 2>/dev/null || echo '.debio/' >> .gitignore
```

- [ ] **Step 3: Run the full suite to confirm a green baseline**

Run: `../venv/bin/python -m pytest -q`
Expected: all tests pass (record the count, e.g. `NNN passed`). If anything fails on a fresh `main`, STOP and report — the baseline must be green before any rename.

- [ ] **Step 4: Capture the wire-key fingerprint (the external contract)**

The dashboard reads the `/api/status` + `/api/history` JSON, whose keys are **string literals** in `visualization/statistics.py`; the `world.F` field keys are string literals in `cell_store.py`. The rename tool never touches string literals, so these must remain byte-identical — this is the cheap, exact guard the later tasks diff against. Capture the baseline:
```bash
{ grep -hoE '"[a-z_]+"' \
    artificial_society/visualization/statistics.py \
    artificial_society/cell_store.py \
    artificial_society/serve/app.py \
    artificial_society/serve/runner.py 2>/dev/null; } | sort -u > .debio/wire_keys_baseline.txt
wc -l .debio/wire_keys_baseline.txt
grep -iE 'sick|disease|children|plant|meat' .debio/wire_keys_baseline.txt
```
Expected: a sorted list including `"avg_sick"`, `"world_disease"`, `"avg_children"`, `"plant_food"`, `"disease"`. This file is git-ignored; the "wire gate" in later tasks re-runs this grep into `.debio/wire_keys_after.txt` and asserts an empty diff.

- [ ] **Step 5: Confirm the checkpoint round-trip works in-session**

The checkpoint API is `Simulation._save_checkpoint()` (writes `CHECKPOINT_PATH` = `checkpoint.pkl`) and `_load_checkpoint()` (auto-called when constructed with `load_checkpoint=True` and the file exists). Verify a fresh save+load works (old pre-rename `.pkl` is intentionally not tested):
```bash
rm -f checkpoint.pkl
../venv/bin/python - <<'PY'
from artificial_society.simulation import Simulation
s = Simulation(headless=True, load_checkpoint=False, seed=1, grid_w=20, grid_h=12, initial_population=8)
for _ in range(5): s.step()
s._save_checkpoint()
s2 = Simulation(headless=True, load_checkpoint=True, seed=1, grid_w=20, grid_h=12, initial_population=8)
print("round-trip OK")
PY
rm -f checkpoint.pkl
```
Expected: `round-trip OK`. No commit.

---

### Task 1: Build & TDD the token-accurate rename tool

**Files:**
- Create: `scripts/debio_rename.py`
- Create: `tests/tools/__init__.py` (empty)
- Test: `tests/tools/test_debio_rename.py`

**Interfaces:**
- Produces:
  - `rename_identifier(name: str, mapping: dict[str, str]) -> str` — rewrites one identifier's word-parts.
  - `rewrite_source(src: str, mapping: dict[str, str]) -> tuple[str, dict[str, int]]` — rewrites NAME tokens in a source string; returns `(new_src, {"old->new": count})`.
  - CLI: `../venv/bin/python scripts/debio_rename.py --map MAP.json [--apply] FILE...` (dry-run prints a change summary; `--apply` writes in place). `mapping` keys are lowercase word-parts; values are replacement strings (may contain underscores).

- [ ] **Step 1: Write the failing test**

Create `tests/tools/__init__.py` (empty) and `tests/tools/test_debio_rename.py`:
```python
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from debio_rename import rename_identifier, rewrite_source  # noqa: E402

import pytest  # noqa: E402

MAP = {
    "gene": "trait", "genes": "traits", "genetic": "trait_based",
    "genetics": "traits", "inherit": "derive", "disease": "fault",
    "endocrine": "modulation",
}


@pytest.mark.parametrize("name,expected", [
    # positive: word-parts that must be renamed
    ("genes", "traits"),
    ("gene", "trait"),
    ("gene_pool", "trait_pool"),
    ("inherit_genes", "derive_traits"),
    ("GeneStore", "TraitStore"),
    ("DISEASE_DECAY", "FAULT_DECAY"),
    ("EndocrineSystem", "ModulationSystem"),
    ("__gene__", "__trait__"),
    # negative: substrings that must be LEFT ALONE
    ("generation", "generation"),
    ("genesis", "genesis"),
    ("generate", "generate"),
    ("Regeneration", "Regeneration"),
    ("eigener", "eigener"),
    ("estimate", "estimate"),
])
def test_rename_identifier(name, expected):
    assert rename_identifier(name, MAP) == expected


def test_strings_and_comments_untouched():
    src = 'x = "gene"  # gene comment\ny = genes\n'
    new, changes = rewrite_source(src, MAP)
    assert new == 'x = "gene"  # gene comment\ny = traits\n'
    assert changes == {"genes->traits": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `../venv/bin/python -m pytest tests/tools/test_debio_rename.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'debio_rename'`.

- [ ] **Step 3: Write the tool**

Create `scripts/debio_rename.py`:
```python
#!/usr/bin/env python3
"""Token-accurate identifier de-biologizer (dev tool; not shipped in the package).

Rewrites Python *identifier* word-parts per a concept map, leaving string
literals and comments untouched. Word-part aware: a NAME token is split on
underscores, each segment further split on CamelCase / digit boundaries; only
parts that equal a map key (case-insensitive) are replaced, each preserving its
original case pattern. So `genes`, `inherit_genes`, `GeneStore` are renamed
while `generation`, `genesis`, `generate`, `eigener` are not.
"""
from __future__ import annotations

import argparse
import json
import re
import tokenize
from pathlib import Path

# ALLCAPS run | Titlecase word | lowercase run | digit run
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")


def _recase(replacement: str, sample: str) -> str:
    """Cast `replacement` to the case pattern of the original word `sample`."""
    if sample.isupper() and not sample.islower():
        return replacement.upper()
    if sample[:1].isupper() and sample[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement  # lowercase / mixed -> verbatim (map values are snake/lower)


def rename_identifier(name: str, mapping: dict[str, str]) -> str:
    """Return `name` with every matching word-part replaced; layout preserved."""
    segments = re.split(r"(_+)", name)  # keep underscore separators
    out: list[str] = []
    for seg in segments:
        if not seg or seg.startswith("_"):
            out.append(seg)  # separator or empty (leading/trailing/dunder)
            continue
        words = _WORD.findall(seg)
        if not words or "".join(words) != seg:
            out.append(seg)  # non-standard token: leave untouched
            continue
        rebuilt = [
            _recase(mapping[w.lower()], w) if w.lower() in mapping else w
            for w in words
        ]
        out.append("".join(rebuilt))
    return "".join(out)


def rewrite_source(src: str, mapping: dict[str, str]):
    """Rewrite NAME tokens in `src`; return (new_src, {"old->new": count})."""
    lines = src.splitlines(keepends=True)
    edits: dict[int, list[tuple[int, int, str]]] = {}
    changes: dict[str, int] = {}
    readline = iter(lines).__next__
    for tok in tokenize.generate_tokens(readline):
        if tok.type != tokenize.NAME:
            continue
        new = rename_identifier(tok.string, mapping)
        if new == tok.string:
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        if srow != erow:
            continue  # NAME tokens are single-line
        edits.setdefault(srow, []).append((scol, ecol, new))
        key = f"{tok.string}->{new}"
        changes[key] = changes.get(key, 0) + 1
    for row, row_edits in edits.items():
        line = lines[row - 1]
        for scol, ecol, new in sorted(row_edits, reverse=True):
            line = line[:scol] + new + line[ecol:]
        lines[row - 1] = line
    return "".join(lines), changes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="JSON file: {part: replacement}")
    ap.add_argument("--apply", action="store_true", help="write files (default: dry-run)")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    mapping = {k.lower(): v for k, v in json.loads(Path(args.map).read_text()).items()}
    total: dict[str, int] = {}
    for f in args.files:
        p = Path(f)
        try:
            src = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        new, changes = rewrite_source(src, mapping)
        if changes:
            for k, v in changes.items():
                total[k] = total.get(k, 0) + v
            if args.apply:
                p.write_text(new)
    for k in sorted(total):
        print(f"{total[k]:5d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `../venv/bin/python -m pytest tests/tools/test_debio_rename.py -q`
Expected: PASS (all parametrized cases + both source-level cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/debio_rename.py tests/tools/__init__.py tests/tools/test_debio_rename.py .gitignore
git commit -m "tools: add token-accurate identifier rename tool for de-biologizing"
```

---

### Cluster-rename procedure (used by Tasks 2–8)

Every cluster task follows this exact shape (only the map differs):

1. Write the cluster map JSON to `.debio/<cluster>.json`.
2. **Dry-run** to review what changes: `../venv/bin/python scripts/debio_rename.py --map .debio/<cluster>.json $(git ls-files '*.py' | grep -v '^archive/')` — read the summary, confirm no surprising `old->new` pairs (e.g. a kept term or a false hit).
3. **Apply**: same command with `--apply`.
4. **Coupled strings** (only where noted): grep for the cluster's OLD words inside `getattr/setattr/hasattr(..., "OLD")`, and dict keys / string literals that mirror a renamed attribute; classify each as internal (rename in lockstep) or wire-visible (keep); apply.
5. **Lint**: `CH=$(git diff --name-only -- '*.py'); [ -n "$CH" ] && ../venv/bin/python -m ruff check --fix $CH && ../venv/bin/python -m ruff format $CH`.
6. **Wire gate** (Task 3 Step 5) → `WIRE UNCHANGED`. **Gate**: `../venv/bin/python -m pytest -q` (all pass) and `../venv/bin/python -m pytest tests/test_regression_golden.py -q` (pass).
7. **Commit** the cluster.

---

### Task 2: Cluster — Genetics / heredity

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`.

- [ ] **Step 1: Write the map**
```bash
cat > .debio/genetics.json <<'JSON'
{"gene":"trait","genes":"traits","genetic":"trait_based","genetics":"traits",
 "genome":"trait_set","genotype":"trait_set","allele":"variant","alleles":"variants",
 "inherit":"derive","inherits":"derives","inherited":"derived","inheritance":"derivation",
 "heredity":"derived","hereditary":"derived",
 "mutation":"perturbation","mutations":"perturbations","mutate":"perturb",
 "mutates":"perturb","mutated":"perturbed","mutating":"perturbing",
 "dna":"trait_code","lineage":"origin","ancestor":"parent","ancestors":"parents"}
JSON
```

- [ ] **Step 2: Dry-run and review**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/genetics.json $(git ls-files '*.py' | grep -v '^archive/')`
Expected: pairs like `genes->traits`, `inherit_genes->derive_traits`, `random_genes->random_traits`, `child_genes->child_traits`, `Genetics->Traits`. Verify NO pair touches `generation`, `genesis`, `generate*`, `Regeneration`, or German words.

- [ ] **Step 3: Apply**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/genetics.json --apply $(git ls-files '*.py' | grep -v '^archive/')`

- [ ] **Step 4: Rename the module file + test file to match the new import paths**

The tool already rewrote the module-path token `genetics`→`traits` inside every `import` statement (a module path component is a NAME token), so imports now read `artificial_society.agents.traits`. Move the files to match, in this same commit:
```bash
git mv artificial_society/agents/genetics.py artificial_society/agents/traits.py
git mv tests/agents/test_genetics_strength.py tests/agents/test_traits_strength.py
```
Fix the manifest path (`artificial_society\\agents\\genetics.py` → `...\\traits.py`):
edit `artificial_society_manifest.json`, then confirm:
```bash
grep -n 'genetics' artificial_society_manifest.json || echo "manifest clean"
```
Expected: `manifest clean`.

- [ ] **Step 5: Check for dynamic-import strings of the old module name**
```bash
git ls-files '*.py' | grep -v archive | xargs grep -nE "import_module\(|__import__\(|['\"][a-z_.]*genetics['\"]" 2>/dev/null
```
If any string literal names the `genetics` module for a dynamic import, update it to `traits`. Expected: none.

- [ ] **Step 6: Gate**

Run: `../venv/bin/python -m pytest -q` → all pass (an ImportError here means a missed module path). Run: `../venv/bin/python -m pytest tests/test_regression_golden.py -q` → pass.

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "refactor: de-biologize genetics/heredity + rename module to traits.py"
```

---

### Task 3: Cluster — Reproduction

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`.

- [ ] **Step 1: Write the map**
```bash
cat > .debio/reproduction.json <<'JSON'
{"reproduce":"replicate","reproduces":"replicates","reproduction":"replication",
 "reproductive":"replication","mate":"partner","mates":"partners","mating":"pairing",
 "breed":"replicate","breeds":"replicates","breeding":"replication",
 "fertility":"replication_rate","fecundity":"replication_rate",
 "pregnant":"pending_spawn","pregnancy":"pending_spawn","gestation":"spawn_delay",
 "offspring":"spawn","child":"spawn","children":"spawn_count",
 "birth":"spawn","born":"spawn","embryo":"pending_spawn"}
JSON
```
(`egg` is intentionally omitted — it collides with the `berry`/food domain; leave it.)

- [ ] **Step 2: Dry-run and review**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/reproduction.json $(git ls-files '*.py' | grep -v '^archive/')`
Expected: `mate->partner`, `_last_mate_id->_last_partner_id`, `reproduction_cooldown->replication_cooldown`, `REPRODUCTION_COST->REPLICATION_COST`, `child_traits->spawn_traits` (from Task 2 output), `children->spawn_count`. **Watch for over-match on `child`/`children`** — confirm every pair reads sensibly; if `child`/`children` appears in a non-offspring context (e.g. a tree/graph node), note it and, if wrong, remove `child`/`children` from the map and re-run this task.

- [ ] **Step 3: Apply**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/reproduction.json --apply $(git ls-files '*.py' | grep -v '^archive/')`

- [ ] **Step 4: Coupled string — the `'mate'` action-mode label**

The string `'mate'` is an action-mode value (`renderer.py:25` colour-map key, `renderer.py:256` `mode == 'mate'`, and wherever `last_action_mode = 'mate'` is set). Decide keep-vs-rename by wire-visibility:
```bash
# Is the action-mode string emitted over the wire / consumed by dashboard JS?
grep -rn "last_action_mode" artificial_society/serve/ artificial_society/renderer.py
git ls-files | grep -iE '\.(js|html)$' | grep -v archive | xargs grep -ln "'mate'\|\"mate\"" 2>/dev/null
```
Decision rule:
- **If** any dashboard `.js`/`.html` switches on the literal `mate`, OR the raw mode string is placed into the frame JSON that the dashboard reads → **KEEP** the string `'mate'` (it is a wire value; out of scope). Leave all `'mate'` string literals as-is.
- **Else** (mode string is server-internal, only converted to a colour) → rename the coupled literals in lockstep: in `renderer.py` change the colour-map key `'mate'` and the `mode == 'mate'` comparison to `'partner'`, and change every `last_action_mode = 'mate'` assignment to `'partner'`. Grep to find them all:
  ```bash
  git ls-files '*.py' | grep -v archive | xargs grep -n "'mate'\|\"mate\"" 2>/dev/null
  ```
  Edit each occurrence from `mate` to `partner` (string value only).

- [ ] **Step 5: Wire gate** (the reusable check referenced by later tasks)

Re-capture the wire keys and diff against the Task 0 baseline:
```bash
{ grep -hoE '"[a-z_]+"' \
    artificial_society/visualization/statistics.py \
    artificial_society/cell_store.py \
    artificial_society/serve/app.py \
    artificial_society/serve/runner.py 2>/dev/null; } | sort -u > .debio/wire_keys_after.txt
diff .debio/wire_keys_baseline.txt .debio/wire_keys_after.txt && echo "WIRE UNCHANGED"
```
Expected: `WIRE UNCHANGED` (empty diff). A non-empty diff means a manual coupled-string edit changed a wire/field key — revert that specific string change (keep the wire literal, decouple the internal symbol only) and re-run.

- [ ] **Step 6: Gate + Commit**

Run: `../venv/bin/python -m pytest -q` → all pass; `../venv/bin/python -m pytest tests/test_regression_golden.py -q` → pass.
```bash
git add -A && git commit -m "refactor: de-biologize reproduction identifiers (reproduce->replicate, mate->partner etc.)"
```

---

### Task 4: Cluster — Metabolism

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`.

Note: `METABOLISM` is also a hormone-array index constant in `endocrine.py` (`METABOLISM = 7`); renaming it to `UPKEEP` here is consistent and intended.

- [ ] **Step 1: Write the map**
```bash
cat > .debio/metabolism.json <<'JSON'
{"metabolism":"upkeep","metabolic":"upkeep","metabolize":"consume_upkeep",
 "metabolise":"consume_upkeep","digest":"process","digestion":"processing",
 "starve":"deplete","starves":"depletes","starving":"depleting","starvation":"depletion",
 "hunger":"energy_need","hungry":"depleted",
 "nutrient":"resource_value","nutrients":"resource_value","nutrition":"resource_value",
 "nutritional":"resource_value","calorie":"energy","calories":"energy"}
JSON
```

- [ ] **Step 2: Dry-run and review**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/metabolism.json $(git ls-files '*.py' | grep -v '^archive/')`
Expected: `metabolism->upkeep`, `METABOLISM->UPKEEP`, `hunger->energy_need`, `starvation->depletion`. Confirm no kept term touched.

- [ ] **Step 3: Apply**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/metabolism.json --apply $(git ls-files '*.py' | grep -v '^archive/')`

- [ ] **Step 4: Coupled strings — check hunger/metabolism tag strings**
```bash
git ls-files '*.py' | grep -v archive | xargs grep -n "\"hunger\"\|'hunger'\|\"metabolism\"\|'metabolism'\|apply_substance(" 2>/dev/null
```
For any string literal that mirrors a renamed attribute/tag AND is server-internal (not a wire key), rename it in lockstep. `plant_food` and other kept-term tags stay. If a tag is passed into a serialized/wire structure, keep it and note it.

- [ ] **Step 5: Gate + Commit**

Run: `../venv/bin/python -m pytest -q` → pass; golden → pass.
```bash
git add -A && git commit -m "refactor: de-biologize metabolism identifiers (metabolism->upkeep, hunger->energy_need etc.)"
```

---

### Task 5: Cluster — Foraging / plants / hunting

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`.

Kept: `plant`, `plant_food`, `food`, `eat`.

- [ ] **Step 1: Write the map**
```bash
cat > .debio/foraging.json <<'JSON'
{"forage":"gather","forages":"gathers","forager":"gatherer","foragers":"gatherers",
 "foraging":"gathering","vegetation":"resource_cover","photosynthesis":"resource_regrowth",
 "predator":"attacker","predators":"attackers","prey":"target",
 "hunt":"pursue","hunts":"pursues","hunting":"pursuing","hunter":"pursuer",
 "pheromone":"signal_marker","pheromones":"signal_marker","scent":"trail","scents":"trail",
 "fermentation":"spoilage","ferment":"spoil","ferments":"spoils","fermented":"spoiled",
 "fermenting":"spoiling"}
JSON
```
(The `fermentation` module concept is folded in here so its module-path token is rewritten alongside the identifiers. `ferment`→`spoil` is a name only — verify in review it reads sensibly for the food-processing logic; behaviour is unchanged regardless.)

- [ ] **Step 2: Dry-run and review**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/foraging.json $(git ls-files '*.py' | grep -v '^archive/')`
Expected: `forage->gather`, `forager->gatherer`, `scent->trail`, `fermentation->spoilage`. Confirm `plant*`/`food`/`eat` are untouched.

- [ ] **Step 3: Apply**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/foraging.json --apply $(git ls-files '*.py' | grep -v '^archive/')`

- [ ] **Step 3b: Rename the `fermentation` module file to match**

The tool rewrote the module-path token `fermentation`→`spoilage` in imports; move the file to match, in this commit:
```bash
git mv artificial_society/environment/fermentation.py artificial_society/environment/spoilage.py
git ls-files '*.py' | grep -v archive | xargs grep -nE "import_module\(|['\"][a-z_.]*fermentation['\"]" 2>/dev/null || echo "no dynamic fermentation import"
```

- [ ] **Step 4: Coupled string — `'forage'` action-mode label**

Same rule as the `'mate'` label (Task 3 Step 4): if `'forage'` is a `last_action_mode` value + `renderer.py` colour key, decide by wire-visibility:
```bash
git ls-files '*.py' | grep -v archive | xargs grep -n "'forage'\|\"forage\"" 2>/dev/null
git ls-files | grep -iE '\.(js|html)$' | grep -v archive | xargs grep -ln "forage" 2>/dev/null
```
Wire-visible → keep the literal; internal → rename the coupled literals to `'gather'` in lockstep.

- [ ] **Step 5: Wire gate** (re-run the Task 3 Step 5 diff) → expect `WIRE UNCHANGED`.

- [ ] **Step 6: Gate + Commit**

Run: `../venv/bin/python -m pytest -q` → pass; golden → pass.
```bash
git add -A && git commit -m "refactor: de-biologize foraging/plants/hunting identifiers (forage->gather etc.)"
```

---

### Task 6: Cluster — Health / disease / injury

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`. Class `DiseaseSystem` → `FaultSystem` (handled automatically by CamelCase splitting). Module `systems/disease.py` → `systems/fault.py` (singular, matching the `disease`→`fault` identifier rename — note this differs from the spec's `faults.py`; singular keeps the module path token and identifier consistent so the rename is fully tool-driven). Kept: `health`.

- [ ] **Step 1: Write the map**
```bash
cat > .debio/health.json <<'JSON'
{"disease":"fault","diseases":"faults","diseased":"impaired",
 "sick":"impaired","sickness":"impairment",
 "infect":"spread","infects":"spreads","infected":"spread","infection":"propagation",
 "infections":"propagation","infectious":"spreading","contagion":"propagation",
 "contagious":"spreading","immune":"resistant","immunity":"resistance",
 "symptom":"indicator","symptoms":"indicators",
 "heal":"recover","heals":"recovers","healing":"recovery","healed":"recovered",
 "wound":"damage","wounds":"damage","wounded":"damaged",
 "injury":"damage","injuries":"damage","injured":"damaged"}
JSON
```

- [ ] **Step 2: Dry-run and review**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/health.json $(git ls-files '*.py' | grep -v '^archive/')`
Expected: `disease->fault`, `DiseaseSystem->FaultSystem`, `is_sick->is_impaired`, `heal->recover`. Confirm `health` is untouched.

- [ ] **Step 3: Apply**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/health.json --apply $(git ls-files '*.py' | grep -v '^archive/')`

- [ ] **Step 3b: Rename the `disease` module file to match**

The tool rewrote the module-path token `disease`→`fault` in imports (`systems.disease`→`systems.fault`); move the file to match, in this commit:
```bash
git mv artificial_society/systems/disease.py artificial_society/systems/fault.py
git ls-files '*.py' | grep -v archive | xargs grep -nE "import_module\(|['\"][a-z_.]*systems.disease['\"]|['\"]disease['\"]" 2>/dev/null
```
Update any dynamic-import string of the `disease` module to `fault`. (Non-import `'disease'` string keys are handled in Step 4.)

- [ ] **Step 4: Coupled strings — disease/sick string keys & modes**
```bash
git ls-files '*.py' | grep -v archive | xargs grep -n "'disease'\|\"disease\"\|'sick'\|\"sick\"" 2>/dev/null
```
Rename internal ones (e.g. an internal status tag `'sick'` mirrored to an `is_impaired` attribute) in lockstep; keep any wire/log-visible ones and note them.

- [ ] **Step 5: Wire gate** (Task 3 Step 5 diff) → `WIRE UNCHANGED`.

- [ ] **Step 6: Gate + Commit**

Run: `../venv/bin/python -m pytest -q` → pass; golden → pass.
```bash
git add -A && git commit -m "refactor: de-biologize health/disease + rename module to fault.py"
```

---

### Task 7: Cluster — Endocrine / drives (with hormone getattr coupling)

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`. Class `EndocrineSystem` → `ModulationSystem`. Module `agents/endocrine.py` → `agents/modulation.py`.

⚠ **Correctness hazard:** hormone names appear both as identifiers (`CORTISOL`, `h[CORTISOL]`) **and** as string literals bound to attributes — `getattr(agent.endocrine, "cortisol", 0)` (`systems/invention.py:395`) and dict keys (`'cortisol': ...` in `agents/emotional_memory.py`). If the identifier/attribute is renamed but the string is not, `getattr` silently returns the default `0` → **behaviour change**. The coupled strings MUST be renamed in the same commit.

- [ ] **Step 1: Write the map**
```bash
cat > .debio/endocrine.json <<'JSON'
{"endocrine":"modulation","hormone":"modulator","hormones":"modulators",
 "hormonal":"modulatory","cortisol":"stress","adrenaline":"arousal",
 "dopamine":"reward","serotonin":"satisfaction","melatonin":"rest",
 "oxytocin":"affiliation","inflammation":"irritation","circadian":"day_cycle"}
JSON
```
(`inflammation` — the `INFLAMMATION = 6` hormone index — is renamed to `irritation`; flag in review if a better neutral term is preferred.)

- [ ] **Step 2: Dry-run and review**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/endocrine.json $(git ls-files '*.py' | grep -v '^archive/')`
Expected: `endocrine->modulation`, `EndocrineSystem->ModulationSystem`, `CORTISOL->STRESS`, `hormone->modulator`, `h[CORTISOL]->h[STRESS]`. Confirm no kept term touched.

- [ ] **Step 3: Apply**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/endocrine.json --apply $(git ls-files '*.py' | grep -v '^archive/')`

- [ ] **Step 3b: Rename the `endocrine` module file to match**

The tool rewrote the module-path token `endocrine`→`modulation` in imports; move the file to match, in this commit:
```bash
git mv artificial_society/agents/endocrine.py artificial_society/agents/modulation.py
git ls-files '*.py' | grep -v archive | xargs grep -nE "import_module\(|['\"][a-z_.]*endocrine['\"]" 2>/dev/null || echo "no dynamic endocrine import"
```

- [ ] **Step 4: Rename the coupled hormone string literals in lockstep**

Find every hormone name used as a string (dict key, `getattr`/`hasattr`/`setattr` arg, `apply_substance` tag):
```bash
git ls-files '*.py' | grep -v archive | xargs grep -nE "['\"](cortisol|adrenaline|dopamine|serotonin|melatonin|oxytocin|endocrine)['\"]" 2>/dev/null
```
For each, apply the same mapping to the string value (`"cortisol"`→`"stress"`, `"dopamine"`→`"reward"`, etc.), EXCEPT any that a preceding grep of dashboard `.js`/`.html` shows is wire-visible:
```bash
git ls-files | grep -iE '\.(js|html)$' | grep -v archive | xargs grep -lnE "cortisol|dopamine|serotonin|adrenaline|oxytocin|melatonin" 2>/dev/null
```
If a hormone name is consumed by the dashboard, KEEP that literal and instead decouple (leave the emitted string, rename only the internal symbol) — note it in the commit message. Otherwise rename the string. Confirm `getattr(agent.endocrine, "cortisol", 0)` in `systems/invention.py` and its target attribute now use the same new name.

- [ ] **Step 5: Targeted coupling assertion**

Verify the endocrine object exposes the new names and the old `getattr` default is not silently hit:
```bash
../venv/bin/python - <<'PY'
import inspect, artificial_society.systems.invention as inv
src = inspect.getsource(inv)
assert 'cortisol' not in src, "stale hormone string still referenced in invention.py"
print("coupling OK")
PY
```
Expected: `coupling OK`.

- [ ] **Step 6: Wire gate** (Task 3 Step 5 diff) → `WIRE UNCHANGED`.

- [ ] **Step 7: Gate + Commit**

Run: `../venv/bin/python -m pytest -q` → pass; golden → pass.
```bash
git add -A && git commit -m "refactor: de-biologize endocrine/drives + rename module to modulation.py + coupled hormone strings"
```

---

### Task 8: Cluster — Lifecycle / organism

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`. Kept: `die`, `death`, `alive`, `age`, `grow`, `spawn`, `selection`.

- [ ] **Step 1: Write the map**
```bash
cat > .debio/lifecycle.json <<'JSON'
{"organism":"agent","organisms":"agents","creature":"agent","creatures":"agents",
 "animal":"agent","animals":"agents","species":"agent_type",
 "senescence":"decay","senescent":"decayed","lifespan":"max_age",
 "evolve":"adapt","evolves":"adapts","evolved":"adapted","evolution":"adaptation",
 "evolutionary":"adaptive","fitness":"score"}
JSON
```
(`selection` is kept — generic. `fitness` is the only borderline that is renamed, to `score`.)

- [ ] **Step 2: Dry-run and review**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/lifecycle.json $(git ls-files '*.py' | grep -v '^archive/')`
Expected: `organism->agent`, `lifespan->max_age`, `fitness->score`. Confirm `die`/`death`/`alive`/`age`/`selection` untouched.

- [ ] **Step 3: Apply**

Run: `../venv/bin/python scripts/debio_rename.py --map .debio/lifecycle.json --apply $(git ls-files '*.py' | grep -v '^archive/')`

- [ ] **Step 4: Coupled strings — none expected.** Quick check:
```bash
git ls-files '*.py' | grep -v archive | xargs grep -nE "['\"](organism|creature|species|fitness)['\"]" 2>/dev/null
```
Rename any internal string mirrors; keep wire-visible ones.

- [ ] **Step 5: Gate + Commit**

Run: `../venv/bin/python -m pytest -q` → pass; golden → pass.
```bash
git add -A && git commit -m "refactor: de-biologize lifecycle/organism identifiers (organism->agent, fitness->score etc.)"
```

---

### Task 9: Module-path & manifest consistency audit

The four biological module files were already `git mv`'d inside the clusters that renamed their concept (`genetics.py`→`traits.py` in Task 2, `fermentation.py`→`spoilage.py` in Task 5, `disease.py`→`fault.py` in Task 6, `endocrine.py`→`modulation.py` in Task 7), and the tool rewrote the import paths in the same commits. This task is a safety net that proves no stale module path or manifest reference survived.

**Files:** verification only (fixes go back into the relevant cluster if something is stale).

- [ ] **Step 1: Confirm the old module files are gone and the new ones exist**
```bash
for old in artificial_society/agents/genetics.py artificial_society/agents/endocrine.py \
           artificial_society/environment/fermentation.py artificial_society/systems/disease.py \
           tests/agents/test_genetics_strength.py; do
  [ -e "$old" ] && echo "STALE STILL PRESENT: $old"
done
for new in artificial_society/agents/traits.py artificial_society/agents/modulation.py \
           artificial_society/environment/spoilage.py artificial_society/systems/fault.py \
           tests/agents/test_traits_strength.py; do
  [ -e "$new" ] || echo "MISSING NEW FILE: $new"
done
echo "module file check done"
```
Expected: only `module file check done` (no STALE/MISSING lines).

- [ ] **Step 2: Confirm no import still references an old module path**
```bash
git ls-files '*.py' | grep -v '^archive/' | xargs grep -nE \
  'agents\.genetics|agents\.endocrine|environment\.fermentation|systems\.disease|import (genetics|endocrine|fermentation)\b' \
  2>/dev/null || echo "no stale module imports"
```
Expected: `no stale module imports`. If any hit appears, it means a module-path token was inside a string (dynamic import) the tool skipped — fix it and fold into the relevant cluster commit via `git commit --amend` if not yet pushed, else a new fixup commit.

- [ ] **Step 3: Confirm the manifest is clean**
```bash
grep -nE 'genetics|endocrine|fermentation|systems.disease' artificial_society_manifest.json || echo "manifest clean"
```
Expected: `manifest clean` (the `agents\genetics.py` path was fixed to `agents\traits.py` in Task 2).

- [ ] **Step 4: Full gate**

Run: `../venv/bin/python -m pytest -q` → all pass. Golden → pass. Wire gate (Task 3 Step 5) → `WIRE UNCHANGED`. No commit unless Step 2 produced a fix.

---

### Task 10: Comments & docstrings sweep

**Files:** all of `git ls-files '*.py' | grep -v '^archive/'`. The rename tool skipped comments/strings; this pass updates the prose in comments and docstrings (cosmetic — no behaviour). Semantic string literals are NOT touched here.

- [ ] **Step 1: Add a comment/docstring rewriter test**

Extend `tests/tools/test_debio_rename.py` with a docstring/comment mode. First write the failing test:
```python
def test_prose_mode_rewrites_comments_and_docstrings_only():
    from debio_rename import rewrite_prose
    src = '"""Handle gene mutation."""\nx = "gene"  # a gene here\n'
    new = rewrite_prose(src, {"gene": "trait", "mutation": "perturbation"})
    # docstring + comment rewritten; the semantic string "gene" left intact
    assert new == '"""Handle trait perturbation."""\nx = "gene"  # a trait here\n'
```
Run: `../venv/bin/python -m pytest tests/tools/test_debio_rename.py::test_prose_mode_rewrites_comments_and_docstrings_only -q` → FAIL (`rewrite_prose` missing).

- [ ] **Step 2: Implement `rewrite_prose` in `scripts/debio_rename.py`**

Add:
```python
import ast


def _sub_words(text: str, mapping: dict[str, str]) -> str:
    def repl(m):
        w = m.group(0)
        r = mapping.get(w.lower())
        return _recase(r, w) if r is not None else w
    return re.sub(r"[A-Za-z]+", repl, text)


def rewrite_prose(src: str, mapping: dict[str, str]) -> str:
    """Rewrite words only inside COMMENT tokens and docstrings; code strings kept."""
    mapping = {k.lower(): v for k, v in mapping.items()}
    # docstring line ranges (module/class/func first-statement string expressions)
    doc_ranges = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(getattr(first, "value", None), ast.Constant)
                        and isinstance(first.value.value, str)):
                    for ln in range(first.lineno, first.end_lineno + 1):
                        doc_ranges.add(ln)
    lines = src.splitlines(keepends=True)
    readline = iter(lines).__next__
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.COMMENT and tok.start[0] == tok.end[0]:
            new = _sub_words(tok.string, mapping)
            if new != tok.string:
                edits.setdefault(tok.start[0], []).append((tok.start[1], tok.end[1], new))
    for row in sorted(doc_ranges):
        line = lines[row - 1]
        edits.setdefault(row, []).append((0, len(line.rstrip("\n")), _sub_words(line.rstrip("\n"), mapping)))
    for row, row_edits in edits.items():
        line = lines[row - 1]
        for scol, ecol, new in sorted(row_edits, reverse=True):
            line = line[:scol] + new + line[ecol:]
        lines[row - 1] = line
    return "".join(lines)
```
Run: `../venv/bin/python -m pytest tests/tools/test_debio_rename.py -q` → PASS.

- [ ] **Step 3: Build the union prose map and dry-run**

Combine all cluster maps into one prose map (keys are the OLD words → neutral words), then rewrite prose across the tree via a small driver:
```bash
../venv/bin/python - <<'PY'
import json, glob, subprocess, pathlib
from importlib import import_module
import sys
sys.path.insert(0, "scripts")
from debio_rename import rewrite_prose
maps = {}
for f in glob.glob(".debio/*.json"):  # every .debio/*.json is a cluster map
    maps.update(json.loads(pathlib.Path(f).read_text()))
files = subprocess.check_output(["bash","-lc","git ls-files '*.py' | grep -v '^archive/'"]).decode().split()
changed = 0
for f in files:
    p = pathlib.Path(f); src = p.read_text(); new = rewrite_prose(src, maps)
    if new != src: p.write_text(new); changed += 1
print("prose-updated files:", changed)
PY
```

- [ ] **Step 4: Review the diff**

Run: `git diff --stat` then spot-check `git diff` for a few files. Confirm only comments/docstrings changed (no code string literals). Revert any file where a semantic string was altered.

- [ ] **Step 5: Gate + Commit**

Run: `../venv/bin/python -m pytest -q` → pass; golden → pass.
```bash
git add -A && git commit -m "refactor: de-biologize comments and docstrings"
```

---

### Task 11: Final leak audit

**Files:** whole branch diff vs `main`.

- [ ] **Step 1: Grep the codebase for any remaining rename-list biological identifier**
```bash
git ls-files '*.py' | grep -v '^archive/' | xargs grep -nEiw \
 'organism|creature|animal|species|gene|genes|genetic|genetics|genome|genotype|allele|mutation|mutate|dna|heredity|hereditary|lineage|reproduce|reproduction|mating|fertility|fecundity|pregnant|gestation|offspring|embryo|metabolism|metabolic|metabolize|digest|starve|starvation|hunger|nutrient|nutrition|forage|forager|vegetation|photosynthesis|predator|prey|pheromone|scent|disease|sick|infect|infection|contagion|immune|immunity|symptom|heal|wound|injury|endocrine|hormone|cortisol|adrenaline|dopamine|serotonin|melatonin|oxytocin|circadian|senescence|lifespan|evolve|evolution|fitness' \
 2>/dev/null | grep -viE 'plant_food|# *kept|selection' | tee .debio/leak_audit.txt | head -60
wc -l .debio/leak_audit.txt
```

- [ ] **Step 2: Classify each remaining hit**

Every line in `.debio/leak_audit.txt` must be one of:
- a **kept term** context (e.g. `plant_food`, `health`, `selection`) — OK;
- an intentionally **kept wire/log string** documented in an earlier task's commit — OK;
- a **missed identifier** — add it to the relevant cluster map and re-run that cluster task (do not hand-edit; keep the tool as the single source of truth), or hand-fix if it is a one-off in a comment.

Iterate until the audit is empty of genuine misses.

- [ ] **Step 3: Full verification battery**

```bash
../venv/bin/python -m pytest -q
../venv/bin/python -m pytest tests/test_regression_golden.py -q
git diff --stat main -- tests/golden_trajectory.json   # expect: NO change to the golden file
{ grep -hoE '"[a-z_]+"' artificial_society/visualization/statistics.py artificial_society/cell_store.py artificial_society/serve/app.py artificial_society/serve/runner.py 2>/dev/null; } | sort -u > .debio/wire_keys_after.txt
diff .debio/wire_keys_baseline.txt .debio/wire_keys_after.txt && echo "WIRE UNCHANGED"
```
Expected: suite passes; golden file unchanged vs `main`; `WIRE UNCHANGED`.

- [ ] **Step 4: Checkpoint round-trip in-session**
```bash
rm -f checkpoint.pkl
../venv/bin/python - <<'PY'
from artificial_society.simulation import Simulation
s = Simulation(headless=True, load_checkpoint=False, seed=1, grid_w=20, grid_h=12, initial_population=8)
for _ in range(5): s.step()
s._save_checkpoint()
s2 = Simulation(headless=True, load_checkpoint=True, seed=1, grid_w=20, grid_h=12, initial_population=8)
print("checkpoint round-trip OK")
PY
rm -f checkpoint.pkl
```
Expected: `checkpoint round-trip OK` (new-session save+load works; old pre-rename `.pkl` intentionally not tested).

- [ ] **Step 5: Commit the audit artifact removal / finalize**

`.debio/` is git-ignored, so nothing to commit unless a re-run of a cluster produced fixes. If clean, the branch is complete. Then hand off to `superpowers:finishing-a-development-branch`.

---

## Notes for the executor

- The rename tool is the single source of truth for identifier renames — prefer editing a cluster map + re-running over hand-editing identifiers, so renames stay consistent and idempotent.
- Only three categories require manual string edits: (1) coupled `getattr`/dict-key hormone strings (Task 7), (2) action-mode label strings `'mate'`/`'forage'` (Tasks 3/5, only if internal), (3) the manifest path (Task 2). Module files are `git mv`'d inside the cluster that renames their concept (Tasks 2/5/6/7); everything else is tool-driven.
- If the wire gate shows a change, the fix is always to KEEP the wire literal and decouple (rename only the internal symbol), never to change the dashboard — external contracts are out of scope.
- The rich live-viz websocket frame is NOT on this branch (it lives on `feat/infra-live-viz-v2`); the wire contract here is the `/api/status`+`/api/history` stats JSON, guarded by the wire-key fingerprint.
