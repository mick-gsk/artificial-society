"""Golden-trajectory regression guard for the structural refactor.

Phase 1 (converging the dual bootstrap/monkeypatch loop into one explicit
``Simulation.step`` and folding the patched agent step back into the class)
must be behaviour-preserving. This locks the exact per-tick population
trajectory so any unintended change is caught.

The trajectory is only reproducible across processes with a fixed
``PYTHONHASHSEED`` (set/dict iteration order otherwise varies per process), so
the comparison run happens in a pinned subprocess.

If a later phase *intentionally* changes behaviour (e.g. Phase 2 re-wires world
regrowth and births), regenerate the golden::

    PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy python -c \
        "import json; from tests._util import compute_trajectory; \
         json.dump(compute_trajectory(), open('tests/golden_trajectory.json','w'), indent=0)"
"""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_trajectory.json")

_SENTINEL = "@@@TRAJ@@@"
_CHILD = (
    "import sys, json; sys.path.insert(0, '.'); "
    "from tests._util import compute_trajectory; "
    f"print('{_SENTINEL}' + json.dumps(compute_trajectory()) + '{_SENTINEL}')"
)


def _trajectory_in_pinned_subprocess():
    env = dict(
        os.environ,
        PYTHONHASHSEED="0",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
    )
    out = subprocess.check_output([sys.executable, "-c", _CHILD], cwd=REPO_ROOT, env=env, text=True)
    payload = out.split(_SENTINEL)[1]
    return json.loads(payload)


def test_trajectory_matches_golden():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert _trajectory_in_pinned_subprocess() == golden
