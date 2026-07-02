"""Runtime instrumentation — zero source edits.

The learned arm must run the *unmodified* simulation, yet we need two things from
it: the number of combination attempts (for compute-matching the recombiner) and
quiet stdout (``DiscoveryRegistry.register`` prints one line per discovery, which
is thousands of lines over a full run).

Both are achieved at runtime without touching any simulation source file:

* :func:`count_combine_calls` patches ``combine_vectors`` *by function identity*
  across every already-imported module. Several modules do
  ``from ...materials import combine_vectors`` (invention, need_driven_invention,
  …) so each holds its own binding; patching by identity covers all of them plus
  the in-module call inside ``materials.py``.
* :func:`quiet_stdout` redirects stdout to ``os.devnull`` for the duration of a
  run.

Both are context managers that fully restore the original state on exit.
"""

from __future__ import annotations

import contextlib
import os
import sys

import artificial_society.environment.materials as _materials


class CombineCounter:
    """Mutable counter handed back by :func:`count_combine_calls`."""

    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0


@contextlib.contextmanager
def count_combine_calls():
    """Count every ``combine_vectors`` invocation across all importing modules.

    Yields a :class:`CombineCounter` whose ``.n`` holds the live attempt count.
    Patching is by function identity, so it catches every module that imported the
    function under any name, then is fully reverted on exit.
    """
    counter = CombineCounter()
    orig = _materials.combine_vectors

    def wrapped(*args, **kwargs):
        counter.n += 1
        return orig(*args, **kwargs)

    patched = []
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "combine_vectors", None) is orig:
                mod.combine_vectors = wrapped
                patched.append(mod)
        except Exception:
            # Some module proxies raise on getattr/setattr; skip them.
            continue

    try:
        yield counter
    finally:
        for mod in patched:
            with contextlib.suppress(Exception):
                mod.combine_vectors = orig


@contextlib.contextmanager
def quiet_stdout(enabled: bool = True):
    """Silence stdout (e.g. the per-discovery ``[DISCOVERY] …`` prints)."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield
