"""Research lane — Stage 0a pilot & decision gate.

Non-invasive tooling to answer the pre-registered gate question of the
open-ended-innovation research design (see
``docs/superpowers/specs/2026-06-28-open-ended-innovation-research-design.md``):

    Does the *functional* (irreducible) innovation complexity produced by the
    learned/social agents beat a compute-matched random-recombiner null, with
    separated bootstrap CIs?

Everything here lives outside the hot-file contract: the learned arm runs the
*unmodified* ``Simulation``; the random-recombiner is a standalone generator that
drives ``materials.combine_vectors`` directly. No source file of the simulation is
edited, so the determinism contract (golden trajectory + headless digest) is
untouched.
"""

from __future__ import annotations

__all__ = [
    "instrument",
    "export",
    "metrics",
    "recombiner",
]
