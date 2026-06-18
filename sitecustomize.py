"""Auto-load runtime patches when Python starts in this repository."""

try:
    import artificial_society.runtime_patches  # noqa: F401
except Exception:
    # Keep startup resilient; the simulation can still run without the patch.
    pass
