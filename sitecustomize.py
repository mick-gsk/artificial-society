"""Auto-load the bootstrap before the simulation module imports."""

try:
    from artificial_society.bootstrap import patch_simulation_class  # noqa: F401
except Exception:
    pass
