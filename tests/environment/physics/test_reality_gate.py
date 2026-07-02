"""Realitäts-Gate (Spec §2): kein Physik-Baustein ohne realen Kalibrierungs-Anker.

Dieser Test ist eine dauerhafte MERGE-BEDINGUNG: Wer eine Dimension, ein
Material oder einen Prozess hinzufügt, muss einen cal()-Eintrag mit realem
Anker + Quelle mitliefern — sonst rot.
"""

from __future__ import annotations

from pathlib import Path

from artificial_society.environment.phys_objects import CALIBRATED_SPAWN_PARAMS
from artificial_society.environment.physics import MATERIALS_V2, PROP_DIMS_V2
from artificial_society.environment.physics.actions import CALIBRATED_ACTION_PARAMS
from artificial_society.environment.physics.body import CALIBRATED_BODY_PARAMS
from artificial_society.environment.physics.calibration import (
    CALIBRATION,
    entry_for,
    render_markdown,
)
from artificial_society.environment.physics.processes import PROCESSES

REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_calibrated(kind: str, name: str) -> None:
    entry = entry_for(kind, name)
    assert entry is not None, f"{kind} '{name}' ohne Kalibrierungs-Eintrag (Realitäts-Gate)"
    assert entry.anchor.strip(), f"{kind} '{name}': leerer Anker"
    assert entry.source.strip(), f"{kind} '{name}': leere Quelle"


def test_every_dim_is_calibrated():
    for dim in PROP_DIMS_V2:
        _assert_calibrated("dim", dim)


def test_every_material_is_calibrated():
    for name in MATERIALS_V2:
        _assert_calibrated("material", name)


def test_every_process_is_calibrated():
    for name in PROCESSES:
        _assert_calibrated("process", name)


def test_every_body_param_is_calibrated():
    for name in CALIBRATED_BODY_PARAMS:
        _assert_calibrated("body", name)


def test_no_orphan_calibration_entries():
    # Ein Eintrag ohne zugehörigen Baustein = Umbenennung ohne Tabellenpflege.
    for kind, name in CALIBRATION:
        if kind == "dim":
            assert name in PROP_DIMS_V2, f"verwaister dim-Eintrag: {name}"
        elif kind == "material":
            assert name in MATERIALS_V2, f"verwaister material-Eintrag: {name}"
        elif kind == "process":
            assert name in PROCESSES, f"verwaister process-Eintrag: {name}"
        elif kind == "body":
            assert name in CALIBRATED_BODY_PARAMS, f"verwaister body-Eintrag: {name}"


def test_kalibrierung_doc_is_in_sync():
    doc = REPO_ROOT / "docs" / "physics" / "kalibrierung.md"
    assert doc.exists(), (
        "docs/physics/kalibrierung.md fehlt — scripts/gen_kalibrierung.py laufen lassen"
    )
    assert doc.read_text(encoding="utf-8") == render_markdown(), (
        "kalibrierung.md ist veraltet — scripts/gen_kalibrierung.py neu laufen lassen"
    )


def test_neue_kinds_spawn_und_action_sind_gueltig():
    """B2/B4: Spawn- und Aktions-Konstanten bekommen eigene Kalibrierungs-Kinds."""
    from artificial_society.environment.physics.calibration import VALID_KINDS, cal

    assert "spawn" in VALID_KINDS
    assert "action" in VALID_KINDS
    # cal() akzeptiert die neuen Kinds (Aufräumen: Test-Eintrag wieder entfernen,
    # sonst schlägt die Orphan-Prüfung späterer Tasks im selben Prozess an).
    try:
        cal("spawn", "_test_eintrag", "Test-Anker", "Test-Quelle")
        assert ("spawn", "_test_eintrag") in CALIBRATION
    finally:
        CALIBRATION.pop(("spawn", "_test_eintrag"), None)


def test_body_mass_default_ist_kalibriert():
    from artificial_society.environment.physics.body import (
        BODY_MASS_DEFAULT_KG,
        CALIBRATED_BODY_PARAMS,
    )

    assert BODY_MASS_DEFAULT_KG == 70.0
    assert "body_mass" in CALIBRATED_BODY_PARAMS
    _assert_calibrated("body", "body_mass")


def test_every_action_param_is_calibrated():
    for name in CALIBRATED_ACTION_PARAMS:
        _assert_calibrated("action", name)


def test_no_orphan_action_entries():
    for kind, name in CALIBRATION:
        if kind == "action":
            assert name in CALIBRATED_ACTION_PARAMS, f"verwaister action-Eintrag: {name}"


def test_every_spawn_param_is_calibrated():
    for name in CALIBRATED_SPAWN_PARAMS:
        _assert_calibrated("spawn", name)


def test_no_orphan_spawn_entries():
    for kind, name in CALIBRATION:
        if kind == "spawn":
            assert name in CALIBRATED_SPAWN_PARAMS, f"verwaister spawn-Eintrag: {name}"
