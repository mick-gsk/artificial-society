"""Kalibrierungstabelle — das Realitäts-Gate der Physik v2 (Spec §2).

Jede Dimension, jedes Startmaterial und jeder Prozess MUSS hier einen Eintrag
mit realem Anker + Quelle haben. tests/environment/physics/test_reality_gate.py
erzwingt das als Merge-Bedingung: was keinen realen Anker hat, kommt nicht in
die Welt. docs/physics/kalibrierung.md wird aus dieser Datei generiert
(scripts/gen_kalibrierung.py) — nie von Hand editieren.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_KINDS = ("dim", "material", "process", "body")


@dataclass(frozen=True)
class CalEntry:
    kind: str
    name: str
    anchor: str  # realer Anker inkl. Normierungsformel/Skala
    source: str  # Herkunft des Ankers


CALIBRATION: dict = {}


def cal(kind: str, name: str, anchor: str, source: str) -> None:
    """Kalibrierungs-Eintrag registrieren; leerer Anker/Quelle ist ein Fehler."""
    if kind not in VALID_KINDS:
        raise ValueError(f"unbekannter Kalibrierungs-Typ: {kind!r}")
    if not anchor.strip() or not source.strip():
        raise ValueError(f"Kalibrierung {kind}:{name} braucht Anker UND Quelle")
    CALIBRATION[(kind, name)] = CalEntry(kind=kind, name=name, anchor=anchor, source=source)


def entry_for(kind: str, name: str):
    return CALIBRATION.get((kind, name))


# ---------------------------------------------------------------------------
# Dimensionen (Normierungsanker)
# ---------------------------------------------------------------------------
cal(
    "dim",
    "hardness",
    "Ritzhärte, Mohs-Skala / 10 (Talk 1 → 0.1, Quarz/Feuerstein 7 → 0.7, Diamant 10 → 1.0)",
    "Mohs-Härteskala (Mineralogie-Standard)",
)
cal(
    "dim",
    "density",
    "Rohdichte ρ / 5000 kg/m³, geklemmt auf [0,1] (Wasser 1000 → 0.2, Granit 2700 → 0.54)",
    "CRC Handbook / Gesteinskunde-Standardwerte",
)
cal(
    "dim",
    "brittleness",
    "Sprödigkeit: 0 = zäh/duktil (Sehne, frisches Holz) … 1 = ideal spröde mit muscheligem "
    "Bruch (Glas/Obsidian ≈ 0.95, Feuerstein ≈ 0.9)",
    "Bruchmechanik spröder vs. duktiler Werkstoffe; Lithik (Feuersteinschlagen)",
)
cal(
    "dim",
    "tensile_strength",
    "Zugfestigkeit / 1000 MPa (Hanffaser ≈ 600 MPa → 0.6, Holz längs ≈ 100 → 0.1, Fels ≈ 10 → 0.01)",
    "Werkstoffkunde-Tabellenwerte",
)
cal(
    "dim",
    "sharpness",
    "Kantenschärfe: 0 = stumpf … 1 ≈ frisch geschlagene Obsidianklinge. NUR als "
    "Prozessergebnis (Bruch) erzeugbar — kein Startmaterial hat sharpness > 0",
    "Experimentelle Archäologie: Schärfe geschlagener Steinwerkzeuge",
)
cal(
    "dim",
    "flammability",
    "Entflammbarkeit 0..1 qualitativ: 0 = nicht brennbar (Stein), 0.8 = trockenes Holz, 1 = Zunder",
    "Brandverhalten von Naturstoffen (Forst-/Brandschutzliteratur)",
)
cal(
    "dim",
    "ignition_temp",
    "Zündtemperatur / 1000 °C (Holz ≈ 300 °C → 0.3); 1.0 = praktisch nicht entzündbar",
    "Zündtemperatur-Tabellen (Holz 280–340 °C)",
)
cal(
    "dim",
    "melting_point",
    "Schmelz-/Sinterpunkt / 2000 °C (Ton sintert ≈ 1000 → 0.5, Quarz ≈ 1670 → 0.84); "
    "1.0 = schmilzt praktisch nicht bzw. zersetzt sich vorher",
    "Keramik-/Petrologie-Standardwerte",
)
cal(
    "dim",
    "thermal_conductivity",
    "Wärmeleitfähigkeit / 10 W/(m·K), geklemmt (Granit ≈ 2.8 → 0.28, Holz ≈ 0.15 → 0.02)",
    "CRC Handbook, Wärmeleitfähigkeiten",
)
cal(
    "dim",
    "nutrition",
    "verwertbare Energie / 400 kcal pro 100 g (mageres Rohfleisch ≈ 140 → 0.35, Beeren ≈ 50 → 0.13)",
    "Nährwerttabellen (USDA)",
)
cal(
    "dim",
    "toxicity",
    "akute Schädlichkeit bei rohem Verzehr, 0..1 qualitativ (rohes Aas ≈ 0.15 wegen Keimbelastung)",
    "Lebensmittelhygiene: Keimbelastung roher Tierprodukte",
)
cal(
    "dim",
    "moisture",
    "Wasseranteil 0..1 (Frischfleisch ≈ 0.7, lufttrockenes Holz ≈ 0.12, Wasser = 1.0)",
    "Holzfeuchte-/Lebensmitteltabellen",
)
cal(
    "dim",
    "grain_fineness",
    "Gefügefeinheit: 0 = grobkristallin (Granit ≈ 0.15) … 1 = kryptokristallin (Feuerstein ≈ 0.95). "
    "Bestimmt muscheligen Bruch und erreichbare Kantenschärfe",
    "Petrologie: Kryptokristallinität von Silex — Grundlage des Feuersteinschlagens",
)

_KIND_TITLES = {
    "dim": "Eigenschafts-Dimensionen",
    "material": "Startmaterialien",
    "process": "Prozesse",
    "body": "Körper-Parameter",
}

_DOC_HEADER = (
    "# Kalibrierungstabelle Physik v2\n\n"
    "> GENERIERT aus `artificial_society/environment/physics/calibration.py` via\n"
    "> `scripts/gen_kalibrierung.py` — nicht von Hand editieren.\n"
    "> Realitäts-Gate: Spec `docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md` §2.\n"
)


def render_markdown() -> str:
    """Kalibrierungstabelle als Markdown (SSOT = diese Datei)."""
    # Import hier, damit alle cal()-Registrierungen der Schwester-Module feuern
    # (kein Import-Zyklus: die Schwestern importieren nur cal aus diesem Modul).
    from . import body, materials_v2, processes  # noqa: F401

    lines = [_DOC_HEADER]
    for kind in VALID_KINDS:
        lines.append(f"\n## {_KIND_TITLES[kind]}\n")
        lines.append("| Name | Realer Anker | Quelle |")
        lines.append("|---|---|---|")
        entries = sorted((e for e in CALIBRATION.values() if e.kind == kind), key=lambda e: e.name)
        for e in entries:
            lines.append(f"| `{e.name}` | {e.anchor} | {e.source} |")
    return "\n".join(lines) + "\n"
