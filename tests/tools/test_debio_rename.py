import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import pytest  # noqa: E402
from debio_rename import rename_identifier, rewrite_source  # noqa: E402

MAP = {
    "gene": "trait",
    "genes": "traits",
    "genetic": "trait_based",
    "genetics": "traits",
    "inherit": "derive",
    "disease": "fault",
    "endocrine": "modulation",
}


@pytest.mark.parametrize(
    "name,expected",
    [
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
    ],
)
def test_rename_identifier(name, expected):
    assert rename_identifier(name, MAP) == expected


def test_strings_and_comments_untouched():
    src = 'x = "gene"  # gene comment\ny = genes\n'
    new, changes = rewrite_source(src, MAP)
    assert new == 'x = "gene"  # gene comment\ny = traits\n'
    assert changes == {"genes->traits": 1}


def test_prose_mode_rewrites_comments_and_docstrings_only():
    from debio_rename import rewrite_prose

    src = '"""Handle gene mutation."""\nx = "gene"  # a gene here\n'
    new = rewrite_prose(src, {"gene": "trait", "mutation": "perturbation"})
    # docstring + comment rewritten; the semantic string "gene" left intact
    assert new == '"""Handle trait perturbation."""\nx = "gene"  # a trait here\n'
