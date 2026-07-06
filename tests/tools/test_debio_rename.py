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
