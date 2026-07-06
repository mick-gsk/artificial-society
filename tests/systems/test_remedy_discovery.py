"""Lane test: remedies are discovered from material properties, not a hardcoded lookup.

Phase 5 de-scripting (Task 5). ``evaluate_remedy`` previously matched consumed tags against
a disease's hardcoded ingredient list (``REMEDY_REGISTRY[...]['ingredients']``). After
de-scripting, cure efficacy emerges from the medicinal PROPERTIES of whatever is consumed --
aromatic, edible, low-toxicity materials heal (by their vector); herbs are generically
medicinal -- so a material that is NOT the disease's authored ingredient can still relieve
it, while a property-irrelevant material cannot.
"""

from artificial_society.systems.remedy import REMEDY_REGISTRY, evaluate_remedy


class _ImpairedAgent:
    def __init__(self, fault_id="malaria", impaired=60.0, health=50.0):
        self.fault_id = fault_id
        self.impaired = impaired
        self.health = health


def test_property_material_relieves_fault_though_not_an_ingredient():
    agent = _ImpairedAgent()
    assert "flower_petals" not in REMEDY_REGISTRY["malaria"]["ingredients"]
    before = agent.impaired

    # flower_petals is aromatic + low-toxicity: medicinal by property, never named a cure.
    reward = evaluate_remedy(agent, ["flower_petals"])

    assert reward > 0.0
    assert agent.impaired < before, "a property-medicinal material gave no relief"


def test_non_medicinal_material_gives_no_relief():
    agent = _ImpairedAgent()
    before = agent.impaired

    reward = evaluate_remedy(agent, ["stone"])

    assert reward == 0.0
    assert agent.impaired == before


def test_sustained_medicinal_treatment_can_fully_cure():
    agent = _ImpairedAgent(impaired=40.0, health=40.0)
    # A sustained aromatic/herbal regimen accumulates enough medicinal dose to fully cure,
    # even though none of these are malaria's authored ingredients.
    for _ in range(8):
        evaluate_remedy(agent, ["flower_petals", "tree_resin", "herb_garlic"])

    assert agent.fault_id is None, "sustained medicinal treatment failed to cure"
