from artificial_society.agents.agent import Agent
from artificial_society.agents.memory import EpisodicMemory
from artificial_society.agents.genetics import inherit_genes


class EvolutionSystem:
    def make_child(self, parent, x, y, genes=None, other_parent=None):
        """
        Erzeugt ein Kind-Agenten.

        Bug-Fix: Bisher wurde `other_parent` (der Vater) komplett ignoriert.
        `inherit_genes` akzeptiert bereits zwei Elternteile – wir uebergeben
        ihn jetzt korrekt.

        Biologisches Vorbild: Sexuelle Reproduktion kombiniert Gene beider
        Elternteile, wodurch genetische Diversitaet entsteht.
        """
        if genes is None:
            if other_parent is not None:
                genes = inherit_genes(parent, other_parent)
            else:
                genes = inherit_genes(parent)
        child = Agent.spawn_child(
            x=x,
            y=y,
            genes=genes,
            generation=parent.generation + 1,
            parent_id=parent.id,
            tribe_id=parent.tribe_id,
        )
        if parent.memory.resource_memory:
            child.memory.resource_memory = parent.memory.resource_memory[-3:]
        if parent.tribe_id is not None:
            child.trust[parent.id] = 0.4
            if other_parent is not None:
                child.trust[other_parent.id] = 0.4
        return child
