from artificial_society.agents.agent import Agent
from artificial_society.agents.memory import EpisodicMemory
from artificial_society.agents.genetics import inherit_genes


class EvolutionSystem:
    def make_child(self, parent, x, y, genes=None, other_parent=None):
        """
        Erzeugt ein Kind mit korrekter Zwei-Eltern-Vererbung.

        NEU: Stamm-Zuweisung ist jetzt paritaetisch -- das Kind erbt den
        Stamm des Elternteils mit hoeherem learning_score (Fitness-Proxy).
        Biologisches Vorbild: In manchen Gesellschaften dominiert der
        Stamm des staerkeren Elternteils (gemischte Systeme).
        Verhindert dass immer nur der Mutter-Stamm weitergegeben wird.
        """
        if genes is None:
            if other_parent is not None:
                genes = inherit_genes(parent, other_parent)
            else:
                genes = inherit_genes(parent)

        # Stamm-Bestimmung: fitness-gewichtet statt immer Mutter
        if other_parent is not None:
            score_a = max(0.01, getattr(parent,       'learning_score', 1.0))
            score_b = max(0.01, getattr(other_parent, 'learning_score', 1.0))
            dominant = parent if score_a >= score_b else other_parent
        else:
            dominant = parent

        child = Agent.spawn_child(
            x=x,
            y=y,
            genes=genes,
            generation=parent.generation + 1,
            parent_id=parent.id,
            tribe_id=dominant.tribe_id,
        )
        if parent.memory.resource_memory:
            child.memory.resource_memory = parent.memory.resource_memory[-3:]
        if other_parent and other_parent.memory.resource_memory:
            # Erbt auch einige Ressourcenerinnerungen des anderen Elternteils
            extra = other_parent.memory.resource_memory[-2:]
            for mem in extra:
                if mem not in child.memory.resource_memory:
                    child.memory.resource_memory.append(mem)

        if parent.tribe_id is not None:
            child.trust[parent.id] = 0.4
        if other_parent is not None:
            child.trust[other_parent.id] = 0.4
        return child
