from artificial_society.agents.agent import Agent
from artificial_society.agents.traits import derive_traits


class AdaptationSystem:
    def make_spawn(self, parent, x, y, traits=None, other_parent=None):
        """
        Erzeugt ein Kind mit korrekter Zwei-Eltern-Vererbung.

        NEU: Stamm-Zuweisung ist jetzt paritaetisch -- das Kind erbt den
        Stamm des Elternteils mit hoeherem learning_score (Score-Proxy).
        Biologisches Vorbild: In manchen Gesellschaften dominiert der
        Stamm des staerkeren Elternteils (gemischte Systeme).
        Verhindert dass immer nur der Mutter-Stamm weitergegeben wird.
        """
        if traits is None:
            if other_parent is not None:
                traits = derive_traits(parent, other_parent)
            else:
                traits = derive_traits(parent)

        # Stamm-Bestimmung: score-gewichtet statt immer Mutter
        if other_parent is not None:
            score_a = max(0.01, getattr(parent, "learning_score", 1.0))
            score_b = max(0.01, getattr(other_parent, "learning_score", 1.0))
            dominant = parent if score_a >= score_b else other_parent
        else:
            dominant = parent

        spawn = Agent.spawn_agent(
            x=x,
            y=y,
            traits=traits,
            generation=parent.generation + 1,
            parent_id=parent.id,
            tribe_id=dominant.tribe_id,
        )
        if not getattr(parent, "physics_v2", False):
            # Plan 4: v2-Kinder erben keine gelernten Ressourcen-Erinnerungen.
            # Gate auf parent.physics_v2 — spawn.physics_v2 wird erst durch das
            # nachgelagerte attach_body() in spawn_agent_from_parent True.
            if parent.memory.resource_memory:
                spawn.memory.resource_memory = parent.memory.resource_memory[-3:]
            if other_parent and other_parent.memory.resource_memory:
                extra = other_parent.memory.resource_memory[-2:]
                for mem in extra:
                    if mem not in spawn.memory.resource_memory:
                        spawn.memory.resource_memory.append(mem)

        if parent.tribe_id is not None:
            spawn.trust[parent.id] = 0.4
        if other_parent is not None:
            spawn.trust[other_parent.id] = 0.4
        return spawn
