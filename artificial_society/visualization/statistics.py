from artificial_society.agents.life_stage import STAGE_ADULT, STAGE_CHILD, STAGE_ELDER


class StatisticsTracker:
    def __init__(self):
        self.population_history = []
        self.knowledge_history = []
        self.cooperation_history = []
        self.food_history = []
        self.energy_history = []
        self.last = {}

    def update(self, tick, agents, world, tribes, technology):
        live = [a for a in agents if a.alive]
        pop = len(live)
        avg_age = sum(a.age for a in live) / pop if pop else 0
        avg_coop = sum(a.genes["cooperation"] for a in live) / pop if pop else 0
        known_sites = sum(len(a.memory.resource_memory) for a in live)
        pregnant = sum(1 for a in live if a.pregnant)
        avg_children = sum(a.children for a in live) / pop if pop else 0
        avg_plant = sum(a.plant_eaten for a in live) / pop if pop else 0
        avg_meat = sum(a.meat_eaten for a in live) / pop if pop else 0
        avg_hydration = sum(a.hydration for a in live) / pop if pop else 0
        avg_sick = sum(a.sick for a in live) / pop if pop else 0
        avg_reward = sum(a.last_reward for a in live) / pop if pop else 0
        avg_loss = sum(a.last_loss for a in live) / pop if pop else 0
        # Life-stage counts
        n_child = sum(1 for a in live if getattr(a, "life_stage", STAGE_ADULT) == STAGE_CHILD)
        n_adult = sum(1 for a in live if getattr(a, "life_stage", STAGE_ADULT) == STAGE_ADULT)
        n_elder = sum(1 for a in live if getattr(a, "life_stage", STAGE_ADULT) == STAGE_ELDER)
        avg_energy = sum(a.energy for a in live) / pop if pop else 0
        world_means = world.regional_means()
        self.population_history.append((tick, pop))
        self.knowledge_history.append((tick, known_sites))
        self.cooperation_history.append((tick, avg_coop))
        self.food_history.append((tick, world_means["food"]))
        self.energy_history.append((tick, avg_energy))
        self.population_history = self.population_history[-200:]
        self.knowledge_history = self.knowledge_history[-200:]
        self.cooperation_history = self.cooperation_history[-200:]
        self.food_history = self.food_history[-200:]
        self.energy_history = self.energy_history[-200:]
        self.last = {
            "tick": tick,
            "population": pop,
            "average_age": avg_age,
            "avg_energy": avg_energy,
            "tribes": tribes.count(),
            "technologies": len(technology.capability_map),
            "knowledge": known_sites,
            "cooperation": avg_coop,
            "pregnant": pregnant,
            "avg_children": avg_children,
            "avg_plant": avg_plant,
            "avg_meat": avg_meat,
            "avg_hydration": avg_hydration,
            "avg_sick": avg_sick,
            "avg_reward": avg_reward,
            "avg_loss": avg_loss,
            "world_food": world_means["food"],
            "world_water": world_means["water"],
            "world_pollution": world_means["pollution"],
            "world_fertility": world_means["soil_fertility"],
            "world_capacity": world_means["carrying_capacity"],
            "world_disease": world_means["disease"],
            "world_disturbance": world_means["disturbance"],
            "active_events": world_means["events"],
            # Life-stage breakdown
            "n_child": n_child,
            "n_adult": n_adult,
            "n_elder": n_elder,
        }
