import random


class CommunicationSystem:
    def __init__(self, message_size=4):
        self.message_size = message_size

    def emit(self, agent, observation_vector, action=None):
        base = []
        for i in range(self.message_size):
            src = observation_vector[i % len(observation_vector)] if observation_vector else 0.0
            act = action[i % len(action)] if action else 0.0
            noise = random.uniform(-0.1, 0.1)
            mixed = 0.6 * src + 0.3 * act + noise * (1.0 - agent.genes['sociality'])
            base.append(max(-1.0, min(1.0, mixed)))
        agent.message_vector = base
        return base

    def evaluate_message_usefulness(self, sender, receiver, before_food, after_food, before_danger, after_danger):
        score = 0.0
        if after_food > before_food:
            score += 0.12
        if after_danger < before_danger:
            score += 0.12
        if sender.tribe_id is not None and sender.tribe_id == receiver.tribe_id:
            score += 0.03
        return score
