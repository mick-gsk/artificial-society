import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device('cpu')

# Feature layout (57 total):
#   0..3   body state (energy, health, hydration, age)
#   4..14  cell percepts (food, water, temp, danger, disease, soil, pollution,
#          carrying_capacity, moisture, ash, disturbance)
#   15..16 social (nearby count, friends count)
#   17..20 genes (curiosity, aggression, cooperation, sociality)
#   21..30 equipment + episodic extras (tool, trust, resources x3, last_reward,
#          herb_presence, warmth, mat_count, inv_size)
#   31..33 structure features (camp_level, well_level, farm_level)  <-- NEU
#   34..36 causal memory features (3)
#   37..48 episodic memory retrieval (12)
#   49..56 endocrine hormones: cortisol, adrenaline, melatonin, serotonin,
#          dopamine, oxytocin, inflammation, metabolism
#
# IMPORTANT: The brain never receives raw world labels like 'light',
# 'is_night', 'sleep_pressure', or 'disease_level'.  All such information
# reaches the brain ONLY through its hormonal consequences.  The agent
# must learn the correlations on its own.
INPUT_SIZE = 57
HIDDEN_SIZE = 96
ACTION_SIZE = 6
GAMMA = 0.97
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
ACTOR_COEF = 1.0
CRITIC_COEF = 0.5
WORLD_COEF = 0.35
ENTROPY_COEF = 0.004
LEARNING_RATE = 3e-4          # Basis-Lernrate; wird durch plasticity-Gen skaliert
GRAD_CLIP = 1.0
REWARD_CLAMP = 6.0
ROLLOUT_HORIZON = 64
PPO_EPOCHS = 6

# --- Model-Based Planning ---
PLAN_CANDIDATES = 12
PLAN_HORIZON = 3
NOVELTY_WEIGHT = 0.15
VALUE_WEIGHT = 0.50
REWARD_WEIGHT = 0.35

# --- Neuronale Praedisposition durch Vererbung ---
WEIGHT_INHERIT_STRENGTH = 0.55
WEIGHT_MUTATION_SCALE   = 0.018

# --- Imitationslernen (Spiegelneuronen-Analogie) ---
# Wie stark ein beobachteter Erfolgsagent die eigenen Gewichte beeinflusst.
# Sehr klein halten: Imitation ist ein Nudge, kein Klon.
IMITATION_STRENGTH   = 0.05
IMITATION_MUTATION   = 0.005


class RolloutBuffer:
    def __init__(self):
        self.storage = []

    def add(self, item):
        self.storage.append(item)

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)


class Brain(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 action_size=ACTION_SIZE, plasticity: float = 1.0):
        """
        plasticity-Gen (0.3..1.8) skaliert die Lernrate.
        Biologisches Vorbild: Neuronale Plastizitaet variiert zwischen Individuen;
        hochplastische Individuen lernen schneller aber sind weniger stabil.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 160),
            nn.Tanh(),
            nn.Linear(160, 128),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(128, hidden_size)
        self.policy_mean = nn.Linear(hidden_size, action_size)
        self.policy_logstd = nn.Parameter(torch.full((action_size,), -0.45, dtype=torch.float32))
        self.value_head = nn.Linear(hidden_size, 1)
        self.world_fc = nn.Sequential(
            nn.Linear(hidden_size + action_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.next_obs_head = nn.Linear(128, input_size)
        self.reward_head = nn.Linear(128, 1)
        # Individuelle Lernrate basierend auf plasticity-Gen
        # Klammerung auf [0.5x .. 2.5x] verhindert Extremwerte
        effective_lr = LEARNING_RATE * max(0.5, min(2.5, plasticity))
        self.optimizer = optim.Adam(self.parameters(), lr=effective_lr)
        self.rollout = RolloutBuffer()

    # ------------------------------------------------------------------
    # Gewichtsvererbung (Epigenetik-Analogie)
    # ------------------------------------------------------------------
    def inherit_weights_from(self, parent_brain: 'Brain',
                              strength: float = WEIGHT_INHERIT_STRENGTH,
                              mutation_scale: float = WEIGHT_MUTATION_SCALE):
        with torch.no_grad():
            for (_, child_param), (_, parent_param) in zip(
                self.named_parameters(), parent_brain.named_parameters()
            ):
                if child_param.shape != parent_param.shape:
                    continue
                mutation = torch.randn_like(child_param) * mutation_scale
                child_param.copy_(
                    strength * parent_param + (1.0 - strength) * child_param + mutation
                )

    # ------------------------------------------------------------------
    # Imitationslernen (Spiegelneuronen-Hypothese / Bandura)
    # ------------------------------------------------------------------
    def imitate_from(self, model_brain: 'Brain',
                     strength: float = IMITATION_STRENGTH,
                     mutation_scale: float = IMITATION_MUTATION):
        with torch.no_grad():
            for (_, my_param), (_, model_param) in zip(
                self.named_parameters(), model_brain.named_parameters()
            ):
                if my_param.shape != model_param.shape:
                    continue
                noise = torch.randn_like(my_param) * mutation_scale
                my_param.copy_(
                    (1.0 - strength) * my_param + strength * model_param + noise
                )

    def initial_hidden(self):
        return torch.zeros(self.hidden_size, dtype=torch.float32, device=device)

    def forward(self, obs_tensor, hidden_tensor):
        z = self.encoder(obs_tensor)
        next_hidden = self.gru(z, hidden_tensor)
        mean = torch.tanh(self.policy_mean(next_hidden))
        log_std = self.policy_logstd.clamp(-2.0, 0.7).unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        value = self.value_head(next_hidden).squeeze(-1)
        return mean, std, value, next_hidden

    def predict_world(self, hidden_tensor, action_tensor):
        z = self.world_fc(torch.cat([hidden_tensor, action_tensor], dim=-1))
        next_obs = torch.tanh(self.next_obs_head(z))
        reward = self.reward_head(z).squeeze(-1)
        return next_obs, reward

    def evaluate_actions(self, obs_tensor, hidden_tensor, action_tensor):
        mean, std, value, next_hidden = self.forward(obs_tensor, hidden_tensor)
        dist = torch.distributions.Normal(mean, std)
        clipped = torch.clamp(action_tensor, -0.999, 0.999)
        raw_action = 0.5 * torch.log((1 + clipped) / (1 - clipped))
        log_prob = (dist.log_prob(raw_action) - torch.log(1 - clipped.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value, next_hidden

    def imagine_rollout(self, hidden_tensor, action_tensor, horizon=PLAN_HORIZON):
        h = hidden_tensor
        a = action_tensor
        total_score = torch.zeros(a.shape[0], device=device)
        discount = 1.0
        for _ in range(horizon):
            pred_next_obs, pred_reward = self.predict_world(h, a)
            novelty = pred_next_obs.abs().mean(dim=-1)
            enc = self.encoder(pred_next_obs)
            h = self.gru(enc, h)
            next_value = self.value_head(h).squeeze(-1)
            step_score = (REWARD_WEIGHT * pred_reward
                          + VALUE_WEIGHT * next_value
                          + NOVELTY_WEIGHT * novelty)
            total_score = total_score + discount * step_score
            discount *= GAMMA
            mean_next = torch.tanh(self.policy_mean(h))
            log_std = self.policy_logstd.clamp(-2.0, 0.7).unsqueeze(0).expand_as(mean_next)
            std_next = torch.exp(log_std)
            a = torch.tanh(torch.distributions.Normal(mean_next, std_next).rsample())
        return h, total_score

    def plan_action(self, obs_tensor, hidden_tensor, n_candidates=PLAN_CANDIDATES):
        with torch.no_grad():
            mean, std, _, _ = self.forward(obs_tensor, hidden_tensor)
            dist = torch.distributions.Normal(mean, std)
            raw_samples = dist.rsample((n_candidates,))
            action_samples = torch.tanh(raw_samples)
            scores = []
            for i in range(n_candidates):
                a = action_samples[i]
                _, score = self.imagine_rollout(hidden_tensor, a, horizon=PLAN_HORIZON)
                scores.append(score.item())
            best_idx = int(torch.tensor(scores).argmax().item())
            best_action = action_samples[best_idx]
            log_std = self.policy_logstd.clamp(-2.0, 0.7).unsqueeze(0).expand_as(mean)
            clipped = torch.clamp(best_action, -0.999, 0.999)
            raw_best = 0.5 * torch.log((1 + clipped) / (1 - clipped + 1e-8))
            log_prob = (dist.log_prob(raw_best) - torch.log(1 - clipped.pow(2) + 1e-6)).sum(dim=-1)
            return best_action, log_prob, torch.tensor(scores)

    def act(self, features, hidden_state, use_planning=True):
        obs = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        hidden = hidden_state.unsqueeze(0)
        mean, std, value, next_hidden = self.forward(obs, hidden)

        if use_planning:
            action, log_prob, candidate_scores = self.plan_action(obs, hidden)
            _, _, _, next_hidden = self.forward(obs, hidden)
        else:
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            clipped = torch.clamp(action, -0.999, 0.999)
            raw_a = 0.5 * torch.log((1 + clipped) / (1 - clipped + 1e-8))
            log_prob = (dist.log_prob(raw_a) - torch.log(1 - clipped.pow(2) + 1e-6)).sum(dim=-1)

        entropy = torch.distributions.Normal(mean, std).entropy().sum(dim=-1)

        return {
            'obs_tensor': obs.detach(),
            'hidden_in': hidden.detach(),
            'value': value.detach(),
            'next_hidden': next_hidden.squeeze(0).detach(),
            'action_tensor': action.detach(),
            'action_list': action.squeeze(0).detach().tolist(),
            'log_prob': log_prob.detach(),
            'entropy': entropy.detach(),
        }

    def intrinsic_reward(self, hidden_in, action_tensor, next_obs):
        with torch.no_grad():
            pred_next_obs, pred_reward = self.predict_world(hidden_in, action_tensor)
            target = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_err = F.mse_loss(pred_next_obs, target)
            rew_err = pred_reward.abs().mean()
            value = (obs_err + 0.2 * rew_err).clamp(0.0, 2.0)
            return float(value.cpu())

    def store_transition(self, obs_tensor, hidden_in, action_tensor, log_prob, value, reward, done, next_obs):
        self.rollout.add({
            'obs': obs_tensor.detach().squeeze(0),
            'hidden': hidden_in.detach().squeeze(0),
            'action': action_tensor.detach().squeeze(0),
            'log_prob': log_prob.detach().squeeze(0),
            'value': value.detach().squeeze(0),
            'reward': max(-REWARD_CLAMP, min(REWARD_CLAMP, reward)),
            'done': done,
            'next_obs': torch.tensor(next_obs, dtype=torch.float32, device=device),
        })

    def maybe_train(self):
        if len(self.rollout) < ROLLOUT_HORIZON:
            return None
        batch = self.rollout.storage
        obs = torch.stack([item['obs'] for item in batch])
        hid = torch.stack([item['hidden'] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        old_log_probs = torch.stack([item['log_prob'] for item in batch]).view(-1)
        values = torch.stack([item['value'] for item in batch]).view(-1)
        rewards = torch.tensor([item['reward'] for item in batch], dtype=torch.float32, device=device)
        dones = torch.tensor([1.0 if item['done'] else 0.0 for item in batch], dtype=torch.float32, device=device)
        next_obs = torch.stack([item['next_obs'] for item in batch])

        with torch.no_grad():
            _, _, next_values, _ = self.forward(next_obs, hid)
            next_values = next_values.view(-1)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(batch))):
            delta = rewards[t] + GAMMA * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_loss = None
        for _ in range(PPO_EPOCHS):
            new_log_probs, entropy, new_values, next_hidden = self.evaluate_actions(obs, hid, actions)
            ratios = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratios * advantages
            clipped_r = torch.clamp(ratios, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            actor_loss = -torch.min(unclipped, clipped_r).mean()
            critic_loss = F.mse_loss(new_values.view(-1), returns)
            pred_next_obs, pred_reward = self.predict_world(next_hidden, actions)
            world_loss = F.mse_loss(pred_next_obs, next_obs) + F.mse_loss(pred_reward.view(-1), rewards)
            entropy_bonus = entropy.mean()
            loss = ACTOR_COEF * actor_loss + CRITIC_COEF * critic_loss + WORLD_COEF * world_loss - ENTROPY_COEF * entropy_bonus
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())

        self.rollout.clear()
        return last_loss
