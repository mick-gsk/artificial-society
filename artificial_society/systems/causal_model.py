"""Causal Model der Physik v2 (Plan 3b, Spec C4) — torch-Neuimplementierung.

Verteilungs-Vorwärtsprädiktor über Objekt-Eigenschaften: aus
detach(GRU-Zustand 96) ⊕ verkörpertem Aktions-Vektor (22: Verben+Effort+Queries)
⊕ detach(Slot-Embedding 32 des gewählten Ziels) werden (μ, log σ) über die 17
Slot-Features DESSELBEN Objekts im nächsten Tick prädiziert (Identitäts-Tracking
aus perception_v2; bei strike-Zerstörung: massereichstes Fragment; aus der
Wahrnehmung gefallen: maskiert, kein Loss — beides regelt agent.py).

Der detach ist entschieden (Spec C4): der eigene Optimizer (LR × Plastizitäts-
Gen) trainiert NICHT durch GRU/Encoder hindurch — das Causal Model liest die
Repräsentation, es formt sie nicht (Repräsentations-Dynamik bleibt bei PPO).

Epistemische Neugier = max(0, mean(z²) − 1) mit z = (x − μ)/σ — Hinge AUSSEN:
für ein perfekt kalibriertes Modell ist E[mean(z²)] = 1, der Hinge schickt den
Term gegen 0; ein Hinge innen würde dauerhaft ≈ 0.48 zahlen (Rausch-Rente).

Ideen (err_ema, Bucket-Counts) aus archive/artificial_society/systems/
causal_model.py — der numpy/12-Prop-Code selbst ist strukturell nicht
übernehmbar. Dies ist KEIN registriertes Sim-System, sondern ein
Pro-Agent-Modul (Besitz: agent.causal_model, nur physics_v2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

CAUSAL_HIDDEN_IN = 96  # detach(gru_h)
CAUSAL_ACTION_DIMS = 22  # 5 Verben + 1 Effort + 8 Target-Query + 8 Tool-Query
CAUSAL_EMBED_DIMS = 32  # detach(Slot-Embed des gewählten Ziels)
CAUSAL_INPUT = CAUSAL_HIDDEN_IN + CAUSAL_ACTION_DIMS + CAUSAL_EMBED_DIMS  # 150
CAUSAL_TARGET = 17  # Slot-Features des getrackten Objekts im nächsten Tick
CAUSAL_LR = 3e-4  # × Plastizitäts-Gen (geklemmt wie Brain: 0.5..2.5)
LOG_SIGMA_FLOOR = -2.0  # Spec C4: früh ist σ winzig → z² kann 100+ erreichen
LOG_SIGMA_CEIL = 2.0
ERR_EMA_ALPHA = 0.1  # Logging-Signal (Archive-Idee)
CAUSAL_BETA = 1.0  # Hinge-Epistemik Regularizer: Loss = NLL + β*max(0,mean(z²)-1)


class CausalModelV2(nn.Module):
    def __init__(self, plasticity: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CAUSAL_INPUT, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, CAUSAL_TARGET)
        self.logsigma_head = nn.Linear(64, CAUSAL_TARGET)
        # logsigma mit Null-Initialisierung
        nn.init.zeros_(self.logsigma_head.weight)
        nn.init.zeros_(self.logsigma_head.bias)
        lr = CAUSAL_LR * max(0.5, min(2.5, float(plasticity)))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.err_ema = 1.0

    def forward(self, x):
        z = self.net(x)
        mu = self.mu_head(z)
        log_sigma = self.logsigma_head(z).clamp(LOG_SIGMA_FLOOR, LOG_SIGMA_CEIL)
        return mu, log_sigma

    def observe(self, gru_h, action22, slot_embed, target17) -> dict:
        """Ein Beobachtungs-Schritt: Epistemik VOR dem Update (Überraschung des
        aktuellen Modells), dann NLL+Hinge-Gradientenschritt, dann err_ema."""
        x = torch.cat(
            [
                gru_h.detach().reshape(-1).float(),
                action22.detach().reshape(-1).float(),
                slot_embed.detach().reshape(-1).float(),
            ]
        ).unsqueeze(0)
        target = target17.detach().reshape(1, CAUSAL_TARGET).float()

        with torch.no_grad():
            mu, log_sigma = self.forward(x)
            z2 = ((target - mu) / log_sigma.exp()).pow(2)
            epistemic = max(0.0, float(z2.mean()) - 1.0)  # Hinge AUSSEN (Spec C4)
            mean_abs = float((target - mu).abs().mean())

        mu, log_sigma = self.forward(x)
        z2_tensor = ((target - mu) / log_sigma.exp()).pow(2)
        nll = (0.5 * z2_tensor + log_sigma).mean()
        hinge = torch.clamp(z2_tensor.mean() - 1.0, min=0.0)
        loss = nll + CAUSAL_BETA * hinge
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.err_ema = (1.0 - ERR_EMA_ALPHA) * self.err_ema + ERR_EMA_ALPHA * mean_abs
        return {
            "nll": float(nll.detach()),
            "epistemic": epistemic,
            "mean_abs_err": mean_abs,
        }
