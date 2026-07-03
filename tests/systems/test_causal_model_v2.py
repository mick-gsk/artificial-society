"""D3 Causal-Model: NLL sinkt deterministisch, σ wächst verrauscht, Hinge-Epistemik → 0."""

from __future__ import annotations

import math

import torch

from artificial_society.systems.causal_model import (
    CAUSAL_INPUT,
    CAUSAL_LR,
    LOG_SIGMA_FLOOR,
    CausalModelV2,
)


def _fixe_eingabe():
    torch.manual_seed(0)
    return torch.randn(96), torch.rand(22) * 2 - 1, torch.randn(32)


def test_formen_und_sigma_floor():
    torch.manual_seed(0)
    model = CausalModelV2()
    assert CAUSAL_INPUT == 96 + 22 + 32 == 150
    mu, log_sigma = model.forward(torch.zeros(1, 150))
    assert mu.shape == (1, 17) and log_sigma.shape == (1, 17)
    assert torch.all(log_sigma >= LOG_SIGMA_FLOOR)  # log σ ≥ −2 (Spec C4)


def test_nll_sinkt_auf_deterministischer_sequenz():
    torch.manual_seed(1)
    model = CausalModelV2()
    h, a, e = _fixe_eingabe()
    ziel = torch.rand(17)
    erste = model.observe(h, a, e, ziel)["nll"]
    for _ in range(300):
        letzte = model.observe(h, a, e, ziel)["nll"]
    assert letzte < erste, "NLL muss auf deterministischer Sequenz sinken"


def test_sigma_waechst_auf_verrauschter_sequenz():
    torch.manual_seed(2)
    det = CausalModelV2()
    noisy = CausalModelV2()
    noisy.load_state_dict(det.state_dict())
    h, a, e = _fixe_eingabe()
    ziel = torch.full((17,), 0.5)
    gen = torch.Generator().manual_seed(7)
    for _ in range(400):
        det.observe(h, a, e, ziel)
        rausch = ziel + torch.randn(17, generator=gen) * 0.3
        noisy.observe(h, a, e, rausch)
    x = torch.cat([h, a, e]).unsqueeze(0)
    _, ls_det = det.forward(x)
    _, ls_noisy = noisy.forward(x)
    assert ls_noisy.mean() > ls_det.mean(), "σ muss auf verrauschtem Prozess wachsen"


def test_epistemik_konvergiert_auf_chi2_plateau_novelty_zahlt():
    """D3 (b): auskonvergierter, rein aleatorischer Prozess zahlt fast nichts —
    ewiges Steineschlagen als Neugier-Farm ist zu (Hinge AUSSEN).

    Statistik-Korrektur (Task-12-Review): mean(z²) ist bei 17 Freiheitsgraden
    chi²-verteilt (rechtsschief) — E[max(0, chi²(17)/17 − 1)] ≈ 0.13 > 0. Die
    Epistemik konvergiert auf ein kleines positives Plateau, nicht exakt 0; die
    ersten ~200 Schritte sind Einschwingphase (σ von Init ≈1 auf 0.2). Geprüft
    wird: (1) Plateau klein, (2) σ kalibriert auf die wahre Rauschamplitude
    (fängt Loss-Kontamination wie NLL+Hinge ab, die σ systematisch verzerrt),
    (3) echte Novelty zahlt weit über dem Plateau, (4) Hinge: nie negativ."""
    torch.manual_seed(3)
    model = CausalModelV2()
    h, a, e = _fixe_eingabe()
    gen = torch.Generator().manual_seed(11)
    epistemik = []
    for _ in range(2000):
        rausch = torch.full((17,), 0.5) + torch.randn(17, generator=gen) * 0.2
        epistemik.append(model.observe(h, a, e, rausch.clamp(0.0, 1.0))["epistemic"])
    spaet = sum(epistemik[-500:]) / 500
    assert spaet < 0.25, (
        f"konvergierte Epistemik muss klein sein (chi²-Plateau ≈ 0.13), ist {spaet}"
    )
    assert all(v >= 0.0 for v in epistemik)  # Hinge: nie negativ

    x = torch.cat([h, a, e]).unsqueeze(0)
    _, log_sigma = model.forward(x)
    assert abs(float(log_sigma.mean()) - math.log(0.2)) < 0.15, (
        f"σ muss auf die wahre Rauschamplitude kalibrieren, log σ = {float(log_sigma.mean())}"
    )

    novel = model.observe(h, a, e, torch.full((17,), 2.0))["epistemic"]
    assert novel > 1.0, f"Novelty muss weit über dem Plateau zahlen, ist {novel}"


def test_lr_skaliert_mit_plastizitaets_gen():
    torch.manual_seed(4)
    lr = CausalModelV2(plasticity=1.8).optimizer.param_groups[0]["lr"]
    assert lr == CAUSAL_LR * 1.8
    lr_geklemmt = CausalModelV2(plasticity=9.0).optimizer.param_groups[0]["lr"]
    assert lr_geklemmt == CAUSAL_LR * 2.5


def test_err_ema_wird_gepflegt():
    torch.manual_seed(5)
    model = CausalModelV2()
    h, a, e = _fixe_eingabe()
    assert model.err_ema == 1.0
    model.observe(h, a, e, torch.rand(17))
    assert model.err_ema != 1.0  # Logging-Signal (Archive-Idee)
