import random

TRAIT_RANGES = {
    "speed": (0.5, 3.0),
    "vision": (2.0, 6.0),
    "curiosity": (0.0, 1.0),
    "aggression": (0.0, 1.0),
    "cooperation": (0.0, 1.0),
    "memory_capacity": (4, 20),
    "sociality": (0.0, 1.0),
    "sense_radius": (3.0, 8.0),
    "efficiency": (0.4, 1.5),
    "fertility": (0.5, 1.5),
    "gestation_efficiency": (0.5, 1.5),
    "plasticity": (0.3, 1.8),
    "memory_retention": (0.75, 0.999),
    "diet_preference": (-1.0, 1.0),
    "social_bandwidth": (0.0, 1.0),
    # 16. Gen (Plan 3b, Spec C5): relative Körperkraft, speist Body(strength=...).
    # ACHTUNG Golden: Draw/Vererbung laufen NUR über ensure_strength_trait /
    # derive_strength (v2-Pfad) — random_traits und der derive_traits-Loop
    # lassen strength aus, sonst verschöbe ein zusätzlicher RNG-Draw den
    # v1-Strom und damit die Golden-Trajectory.
    "strength": (0.1, 0.9),
}

# Selektionsdruck: Trait naeher am Eltern-Wert mutieren mit kleinerem Sigma,
# aber die Richtung der Perturbation ist leicht zum score-staerken Elternteil gezogen.
# Biologisches Vorbild: Mendel'sche Segregation + natuerliche Selektion
# -- erfolgreiche Varianten setzen sich durch, rein zufaellige Drift ist sekundaer.
PERTURBATION_BASE = 0.06  # Grundrauschen
PERTURBATION_SCORE_BIAS = 0.55  # Wie stark das bessere Elternteil das Kind dominiert
STRENGTH_PERTURBATION_SIGMA = (
    0.012  # 0.012-Klasse (Spec C5) — NICHT 0.25 (~25 % der Range/Generation)
)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def random_traits():
    return {
        "speed": random.uniform(*TRAIT_RANGES["speed"]),
        "vision": random.uniform(*TRAIT_RANGES["vision"]),
        "curiosity": random.uniform(*TRAIT_RANGES["curiosity"]),
        "aggression": random.uniform(*TRAIT_RANGES["aggression"]),
        "cooperation": random.uniform(*TRAIT_RANGES["cooperation"]),
        "memory_capacity": random.randint(*TRAIT_RANGES["memory_capacity"]),
        "sociality": random.uniform(*TRAIT_RANGES["sociality"]),
        "sense_radius": random.uniform(*TRAIT_RANGES["sense_radius"]),
        "efficiency": random.uniform(*TRAIT_RANGES["efficiency"]),
        "fertility": random.uniform(*TRAIT_RANGES["fertility"]),
        "gestation_efficiency": random.uniform(*TRAIT_RANGES["gestation_efficiency"]),
        "plasticity": random.uniform(*TRAIT_RANGES["plasticity"]),
        "memory_retention": random.uniform(*TRAIT_RANGES["memory_retention"]),
        "diet_preference": random.uniform(*TRAIT_RANGES["diet_preference"]),
        "social_bandwidth": random.uniform(*TRAIT_RANGES["social_bandwidth"]),
    }


def derive_traits(parent_a, parent_b=None, perturbation=PERTURBATION_BASE):
    """
    Vererbung mit Selektionsdruck:
    - Das Elternteil mit hoeherem learning_score (Proxy fuer Score) dominiert
      das Kind leicht (PERTURBATION_SCORE_BIAS).
    - Perturbation ist gaussisch (nicht uniform) -> seltene grosse Spruenge,
      haeufige kleine Anpassungen (realistischer als uniform).
    - Gauss-Rauschen statt uniform verhindert dass Extreme (0.0, 1.0) uebermaessig
      oft per Zufall produziert werden.
    """
    parent_b = parent_b or parent_a

    # Score-gewichtetes Mischen: besserer Elternteil hat mehr Einfluss
    score_a = max(0.01, getattr(parent_a, "learning_score", 1.0))
    score_b = max(0.01, getattr(parent_b, "learning_score", 1.0))
    w_a = score_a / (score_a + score_b)
    w_b = 1.0 - w_a

    traits = {}
    for k, (lo, hi) in TRAIT_RANGES.items():
        if k == "strength":
            continue  # v2-only (derive_strength); ein gauss-Draw hier würde den v1-RNG-Strom verschieben
        val_a = parent_a.traits[k]
        val_b = parent_b.traits[k]
        # Score-gewichteter Mittelwert als Basis
        base = w_a * val_a + w_b * val_b
        # Gaussisches Rauschen: Sigma skaliert mit Differenz der Eltern (heterozygot -> mehr Variation)
        sigma = perturbation + 0.15 * abs(val_a - val_b)
        noise = random.gauss(0, sigma)
        if k == "memory_capacity":
            base = w_a * val_a + w_b * val_b
            traits[k] = int(clamp(round(base + random.gauss(0, 1.5)), lo, hi))
        elif k in ("vision", "sense_radius"):
            traits[k] = clamp(base + random.gauss(0, 0.25), lo, hi)
        elif k in ("memory_retention",):
            traits[k] = clamp(base + random.gauss(0, 0.012), lo, hi)
        elif k in ("plasticity",):
            traits[k] = clamp(base + random.gauss(0, 0.04), lo, hi)
        else:
            traits[k] = clamp(base + noise, lo, hi)
    return traits


def ensure_strength_trait(traits: dict) -> None:
    """Zieht das strength-Gen (NUR im v2-Pfad aufrufen: attach_body).

    random_traits lässt strength bewusst aus — ein zusätzlicher Draw dort würde
    den v1-RNG-Strom und damit die Golden-Trajectory verschieben. Idempotent.
    """
    if "strength" not in traits:
        traits["strength"] = random.uniform(*TRAIT_RANGES["strength"])


def derive_strength(spawn_traits: dict, parent_a, parent_b=None) -> None:
    """Vererbung des strength-Gens (NUR im v2-Kind-Pfad aufrufen).

    Score-gewichtetes Mittel wie derive_traits, Perturbations-σ 0.012-Klasse
    (Spec C5). Eltern ohne Gen (Alt-Checkpoints) zählen als 0.5 (3a-Default).
    """
    parent_b = parent_b or parent_a
    lo, hi = TRAIT_RANGES["strength"]
    val_a = parent_a.traits.get("strength", 0.5)
    val_b = parent_b.traits.get("strength", 0.5)
    score_a = max(0.01, getattr(parent_a, "learning_score", 1.0))
    score_b = max(0.01, getattr(parent_b, "learning_score", 1.0))
    w_a = score_a / (score_a + score_b)
    base = w_a * val_a + (1.0 - w_a) * val_b
    spawn_traits["strength"] = clamp(base + random.gauss(0, STRENGTH_PERTURBATION_SIGMA), lo, hi)
