import random


GENE_RANGES = {
    'speed':                (0.5, 3.0),
    'vision':               (2.0, 6.0),
    'curiosity':            (0.0, 1.0),
    'aggression':           (0.0, 1.0),
    'cooperation':          (0.0, 1.0),
    'memory_capacity':      (4, 20),
    'sociality':            (0.0, 1.0),
    'sense_radius':         (3.0, 8.0),
    'efficiency':           (0.4, 1.5),
    'fertility':            (0.5, 1.5),
    'gestation_efficiency': (0.5, 1.5),
    'plasticity':           (0.3, 1.8),
    'memory_retention':     (0.75, 0.999),
    'diet_preference':      (-1.0, 1.0),
    'social_bandwidth':     (0.0, 1.0),
}

# Selektionsdruck: Gene naeher am Eltern-Wert mutieren mit kleinerem Sigma,
# aber die Richtung der Mutation ist leicht zum fitness-staerken Elternteil gezogen.
# Biologisches Vorbild: Mendel'sche Segregation + natuerliche Selektion
# -- erfolgreiche Allele setzen sich durch, rein zufaellige Drift ist sekundaer.
MUTATION_BASE    = 0.06   # Grundrauschen
MUTATION_FITNESS_BIAS = 0.55  # Wie stark das bessere Elternteil das Kind dominiert


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def random_genes():
    return {
        'speed':                random.uniform(*GENE_RANGES['speed']),
        'vision':               random.uniform(*GENE_RANGES['vision']),
        'curiosity':            random.uniform(*GENE_RANGES['curiosity']),
        'aggression':           random.uniform(*GENE_RANGES['aggression']),
        'cooperation':          random.uniform(*GENE_RANGES['cooperation']),
        'memory_capacity':      random.randint(*GENE_RANGES['memory_capacity']),
        'sociality':            random.uniform(*GENE_RANGES['sociality']),
        'sense_radius':         random.uniform(*GENE_RANGES['sense_radius']),
        'efficiency':           random.uniform(*GENE_RANGES['efficiency']),
        'fertility':            random.uniform(*GENE_RANGES['fertility']),
        'gestation_efficiency': random.uniform(*GENE_RANGES['gestation_efficiency']),
        'plasticity':           random.uniform(*GENE_RANGES['plasticity']),
        'memory_retention':     random.uniform(*GENE_RANGES['memory_retention']),
        'diet_preference':      random.uniform(*GENE_RANGES['diet_preference']),
        'social_bandwidth':     random.uniform(*GENE_RANGES['social_bandwidth']),
    }


def inherit_genes(parent_a, parent_b=None, mutation=MUTATION_BASE):
    """
    Vererbung mit Selektionsdruck:
    - Das Elternteil mit hoeherem learning_score (Proxy fuer Fitness) dominiert
      das Kind leicht (MUTATION_FITNESS_BIAS).
    - Mutation ist gaussisch (nicht uniform) -> seltene grosse Spruenge,
      haeufige kleine Anpassungen (realistischer als uniform).
    - Gauss-Rauschen statt uniform verhindert dass Extreme (0.0, 1.0) uebermaessig
      oft per Zufall produziert werden.
    """
    parent_b = parent_b or parent_a

    # Fitness-gewichtetes Mischen: besserer Elternteil hat mehr Einfluss
    score_a  = max(0.01, getattr(parent_a, 'learning_score', 1.0))
    score_b  = max(0.01, getattr(parent_b, 'learning_score', 1.0))
    w_a = score_a / (score_a + score_b)
    w_b = 1.0 - w_a

    genes = {}
    for k, (lo, hi) in GENE_RANGES.items():
        val_a = parent_a.genes[k]
        val_b = parent_b.genes[k]
        # Fitness-gewichteter Mittelwert als Basis
        base  = w_a * val_a + w_b * val_b
        # Gaussisches Rauschen: Sigma skaliert mit Differenz der Eltern (heterozygot -> mehr Variation)
        sigma = mutation + 0.15 * abs(val_a - val_b)
        noise = random.gauss(0, sigma)
        if k == 'memory_capacity':
            base  = w_a * val_a + w_b * val_b
            genes[k] = int(clamp(round(base + random.gauss(0, 1.5)), lo, hi))
        elif k in ('vision', 'sense_radius'):
            genes[k] = clamp(base + random.gauss(0, 0.25), lo, hi)
        elif k in ('memory_retention',):
            genes[k] = clamp(base + random.gauss(0, 0.012), lo, hi)
        elif k in ('plasticity',):
            genes[k] = clamp(base + random.gauss(0, 0.04), lo, hi)
        else:
            genes[k] = clamp(base + noise, lo, hi)
    return genes
