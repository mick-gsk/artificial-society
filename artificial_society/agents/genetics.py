import random


GENE_RANGES = {
    'speed': (0.5, 3.0),
    'vision': (2.0, 6.0),
    'curiosity': (0.0, 1.0),
    'aggression': (0.0, 1.0),
    'cooperation': (0.0, 1.0),
    'memory_capacity': (4, 20),
    'sociality': (0.0, 1.0),
    'sense_radius': (3.0, 8.0),
    'efficiency': (0.4, 1.5),
    'fertility': (0.5, 1.5),
    'gestation_efficiency': (0.5, 1.5),
    'plasticity': (0.3, 1.8),
    'memory_retention': (0.75, 0.999),
    'diet_preference': (-1.0, 1.0),
    'social_bandwidth': (0.0, 1.0),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def random_genes():
    return {
        'speed': random.uniform(*GENE_RANGES['speed']),
        'vision': random.uniform(*GENE_RANGES['vision']),
        'curiosity': random.uniform(*GENE_RANGES['curiosity']),
        'aggression': random.uniform(*GENE_RANGES['aggression']),
        'cooperation': random.uniform(*GENE_RANGES['cooperation']),
        'memory_capacity': random.randint(*GENE_RANGES['memory_capacity']),
        'sociality': random.uniform(*GENE_RANGES['sociality']),
        'sense_radius': random.uniform(*GENE_RANGES['sense_radius']),
        'efficiency': random.uniform(*GENE_RANGES['efficiency']),
        'fertility': random.uniform(*GENE_RANGES['fertility']),
        'gestation_efficiency': random.uniform(*GENE_RANGES['gestation_efficiency']),
        'plasticity': random.uniform(*GENE_RANGES['plasticity']),
        'memory_retention': random.uniform(*GENE_RANGES['memory_retention']),
        'diet_preference': random.uniform(*GENE_RANGES['diet_preference']),
        'social_bandwidth': random.uniform(*GENE_RANGES['social_bandwidth']),
    }


def inherit_genes(parent_a, parent_b=None, mutation=0.08):
    parent_b = parent_b or parent_a
    genes = {}
    for k, (lo, hi) in GENE_RANGES.items():
        if k == 'memory_capacity':
            base = round((parent_a.genes[k] + parent_b.genes[k]) / 2 + random.uniform(-2, 2))
            genes[k] = int(clamp(base, lo, hi))
        elif k in ('vision', 'sense_radius'):
            base = (parent_a.genes[k] + parent_b.genes[k]) / 2 + random.uniform(-0.3, 0.3)
            genes[k] = clamp(base, lo, hi)
        elif k in ('memory_retention',):
            base = (parent_a.genes[k] + parent_b.genes[k]) / 2 + random.uniform(-0.015, 0.015)
            genes[k] = clamp(base, lo, hi)
        elif k in ('plasticity',):
            base = (parent_a.genes[k] + parent_b.genes[k]) / 2 + random.uniform(-0.05, 0.05)
            genes[k] = clamp(base, lo, hi)
        else:
            base = (parent_a.genes[k] + parent_b.genes[k]) / 2 + random.uniform(-mutation, mutation)
            genes[k] = clamp(base, lo, hi)
    return genes
