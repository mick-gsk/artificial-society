import pygame
from artificial_society.visualization.graphs import draw_line_graph

# Dashboard palette
BG        = (18, 18, 24)
BG2       = (26, 26, 34)
ACCENT    = (80, 140, 220)
SEP       = (50, 50, 65)
TEXT      = (220, 222, 228)
TEXT_DIM  = (140, 142, 155)
GREEN     = (90, 200, 110)
YELLOW    = (240, 200, 70)
RED       = (220, 70, 70)
PURPLE    = (180, 100, 220)

STAT_BAR_W = 80
STAT_BAR_H = 7

# Disease ring colors defined here to avoid circular import with renderer.py
DISEASE_COLORS = {
    'malaria':      (200, 230,  80),
    'dysentery':    (150, 100,  50),
    'tuberculosis': (200, 200, 200),
    'typhoid':      (240, 180,  40),
    'scurvy':       (220, 100,  30),
    'wound_fever':  (220,  40,  40),
}


def _color_for_value(value, low=0.0, high=100.0, invert=False):
    t = max(0.0, min(1.0, (value - low) / max(1.0, high - low)))
    if invert:
        t = 1.0 - t
    if t < 0.5:
        r = int(t * 2 * 230)
        g = 200
    else:
        r = 220
        g = int((1 - t) * 2 * 200)
    return (r, g, 60)


def _draw_stat_bar(screen, x, y, value, low=0.0, high=100.0, invert=False, color=None):
    pygame.draw.rect(screen, (40, 42, 52), (x, y, STAT_BAR_W, STAT_BAR_H), border_radius=3)
    fill = max(2, int(STAT_BAR_W * max(0.0, min(1.0, (value - low) / max(1.0, high - low)))))
    c = color or _color_for_value(value, low, high, invert)
    pygame.draw.rect(screen, c, (x, y, fill, STAT_BAR_H), border_radius=3)


def _label(screen, font, text, x, y, color=TEXT):
    screen.blit(font.render(text, True, color), (x, y))


def _section(screen, font_bold, text, x, y, w):
    screen.blit(font_bold.render(text, True, ACCENT), (x, y))
    pygame.draw.line(screen, ACCENT, (x, y + 18), (x + w - 20, y + 18), 1)
    return y + 24


def draw_dashboard(screen, x, y, w, h, stats, tribes, technology, font, font_bold, small, tick=0):
    pygame.draw.rect(screen, BG, (x, y, w, h))
    pygame.draw.line(screen, ACCENT, (x, 0), (x, h), 2)

    last = stats.last or {}
    pop  = last.get('population', 0)
    yy   = 10

    # Title
    screen.blit(font_bold.render('Artificial Society', True, ACCENT), (x + 10, yy))
    _label(screen, small, f'v3.2   tick {tick}', x + 10, yy + 18, TEXT_DIM)
    yy += 40
    pygame.draw.line(screen, SEP, (x + 8, yy), (x + w - 8, yy), 1)
    yy += 8

    # Population
    yy = _section(screen, font_bold, 'POPULATION', x + 10, yy, w)
    yy += 2
    _label(screen, font, f"{pop} agents", x + 10, yy)
    _label(screen, small, f"Avg age  {last.get('average_age', 0):.0f}", x + 10, yy + 18, TEXT_DIM)
    yy += 38

    # Vitals
    yy = _section(screen, font_bold, 'VITALS', x + 10, yy, w)
    yy += 2
    vitals = [
        ('Hydration', last.get('avg_hydration', 0), 0, 100, False, None),
        ('Sickness',  last.get('avg_sick', 0),       0, 100, True,  RED),
        ('Reward',    last.get('avg_reward', 0),     -2,   3, False, GREEN),
    ]
    for label, val, lo, hi, inv, col in vitals:
        _label(screen, small, label, x + 10, yy, TEXT_DIM)
        _label(screen, small, f'{val:.1f}', x + w - 55, yy, TEXT)
        _draw_stat_bar(screen, x + 10, yy + 13, val, lo, hi, inv, col)
        yy += 26
    yy += 4

    # World
    yy = _section(screen, font_bold, 'WORLD', x + 10, yy, w)
    yy += 2
    world_stats = [
        ('Food',        last.get('world_food', 0),        0, 100, False, GREEN),
        ('Water',       last.get('world_water', 0),       0, 100, False, (80, 160, 255)),
        ('Disease',     last.get('world_disease', 0),     0,  60, True,  PURPLE),
        ('Pollution',   last.get('world_pollution', 0),   0,  60, True,  RED),
        ('Disturbance', last.get('world_disturbance', 0), 0,  60, True,  YELLOW),
    ]
    for label, val, lo, hi, inv, col in world_stats:
        _label(screen, small, label, x + 10, yy, TEXT_DIM)
        _label(screen, small, f'{val:.1f}', x + w - 55, yy, TEXT)
        _draw_stat_bar(screen, x + 10, yy + 13, val, lo, hi, inv, col)
        yy += 26
    yy += 4

    pygame.draw.line(screen, SEP, (x + 8, yy), (x + w - 8, yy), 1)
    yy += 6

    # Graphs
    gw = w - 24
    gh = 50

    _label(screen, small, 'Population', x + 10, yy, TEXT_DIM)
    yy += 14
    draw_line_graph(screen, x + 12, yy, gw, gh, stats.population_history, (90, 190, 255), bg=BG2)
    yy += gh + 8

    _label(screen, small, 'Knowledge', x + 10, yy, TEXT_DIM)
    yy += 14
    draw_line_graph(screen, x + 12, yy, gw, gh, stats.knowledge_history, (160, 220, 90), bg=BG2)
    yy += gh + 8

    _label(screen, small, 'Cooperation', x + 10, yy, TEXT_DIM)
    yy += 14
    draw_line_graph(screen, x + 12, yy, gw, gh, stats.cooperation_history, (210, 120, 240), bg=BG2)
    yy += gh + 10

    pygame.draw.line(screen, SEP, (x + 8, yy), (x + w - 8, yy), 1)
    yy += 8

    # Disease legend
    _label(screen, small, 'Active diseases', x + 10, yy, TEXT_DIM)
    yy += 16
    disease_icons = [
        ('malaria',      'Malaria',      (200, 230,  80)),
        ('dysentery',    'Dysentery',    (150, 100,  50)),
        ('tuberculosis', 'Tuberculosis', (200, 200, 200)),
        ('typhoid',      'Typhoid',      (240, 180,  40)),
        ('scurvy',       'Scurvy',       (220, 100,  30)),
        ('wound_fever',  'Wound Fever',  (220,  40,  40)),
    ]
    for did, dname, dcol in disease_icons:
        if yy > h - 30:
            break
        pygame.draw.circle(screen, dcol, (x + 16, yy + 6), 5)
        _label(screen, small, dname, x + 26, yy, TEXT_DIM)
        yy += 16

    yy += 4
    pygame.draw.line(screen, SEP, (x + 8, yy), (x + w - 8, yy), 1)
    yy += 6

    # Agent ring legend
    ring_legend = [
        ((255, 110, 170), 'Pregnant'),
        ((255, 255, 255), 'Has tool'),
        ((100, 180, 255), 'Male dot'),
        ((255, 140, 180), 'Female dot'),
    ]
    for rcol, rlabel in ring_legend:
        if yy > h - 18:
            break
        pygame.draw.circle(screen, rcol, (x + 16, yy + 6), 4)
        _label(screen, small, rlabel, x + 26, yy, TEXT_DIM)
        yy += 15


def draw_heatmap_legend(screen, x, y, font):
    """Legacy stub — kept for backward compat if anything still calls it."""
    pass
