import pygame
from artificial_society.visualization.graphs import draw_line_graph

BG       = (18, 18, 24)
BG2      = (26, 26, 34)
ACCENT   = (80, 140, 220)
SEP      = (50, 50, 65)
TEXT     = (220, 222, 228)
TEXT_DIM = (140, 142, 155)
GREEN    = (90, 200, 110)
YELLOW   = (240, 200, 70)
RED      = (220, 70, 70)
PURPLE   = (180, 100, 220)
BLUE     = (80, 160, 255)

STAT_BAR_W = 100
STAT_BAR_H = 7

DISEASE_ICONS = [
    ('malaria',      'Malaria',      (200, 230,  80)),
    ('dysentery',    'Dysentery',    (150, 100,  50)),
    ('tuberculosis', 'Tuberculosis', (200, 200, 200)),
    ('typhoid',      'Typhoid',      (240, 180,  40)),
    ('scurvy',       'Scurvy',       (220, 100,  30)),
    ('wound_fever',  'Wound Fever',  (220,  40,  40)),
]


def _bar_color(t, invert=False):
    if invert: t = 1.0 - t
    if t < 0.5: return (int(t*2*220), 200, 60)
    return (220, int((1-t)*2*200), 60)


def _draw_stat_bar(screen, x, y, value, lo, hi, invert=False, color=None):
    pygame.draw.rect(screen, (38,40,52), (x, y, STAT_BAR_W, STAT_BAR_H), border_radius=3)
    t    = max(0.0, min(1.0, (value-lo) / max(1e-6, hi-lo)))
    fill = max(2, int(STAT_BAR_W * t))
    c    = color if color else _bar_color(t, invert)
    pygame.draw.rect(screen, c, (x, y, fill, STAT_BAR_H), border_radius=3)


def _label(screen, font, text, x, y, color=TEXT):
    screen.blit(font.render(text, True, color), (x, y))


def _section(screen, font_bold, text, x, y, w):
    screen.blit(font_bold.render(text, True, ACCENT), (x, y))
    pygame.draw.line(screen, ACCENT, (x, y+18), (x+w-16, y+18), 1)
    return y + 24


def _stat_row(screen, small, label, value_str, bar_val, lo, hi, x, y, w,
              invert=False, color=None):
    _label(screen, small, label, x, y, TEXT_DIM)
    _label(screen, small, value_str, x+w-52, y, TEXT)
    _draw_stat_bar(screen, x, y+13, bar_val, lo, hi, invert, color)
    return y + 28


def draw_dashboard(screen, x, y, w, h, stats, tribes, technology, font, font_bold, small, tick=0):
    pygame.draw.rect(screen, BG, (x, y, w, h))
    pygame.draw.line(screen, ACCENT, (x, 0), (x, h), 2)

    last = stats.last or {}
    pad  = x + 10
    yy   = 10

    # Title
    screen.blit(font_bold.render('Artificial Society', True, ACCENT), (pad, yy))
    _label(screen, small, f'v3.3   tick {tick}', pad, yy+18, TEXT_DIM)
    yy += 40
    pygame.draw.line(screen, SEP, (x+6, yy), (x+w-6, yy), 1)
    yy += 8

    # Population
    yy = _section(screen, font_bold, 'POPULATION', pad, yy, w)
    yy += 2
    pop = last.get('population', 0)
    _label(screen, font, f"{pop} agents", pad, yy)

    # Life-stage breakdown bar
    n_c = last.get('n_child', 0)
    n_a = last.get('n_adult', 0)
    n_e = last.get('n_elder', 0)
    total = max(1, n_c + n_a + n_e)
    bar_x, bar_y, bar_w, bar_h2 = pad, yy+17, w-22, 8
    pygame.draw.rect(screen, (38,40,52), (bar_x, bar_y, bar_w, bar_h2), border_radius=3)
    seg_c = int(bar_w * n_c / total)
    seg_a = int(bar_w * n_a / total)
    seg_e = bar_w - seg_c - seg_a
    if seg_c > 0: pygame.draw.rect(screen, (200,230,255), (bar_x,          bar_y, seg_c, bar_h2), border_radius=2)
    if seg_a > 0: pygame.draw.rect(screen, (90, 200,110), (bar_x+seg_c,    bar_y, seg_a, bar_h2))
    if seg_e > 0: pygame.draw.rect(screen, (200,170, 90), (bar_x+seg_c+seg_a, bar_y, seg_e, bar_h2), border_radius=2)
    yy += 28
    # Legend for bar
    for col, lbl, cnt in [((200,230,255),'Child',n_c),((90,200,110),'Adult',n_a),((200,170,90),'Elder',n_e)]:
        pygame.draw.rect(screen, col, (pad, yy, 8, 8), border_radius=2)
        _label(screen, small, f'{lbl} {cnt}', pad+12, yy-1, TEXT_DIM)
        pad += 75
    pad = x + 10
    yy += 14

    details = [
        f"Avg age {last.get('average_age',0):.0f}",
        f"Tribes {last.get('tribes',0)}",
        f"Pregnant {last.get('pregnant',0)}",
        f"Events {last.get('active_events',0)}",
    ]
    for i, d in enumerate(details):
        col = pad if i%2==0 else pad+(w-20)//2
        _label(screen, small, d, col, yy+(i//2)*14, TEXT_DIM)
    yy += 32

    # Vitals
    yy = _section(screen, font_bold, 'VITALS', pad, yy, w)
    yy += 2
    yy = _stat_row(screen, small, 'Hydration',
                   f"{last.get('avg_hydration',0):.1f}/100",
                   last.get('avg_hydration',0), 0, 100, False, BLUE, pad, yy, w-20)
    yy = _stat_row(screen, small, 'Sickness',
                   f"{last.get('avg_sick',0):.1f}/100",
                   last.get('avg_sick',0), 0, 100, True, RED, pad, yy, w-20)
    yy = _stat_row(screen, small, 'Avg reward',
                   f"{last.get('avg_reward',0):.2f}",
                   last.get('avg_reward',0), -2, 3, False, GREEN, pad, yy, w-20)
    yy += 4

    # World
    yy = _section(screen, font_bold, 'WORLD', pad, yy, w)
    yy += 2
    yy = _stat_row(screen, small, 'Food',        f"{last.get('world_food',0):.1f}",        last.get('world_food',0),        0,100,False,GREEN,  pad,yy,w-20)
    yy = _stat_row(screen, small, 'Water',       f"{last.get('world_water',0):.1f}",       last.get('world_water',0),       0,100,False,BLUE,   pad,yy,w-20)
    yy = _stat_row(screen, small, 'Disease',     f"{last.get('world_disease',0):.2f}",     last.get('world_disease',0),     0,  8,True, PURPLE, pad,yy,w-20)
    yy = _stat_row(screen, small, 'Pollution',   f"{last.get('world_pollution',0):.2f}",   last.get('world_pollution',0),   0,  8,True, RED,    pad,yy,w-20)
    yy = _stat_row(screen, small, 'Disturbance', f"{last.get('world_disturbance',0):.2f}", last.get('world_disturbance',0), 0,  5,True, YELLOW, pad,yy,w-20)
    yy += 4
    pygame.draw.line(screen, SEP, (x+6, yy), (x+w-6, yy), 1)
    yy += 6

    # Graphs
    gw, gh = w-22, 46
    for lbl, hist, col in [
        ('Population',  stats.population_history,  (90,190,255)),
        ('Knowledge',   stats.knowledge_history,   (160,220,90)),
        ('Cooperation', stats.cooperation_history, (210,120,240)),
    ]:
        _label(screen, small, lbl, pad, yy, TEXT_DIM)
        yy += 13
        draw_line_graph(screen, pad, yy, gw, gh, hist, col, bg=BG2)
        yy += gh + 7

    pygame.draw.line(screen, SEP, (x+6, yy), (x+w-6, yy), 1)
    yy += 6

    # Disease legend
    _label(screen, small, 'Disease ring colors', pad, yy, TEXT_DIM)
    yy += 15
    for _, dname, dcol in DISEASE_ICONS:
        if yy > h-18: break
        pygame.draw.circle(screen, dcol, (pad+6, yy+6), 5)
        _label(screen, small, dname, pad+16, yy, TEXT_DIM)
        yy += 15

    yy += 2
    if yy < h-70:
        pygame.draw.line(screen, SEP, (x+6, yy), (x+w-6, yy), 1)
        yy += 6
        for rcol, rlabel in [((255,110,170),'Pregnant'),((255,255,255),'Has tool'),
                              ((200,230,255),'Child'),((240,210,100),'Elder crown')]:
            if yy > h-16: break
            pygame.draw.circle(screen, rcol, (pad+6, yy+6), 4)
            _label(screen, small, rlabel, pad+16, yy, TEXT_DIM)
            yy += 14


def draw_heatmap_legend(screen, x, y, font):
    pass


def _stat_row(screen, small, label, value_str, bar_val, lo, hi,
              x, y, w, invert=False, color=None):
    """Labeled stat row: name | value | bar."""
    _label(screen, small, label, x, y, TEXT_DIM)
    _label(screen, small, value_str, x+w-52, y, TEXT)
    _draw_stat_bar(screen, x, y+13, bar_val, lo, hi, invert, color)
    return y + 28
