import pygame


def _to_float(v):
    """Extract scalar from v.
    Handles: plain int/float, (tick, value) tuples, or any sequence.
    History entries are stored as (tick, value) — we want index 1.
    """
    if isinstance(v, (int, float)):
        return float(v)
    try:
        # (tick, value) tuple — value is always the second element
        return float(v[1])
    except (TypeError, IndexError):
        return 0.0


def draw_line_graph(screen, x, y, w, h, history, color, bg=(22, 22, 28)):
    """Draw a smooth filled line graph with a background panel."""
    pygame.draw.rect(screen, bg, (x, y, w, h), border_radius=4)
    pygame.draw.rect(screen, (50, 52, 65), (x, y, w, h), 1, border_radius=4)

    if len(history) < 2:
        return

    vals = [_to_float(v) for v in list(history)[-w:]]
    mn = min(vals)
    mx = max(vals)
    span = max(1e-6, mx - mn)

    def px_for(i, v):
        px = x + int(i / max(1, len(vals) - 1) * (w - 2)) + 1
        py = y + h - 2 - int((v - mn) / span * (h - 4))
        return px, py

    pts = [px_for(i, v) for i, v in enumerate(vals)]

    # Filled area under curve
    poly = [pts[0]] + pts + [(pts[-1][0], y + h - 1), (pts[0][0], y + h - 1)]
    fill_color = tuple(max(0, min(255, c // 3)) for c in color)
    pygame.draw.polygon(screen, fill_color, poly)

    # Line
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, 2)

    # Latest value dot + label
    if pts:
        pygame.draw.circle(screen, color, pts[-1], 3)

    # Min/max labels
    try:
        small = pygame.font.SysFont('segoeui', 10)
        screen.blit(small.render(f'{mx:.0f}', True, (160, 162, 170)), (x + 2, y + 1))
        screen.blit(small.render(f'{mn:.0f}', True, (160, 162, 170)), (x + 2, y + h - 12))
        screen.blit(small.render(f'{vals[-1]:.0f}', True, color), (pts[-1][0] + 3, pts[-1][1] - 6))
    except Exception:
        pass
