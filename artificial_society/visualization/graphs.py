import pygame


def draw_line_graph(screen, x, y, w, h, history, color, bg=(22, 22, 28)):
    """Draw a smooth filled line graph with background panel."""
    # Background
    pygame.draw.rect(screen, bg, (x, y, w, h), border_radius=4)
    pygame.draw.rect(screen, (50, 52, 65), (x, y, w, h), 1, border_radius=4)

    if len(history) < 2:
        return

    vals = list(history[-w:])
    mn = min(vals)
    mx = max(vals)
    span = max(1e-6, mx - mn)

    def px_for(i, v):
        px = x + int(i / max(1, len(vals) - 1) * (w - 2)) + 1
        py = y + h - 2 - int((v - mn) / span * (h - 4))
        return px, py

    # Fill area under curve
    if len(vals) >= 2:
        pts = [px_for(i, v) for i, v in enumerate(vals)]
        poly = [pts[0]] + pts + [(pts[-1][0], y + h - 1), (pts[0][0], y + h - 1)]
        fill_color = tuple(max(0, min(255, c // 3)) for c in color)
        pygame.draw.polygon(screen, fill_color, poly)

    # Line
    pts = [px_for(i, v) for i, v in enumerate(vals)]
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, 2)

    # Latest value dot
    if pts:
        pygame.draw.circle(screen, color, pts[-1], 3)
