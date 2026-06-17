import pygame


def draw_line_graph(screen, x, y, w, h, points, color):
    pygame.draw.rect(screen, (35, 35, 42), (x, y, w, h))
    pygame.draw.rect(screen, (70, 70, 80), (x, y, w, h), 1)
    if len(points) < 2:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    mapped = []
    for px, py in points:
        mx = x + ((px - min_x) / span_x) * w
        my = y + h - ((py - min_y) / span_y) * h
        mapped.append((mx, my))
    pygame.draw.lines(screen, color, False, mapped, 2)
