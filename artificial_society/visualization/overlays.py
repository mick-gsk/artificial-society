import pygame
from artificial_society.visualization.graphs import draw_line_graph


def draw_text(screen, font, text, x, y, color=(230, 230, 230)):
    screen.blit(font.render(text, True, color), (x, y))


def draw_dashboard(screen, x, y, w, h, stats, tribes, technology, font, small):
    pygame.draw.rect(screen, (24, 24, 28), (x, y, w, h))
    pygame.draw.line(screen, (60, 60, 70), (x, 0), (x, h), 2)
    draw_text(screen, font, 'Artificial Society v3.0', x + 10, 10)
    last = stats.last or {}
    lines = [
        f"Population: {last.get('population', 0)}",
        f"Avg age: {last.get('average_age', 0):.1f}",
        f"Hydration: {last.get('avg_hydration', 0):.1f}",
        f"Sickness: {last.get('avg_sick', 0):.1f}",
        f"Reward: {last.get('avg_reward', 0):.2f}",
        f"Loss: {last.get('avg_loss', 0):.2f}",
        f"Events: {last.get('active_events', 0)}",
        f"World food: {last.get('world_food', 0):.1f}",
        f"Water: {last.get('world_water', 0):.1f}",
        f"Pollution: {last.get('world_pollution', 0):.1f}",
        f"Disease: {last.get('world_disease', 0):.1f}",
        f"Disturbance: {last.get('world_disturbance', 0):.1f}",
    ]
    yy = 40
    for line in lines:
        draw_text(screen, font, line, x + 10, yy)
        yy += 19
    draw_text(screen, font, 'Population over time', x + 10, 270)
    draw_line_graph(screen, x + 10, 295, w - 20, 58, stats.population_history, (90, 190, 255))
    draw_text(screen, font, 'Knowledge over time', x + 10, 370)
    draw_line_graph(screen, x + 10, 395, w - 20, 58, stats.knowledge_history, (180, 220, 90))
    draw_text(screen, font, 'Cooperation over time', x + 10, 470)
    draw_line_graph(screen, x + 10, 495, w - 20, 58, stats.cooperation_history, (220, 130, 230))
    draw_text(screen, font, 'Legend', x + 10, 575)
    legend = [
        ('Pregnant ring', (255, 110, 170)),
        ('Tool ring', (250, 250, 250)),
        ('Sick outline', (210, 60, 60)),
        ('Event haze', (240, 160, 60)),
    ]
    yy = 600
    for label, color in legend:
        pygame.draw.circle(screen, color, (x + 18, yy + 7), 5)
        draw_text(screen, small, label, x + 30, yy, (200, 200, 200))
        yy += 18


def draw_heatmap_legend(screen, x, y, font):
    lines = [
        'v3.0 reduces hard-coded path scoring and adds disturbances.',
        'Orange haze = disturbance, red haze = disease/pollution.',
        'Agents act via primitive control and curiosity-augmented learning.',
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (170, 170, 170)), (x, y + i * 14))
