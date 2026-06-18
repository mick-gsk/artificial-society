import pygame

try:
    import artificial_society.bootstrap as bootstrap
except Exception:
    bootstrap = None

from artificial_society.simulation import Simulation

if bootstrap is not None:
    try:
        bootstrap.patch_simulation_class(Simulation)
    except Exception:
        pass


def main():
    pygame.init()
    sim = Simulation(width=1200, height=800, grid_w=60, grid_h=40, initial_population=36)
    sim.run()
    pygame.quit()


if __name__ == '__main__':
    main()
