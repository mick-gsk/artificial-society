"""
Day/Night Cycle
---------------
TICKS_PER_DAY = 240  (one full day)

Phases (0..1 = fraction of day):
  0.00-0.08  dawn     -- low light, cool, low danger
  0.08-0.50  day      -- full light, normal
  0.50-0.58  dusk     -- dimming
  0.58-1.00  night    -- darkness, high predator danger, sleep pressure

Light is a float 0..1 that drives:
  - Foraging efficiency (agent.py)
  - Predator danger bonus on cells (world)
  - Fire/warmth value bonus at night

Sleep pressure builds during night, drains cognitive performance
(encoded as reduced Brain reward scaling and higher loss).
"""
import math

TICKS_PER_DAY = 240


def day_phase(tick: int) -> dict:
    """
    Returns a state dict for the current tick:
      phase   : 'dawn' | 'day' | 'dusk' | 'night'
      light   : float 0..1
      danger_mult : float -- night predators
      sleep_pressure : float 0..1  (builds through night)
    """
    t = (tick % TICKS_PER_DAY) / TICKS_PER_DAY  # 0..1

    # Smooth light curve: sunrise at t=0.08, sunset at t=0.58
    # Use a cosine to get natural ramp
    if t < 0.08:
        # pre-dawn: ramp from 0.1 to 0.4
        light = 0.1 + 0.3 * (t / 0.08)
        phase = 'dawn'
    elif t < 0.50:
        # full day: sine peak at t=0.29
        angle = math.pi * (t - 0.08) / 0.42
        light = 0.4 + 0.6 * math.sin(angle)
        phase = 'day'
    elif t < 0.58:
        # dusk: ramp down
        light = 0.4 * (1.0 - (t - 0.50) / 0.08)
        phase = 'dusk'
    else:
        # night
        light = 0.02 + 0.08 * math.cos(math.pi * (t - 0.58) / 0.42)
        phase = 'night'

    # Night danger multiplier
    danger_mult = 1.0 if phase in ('day', 'dawn') else (1.6 if phase == 'night' else 1.2)

    # Sleep pressure: accumulates 0->1 over the night half, resets during day
    if phase == 'night':
        night_progress = (t - 0.58) / 0.42
        sleep_pressure = min(1.0, night_progress * 1.4)
    elif phase == 'dawn':
        sleep_pressure = max(0.0, 1.0 - t / 0.08)
    else:
        sleep_pressure = 0.0

    return {
        'phase': phase,
        'light': round(light, 4),
        'danger_mult': danger_mult,
        'sleep_pressure': round(sleep_pressure, 4),
        't': round(t, 4),
    }
