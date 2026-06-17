import math


class SeasonCycle:
    names = ['spring', 'summer', 'autumn', 'winter']

    def update(self, tick):
        phase = (tick / 900.0) * 2 * math.pi
        idx = int((tick // 900) % 4)
        season = self.names[idx]
        food_factor = {
            'spring': 1.35,
            'summer': 1.1,
            'autumn': 0.95,
            'winter': 0.55,
        }[season]
        temperature_shift = {
            'spring': 2,
            'summer': 7,
            'autumn': -1,
            'winter': -8,
        }[season]
        return {
            'name': season,
            'phase': phase,
            'food_factor': food_factor,
            'temperature_shift': temperature_shift,
        }
