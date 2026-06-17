import math


class WeatherSystem:
    def __init__(self):
        self.temperature_shift = 0.0
        self.wind = 0.0
        self.storm_risk = 0.0

    def update(self, world, season_state, tick):
        self.temperature_shift = math.sin(tick * 0.01) * 4
        self.wind = abs(math.sin(tick * 0.014)) * 8
        self.storm_risk = max(0.0, math.sin(tick * 0.008 + 1.2))
        rain_map = 0.5 + 0.5 * math.sin(tick * 0.01 + season_state['phase'])
        return {
            'temperature_shift': self.temperature_shift,
            'wind': self.wind,
            'storm_risk': self.storm_risk,
            'rain_map': rain_map,
        }
