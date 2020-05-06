from typing import List

from .targets import LinearTarget
from shapely.geometry import LineString, Point


class Environment:
    targets: List[LinearTarget]

    def __init__(self):
        self.targets = []

    def add_ts(self, target: LinearTarget):
        self.targets.append(target)

    def test(self, p, t):
        for ts in self.targets:
            if ts.evaluate(p, t) < 1:
                return False

        return True


class Route:
    def __init__(self, goals, width):
        self.goals = goals
        self.linestring = LineString(goals)
        self.width = width

    def distance(self, p):
        return self.linestring.distance(Point(*p))
