from geographiclib.geodesic import Geodesic

from math import radians, cos, sin, degrees, atan2


class Frame:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def fromwgs(self, lat, lon):
        """

        :param lat:
        :param lon:
        :return: x, y, distance, bearing
        """
        path = Geodesic.WGS84.Inverse(self.lat, self.lon, lat, lon)

        angle = radians(path['azi1'])
        dist = path['s12'] / 1852
        return dist * cos(angle), dist * sin(angle), dist, degrees(angle)

    def towgs(self, x, y):
        """

        :param x:
        :param y:
        :return: lat, lon
        """
        azi1 = degrees(atan2(y, x))
        dist = (x ** 2 + y ** 2) ** .5
        path = Geodesic.WGS84.Direct(self.lat, self.lon, azi1, dist * 1852)
        return path['lat2'], path['lon2']
