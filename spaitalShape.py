from shapely.geometry import Polygon


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def xy(self):
        return (self.x, self.y)

    def yx(self):
        return (self.y, self.x)


class OdPair:
    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination


class Box:
    def __init__(self, lonMin, lonMax, latMin, latMax):
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax

    def polygon(self):
        x1, x2, y1, y2 = self.lonMin, self.lonMax, self.latMin, self.latMax
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
