from math import hypot


class LinearTarget:
    def __init__(self, init_p, init_v, safe_r):
        self.safe_r = safe_r
        self.init_v = init_v
        self.init_p = init_p

    def evaluate(self, p, t):
        """
            Evaluate safety score
            Basically (distance to target)/(safe radius)
        :param p: position
        :param t: time
        :return:
        """
        return hypot(p[0] - self.init_p[0] - self.init_v[0] * t,
                     p[1] - self.init_p[1] - self.init_v[1] * t) / self.safe_r
