from intersection import *
from math import pi,cos,sin,acos,asin,sqrt
import numpy as np
import random

class Arm:
    def __init__(self, base_pos = [0, 0], arms_length = [1, 1, 1], joints_config = [pi/2, 0, 0], radius = 0.05, seed = None):
        self.base_pos = base_pos
        self.arms_length = np.array(arms_length)
        self.radius = radius
        self.current_config = np.array(joints_config)
        self.movable = [True, True, True]
        self.ranges = [(0, pi), (-pi, pi), (-pi, pi)]
        random.seed(seed)
        self.speed = np.array([0.01, 0.01, 0.01]) # rad/tick = 0.2 rad/sec

    def with_constraints(self, movable = [True, True, True], joints_ranges = [(0, pi), (-pi, pi), (-pi, pi)]):
        self.movable = movable
        self.ranges = joints_ranges

    def get_radius(self):
        return self.radius

    def config_to_pos(self, config):
        x0, y0 = self.base_pos

        x1 = self.arms_length[0]*cos(config[0]) + x0
        y1 = self.arms_length[0]*sin(config[0]) + y0

        x2 = self.arms_length[1]*cos(config[0]+config[1]) + x1
        y2 = self.arms_length[1]*sin(config[0]+config[1]) + y1

        x3 = self.arms_length[2]*cos(config[0]+config[1]+config[2]) + x2
        y3 = self.arms_length[2]*sin(config[0]+config[1]+config[2]) + y2

        return [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]

    def goto_random_pos(self):
        final_config = self.get_random_config()
        if final_config is None:
            return None
        self.current_config = final_config
        return goto_from(final_config, self.current_config)

    def goto_from(final_config, current_config):
        config_sign = [x/abs(x) if x != 0 else 0 for x in (final_config-current_config)]
        mvt = np.array([current_config[:]])
        while not mvt[-1] != final_config:
            next_pos = mvt[-1] + config_sign*self.speed
            new_pos = config_sign*np.array([min(config_sign*next_pos[i], config_sign*final_config[i]) for i in range(len(final_config))])
            mvt.append[new_pos[:]]

    def get_random_config(self):
        theta0 = random.uniform(self.ranges[0][0], self.ranges[0][1]) if self.movable[0] else self.current_config[0]
        theta1 = random.uniform(self.ranges[1][0], self.ranges[1][1]) if self.movable[1] else self.current_config[1]
        theta2 = random.uniform(self.ranges[2][0], self.ranges[2][1]) if self.movable[2] else self.current_config[2]
        return np.array([theta0, theta1, theta2])

if __name__ == '__main__':
    import doctest
    doctest.testfile("robot_control_doctest.txt")
