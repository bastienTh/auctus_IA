from intersection import *
from math import pi,cos,sin,acos,asin,sqrt
import numpy as np
import random

class Arm:
    def __init__(self, base_pos = [0, 0], arms_length = [1, 1, 1], joints_config = [pi/2, 0, 0], seed = None):
        self.base_pos = base_pos
        self.arms_length = np.array(arms_length)
        self.current_config = np.array(joints_config)
        self.movable = [True, True, True]
        self.ranges = [(0, pi), (-pi, pi), (-pi, pi)]
        random.seed(seed)

    def with_constraints(self, movable = [True, True, True], joints_ranges = [(0, pi), (-pi, pi), (-pi, pi)]):
        self.movable = movable
        self.ranges = joints_ranges

    def config_to_pos(self, config):
        x0, y0 = self.base_pos

        x1 = self.arms_length[0]*cos(config[0]) + x0
        y1 = self.arms_length[0]*sin(config[0]) + y0

        x2 = self.arms_length[1]*cos(config[1]) + x1
        y2 = self.arms_length[1]*sin(config[1]) + y1

        x3 = self.arms_length[2]*cos(config[2]) + x2
        y3 = self.arms_length[2]*sin(config[2]) + y2

        return [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]

    def goto_random_pos(self, t):
        x = x - self.base_pos[0]
        y = y - self.base_pos[1]
        final_config = self.get_random_config()
        if final_config is None:
            return None
        config_delta = final_config - self.current_config
        mvt = [config_to_pos(self.current_config + i*config_delta/t) for i in range(t+1)]
        self.current_config = final_config
        return mvt

    def get_random_config(self):
        theta0 = random.uniform(self.ranges[0][0], self.ranges[0][1]) if self.movable[0] else self.current_config[0]
        theta1 = random.uniform(self.ranges[1][0], self.ranges[1][1]) if self.movable[1] else self.current_config[1]
        theta2 = random.uniform(self.ranges[2][0], self.ranges[2][1]) if self.movable[2] else self.current_config[2]
        return np.array([theta0, theta1, theta2])

if __name__ == '__main__':
    import doctest
    doctest.testfile("robot_control_doctest.txt")
