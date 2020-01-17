import numpy as np
from intersection import *
from math import pi,cos,sin,acos,asin,sqrt
import random

class Arm:
    def __init__(self, base_pos = [0, 0], arms_length = [1, 1, 1], joints_config = [pi/2, 0, 0]):
        self.base_pos = base_pos
        self.arms_length = np.array(arms_length)
        self.current_config = np.array(joints_config)

    def get_workspace(self):
        m = sum(self.arms_length)
        return -m, m, 0, m, -pi, pi

    def goto_pos(self, x, y, theta, t):
        x = x  - self.base_pos[0]
        y = y  - self.base_pos[1]
        final_config = self.mgi(x, y, theta)
        if final_config is None:
            return None
        config_delta = final_config - self.current_config
        mvt = [self.current_config + i*config_delta/t for i in range(t+1)]
        self.current_config = final_config
        return mvt

    def mgi(self, x, y, theta):
        # l1 = self.arms_length[0]
        # l2 = self.arms_length[1]
        # l3 = self.arms_length[2]
        #
        # u1 = x-l3*cos(theta)
        # u2 = y-l3*sin(theta)
        # X = u2
        # Y = u1
        # Z = (l1**2-l2**2+u1**2+u2**2)/(2*l1)
        #
        # # Solution 1 ------------------------------------------------
        # # th1
        # c1_1 = ((Y*Z) - X*sqrt(X**2 +Y**2-Z**2)) / (X**2 + Y**2)
        # s1_1 = ((X*Z) + Y*sqrt(X**2 +Y**2-Z**2)) / (X**2 + Y**2)
        # th1_1 = acos(c1_1)
        # if ((sin(th1_1)*s1_1) < 0): # different sign
        #     th1_1 = -th1_1
        # # th2
        # c1plus2_1 = (u1-l1*c1_1)/l2
        # s1plus2_1 = (u2-l1*s1_1)/l2
        # th1plus2_1 = acos(c1plus2_1)
        # if ((sin(th1plus2_1)*s1plus2_1) < 0): # different sign
        #     th1plus2_1 = -th1plus2_1
        # th2_1 = angle_sum(th1plus2_1, -th1_1)
        # # th3
        # th3_1 = angle_sum(angle_sum(theta, -th2_1), -th1_1)
        # print(th1_1,th2_1,th3_1)
        return np.array([random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)])

class Robot:
    """
    """
    def __init__(self, model):
        self.model = model

    def goto_pos(self, x, y, theta, t):
        # We need to account for the passing of time, as a global simulation parameter
        return self.model.goto_pos(x, y, theta, t)

    def goto_random_pos(self):
        dt = 0.05
        x_min, x_max, y_min, y_max, theta_min, theta_max = self.model.get_workspace()
        t_min, t_max = 2/dt, 10/dt
        mvt = None
        while mvt == None:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            theta = random.uniform(theta_min, theta_max)
            t = random.randint(t_min, t_max)
            mvt = self.goto_pos(x, y, theta, t)
        return mvt

    def get_base_pos(self):
        return self.model.base_pos

if __name__ == '__main__':
    import doctest
    doctest.testfile("robot_control_doctest.txt")
