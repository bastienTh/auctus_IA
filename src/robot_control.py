import numpy as np
from intersection import *
from math import pi,cos,sin,acos,asin,sqrt

class Arm:
    def __init__(self, joints_config = [pi, pi, pi], arms_length = [1, 1, 1]):
        self.current_config = np.array(joints_config)
        self.arms_length = np.array(arms_length)

    def goto_pos(self, x, y, theta, t):
        final_config = self.mgi(x, y, theta)
        config_delta = final_config - self.current_config
        mvt = [current_config + i*config_delta/t for i in range(t+1)]
        self.current_config = final_config
        return mvt

    def mgi(self, x, y, theta):
        l1 = self.arms_length[0]
        l2 = self.arms_length[1]
        l3 = self.arms_length[2]

        u1 = x-l3*cos(theta)
        u2 = y-l3*sin(theta)
        X = u2
        Y = u1
        Z = (l1**2-l2**2+u1**2+u2**2)/(2*l1)

        # Solution 1 ------------------------------------------------
        # th1
        c1_1 = ((Y*Z) - X*sqrt(X**2 +Y**2-Z**2)) / (X**2 + Y**2)
        s1_1 = ((X*Z) + Y*sqrt(X**2 +Y**2-Z**2)) / (X**2 + Y**2)
        th1_1 = acos(c1_1)
        if ((sin(th1_1)*s1_1) < 0): # different sign
            th1_1 = -th1_1
        # th2
        c1plus2_1 = (u1-l1*c1_1)/l2
        s1plus2_1 = (u2-l1*s1_1)/l2
        th1plus2_1 = acos(c1plus2_1)
        if ((sin(th1plus2_1)*s1plus2_1) < 0): # different sign
            th1plus2_1 = -th1plus2_1
        th2_1 = angle_sum(th1plus2_1, -th1_1)
        # th3
        th3_1 = angle_sum(angle_sum(theta, -th2_1), -th1_1)
        print(th1_1,th2_1,th3_1)
        


class Robot:
    """
    """
    def __init__(self, model):
        self.model = model

    def goto_pos(self, x, y, theta, t):
        # We need to account for the passing of time, as a global simulation parameter
        return self.model.goto_pos(x, y, theta, t)

if __name__ == '__main__':
    import doctest
    doctest.testfile("robot_control_doctest.txt")
    