import numpy as np
from math import pi
import random

class Arm:
    def __init__(self, base_pos = [0, 0], arms_length = [1, 1, 1], joints_config = [pi, pi, pi]):
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
        return np.array([random.uniform(-pi, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)])

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
    doctest.testmod()
    # doctest.testfile("robot_doctest.txt")
