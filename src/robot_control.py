import numpy as np
from math import pi

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
    doctest.testmod()
    # doctest.testfile("robot_doctest.txt")
