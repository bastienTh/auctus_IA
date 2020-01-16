from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from robot_control import *


# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 100, dt)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 7), ylim=(-3, 3))
ax.grid()

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

arms = [Robot(Arm()), Robot(Arm(base_pos=[4, 0]))]
lines = [(ax.plot([], [], '-', lw=10, solid_capstyle='round'))[0] for i in range(len(arms))]
print(lines[0])
x, y = [], []

for arm in arms:
    mvts = []
    while len(mvts) < len(t):
        mvts += arm.goto_random_pos()

    mvts = np.array(mvts[:len(t)])

    x0, y0 = arm.get_base_pos()
    x0 = [x0] * len(mvts)
    y0 = [y0] * len(mvts)

    x1 = cos(mvts[:, 0]) + x0
    y1 = sin(mvts[:, 0]) + y0

    x2 = cos(mvts[:, 1]) + x1
    y2 = sin(mvts[:, 1]) + y1

    x3 = cos(mvts[:, 2]) + x2
    y3 = sin(mvts[:, 2]) + y2

    x.append([x0, x1, x2, x3])
    y.append([y0, y1, y2, y3])

def init():
    for i in range(len(lines)):
        lines[i].set_data([], [])

    time_text.set_text('')
    return [time_text] + lines.copy()


def animate(i):
    for j in range(len(lines)):
        thisx = [x[j][0][i], x[j][1][i], x[j][2][i], x[j][3][i]]
        thisy = [y[j][0][i], y[j][1][i], y[j][2][i], y[j][3][i]]
        lines[j].set_data(thisx, thisy)

    time_text.set_text(time_template % (i*dt))
    return [time_text] + lines.copy()

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(mvts)),
                              interval=1000*dt, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()

#TODO : change the arm's part frames so that they are not always
#       colinear with the world frame
