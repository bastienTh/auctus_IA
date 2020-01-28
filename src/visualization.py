import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

arm_1_x0 = 0
arm_1_y0 = 1
arm_1_x1 = 2
arm_1_y1 = 3
arm_1_x2 = 4
arm_1_y2 = 5
arm_1_x3 = 6
arm_1_y3 = 7
arm_2_x0 = 8
arm_2_y0 = 9
arm_2_x1 = 10
arm_2_y1 = 11
arm_2_x2 = 12
arm_2_y2 = 13
arm_2_x3 = 14
arm_2_y3 = 15

if len(sys.argv) != 2 :
    print("Usage: %s <csv_file>" % sys.argv[0])
    exit()

x, y = [], []

with open(sys.argv[1], newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    spamreader.__next__()
    for row in spamreader:
        x.append([
            [float(row[arm_1_x0]), float(row[arm_1_x1]), float(row[arm_1_x2]), float(row[arm_1_x3])],
            [float(row[arm_2_x0]), float(row[arm_2_x1]), float(row[arm_2_x2]), float(row[arm_2_x3])]
        ])
        y.append([
            [float(row[arm_1_y0]), float(row[arm_1_y1]), float(row[arm_1_y2]), float(row[arm_1_y3])],
            [float(row[arm_2_y0]), float(row[arm_2_y1]), float(row[arm_2_y2]), float(row[arm_2_y3])]
        ])


# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.005
t = np.arange(0.0, dt*len(x), dt)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 7), ylim=(-3, 3))
ax.grid()

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

lines = [(ax.plot([], [], '-', lw=10, solid_capstyle='round'))[0] for i in range(len(x[0]))]


def init():
    for i in range(len(lines)):
        lines[i].set_data([], [])

    time_text.set_text('')
    return [time_text] + lines.copy()

def animate(i):
    for j in range(len(lines)):
        lines[j].set_data(x[i][j], y[i][j])

    time_text.set_text(time_template % (i*dt))
    return [time_text] + lines.copy()

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x)),
                              interval=1000*dt, blit=True, init_func=init)

print(t[-1])
# ani.save('double_pendulum.mp4', fps=15)
plt.show()

#TODO : change the arm's part frames so that they are not always
#       colinear with the world frame
