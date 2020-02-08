import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rn
import pandas as pd
import tensorflow as tf
import seaborn as sn
import cv2
import os
import glob
import json
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from IPython.display import display
from load_data import load_data
# from parameters import *

rn.seed(12345)

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

if len(sys.argv) != 3 :
    print("Usage: %s <csv_file> <model.h5> <TIME_WINDOW_SIZE>" % sys.argv[0])
    exit()


x, y= [], []
x_min, x_max, y_min, y_max = 0,0,0,0

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
        x_min = min(x_min, min(min(x[-1][0]),min(x[-1][1])))
        x_max = max(x_max, max(max(x[-1][0]),max(x[-1][1])))
        y_min = min(y_min, min(min(y[-1][0]),min(y[-1][1])))
        y_max = max(y_max, max(max(y[-1][0]),max(y[-1][1])))


# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.01
t = np.arange(0.0, dt*len(x), dt)

### ===============================================================
###                  LOADING THE TRAINED IA
### ===============================================================
model = tf.keras.models.load_model(sys.argv[2])

### ===============================================================
###                  LOADING THE SCENARIO
### ===============================================================
Xs, ys, classes = load_data(path=sys.argv[1], shuffle=False, full_path=True)
classification=['0','1','2','3']
scores = []

for i in range(TIME_WINDOW_SIZE):
    scores.append([0.,0.,0.,0.])

for i in range(TIME_WINDOW_SIZE, len(Xs)-1):
    pred = model.predict(np.array([Xs[i]])).tolist()
    for j in range(i,i+TIME_WINDOW_SIZE):
        scores.append(pred[0])
    
print("SCORE SIZE ", len(scores))
print("x SIZE ", len(x))

# for i in range(TIME_WINDOW_SIZE,len(Xs)):
#     for j in range(i,i+TIME_WINDOW_SIZE):
#         scores.append([rn.uniform(0,1),rn.uniform(0,1),rn.uniform(0,1),rn.uniform(0,1)])

# print(scores)

def plot_student_results(classification, scores):
    #  create the figure
    fig = plt.figure()
    # fig, ax_robot = plt.subplots(autoscale_on=True, figsize=(9, 7))
    ax_robot = fig.add_subplot(111, autoscale_on=False, aspect='equal', xlim=(x_min-0.5, x_max+0.5), ylim=(y_min-0.5, y_max+2))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # TODO ajouter un titre au dessus du graph des classifications
    ax_IA = inset_axes(ax_robot,
                        width="30%", # width = 30% of parent_bbox
                        height=1., # height : 1 inch
                        loc=1)
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')

    pos = np.arange(len(classification))

    rects = ax_IA.barh(pos, scores[0],
                     align='center',
                     height=0.5,
                     tick_label=classification)

    # ax_IA.set_title('titre ia')

    ax_IA.set_xlim([0., 1.])
    ax_IA.xaxis.set_major_locator(MaxNLocator(11))
    ax_IA.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)

    # Plot a solid vertical gridline to highlight the median position
    ax_IA.axvline(50, color='grey', alpha=0.25)
    scoreLabels = classification
    xlabel = ('Titre xlabel')

    ### ROBOT arm plot
    ax_robot.grid()
    ax_robot.xlim=(x_min-0.5, x_max+0.5)
    ax_robot.ylim=(y_min-0.5, y_max+0.5)
    ax_robot.autoscale_on=False
    ax_robot.aspect='equal'

    # Time plotting of fig
    time_template = 'time = %.1fs'
    time_text = ax_robot.text(0.05, 0.9, '', transform=ax_robot.transAxes)

    lines = [(ax_robot.plot([], [], '-', lw=10, solid_capstyle='round'))[0] for i in range(len(x[0]))]

    def init_robot():
        for i in range(len(lines)):
            lines[i].set_data([], [])
        time_text.set_text('')



        for rect, s in zip(rects, scores[0]):
            rect.set_width(s)
        # fig.clf()
        # ax_IA.clf()
        # ax_robot.clf()

        return [time_text] + lines + rects.patches

    def animate_robot(i):
        for j in range(len(lines)):
            lines[j].set_data(x[i][j], y[i][j])
        time_text.set_text(time_template % (i*dt))


        for rect, s in zip(rects, scores[i]):
            rect.set_width(s)
        return [time_text] + lines + rects.patches

    ani_robot = animation.FuncAnimation(fig, animate_robot, np.arange(1, len(x)),
                                interval=5000*dt, blit=True, init_func=init_robot, repeat=False)

    return {'fig': fig,
            'ax_robot': ax_robot,
            'ax_IA': ax_IA,
            'bars': rects,
            'ani': ani_robot}

arts = plot_student_results(classification, scores)
fig = arts['fig']
ani = arts['ani']

### Live action netflix adaptation
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()