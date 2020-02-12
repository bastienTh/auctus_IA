import pandas as pd
import cv2
import os
import glob
import numpy as np
import random as rn
# from parameters import *

"""
Load data will convert data.csv and label.txt files into a IA usable Xs and ys
Xs will be Time_window of frames
full_path can be used for a single scenario to be used
"""
def load_data(path="../data/", shuffle=True, collision_only=False, full_path=False, TIME_WINDOW_SIZE=20, STEP=10):
    if (collision_only):
        if (full_path):
            filepaths =  glob.glob(path)
        else:
            filepaths =  glob.glob(path+"*/collision/*.csv")
    else:
        if (full_path):
            filepaths =  glob.glob(path)
        else:
            filepaths =  glob.glob(path+"*/*/*.csv")

    if (shuffle):
        rn.shuffle(filepaths)


    # Real columns names are long in the data file format, we prefer to rename them.
    features = [
        'arm_1_x0','arm_1_y0',
        'arm_1_x1','arm_1_y1',
        'arm_1_x2','arm_1_y2',
        'arm_1_x3','arm_1_y3',

        'arm_2_x0','arm_2_y0',
        'arm_2_x1','arm_2_y1',
        'arm_2_x2','arm_2_y2',
        'arm_2_x3','arm_2_y3'
    ]

    nb_features = len(features)
    Xs = []
    ys = []
    classes = set()
    k=0
    for filepath in filepaths:
        print("CSV file loaded: ", k, '/', len(filepaths), end='\r')
        k=k+1
        # Preprocess data
        df = pd.read_csv(filepath, header=None, names=features, skiprows=[0])
        df.dropna(inplace=True) # Remove missing values, just in case

        # Getting the label of the current csv file
        folderpath, filename = os.path.split(filepath)

        # to remove "data.csv" at the end of the file
        filename = filename[:-8]
        _, foldername = os.path.split(folderpath)
        with open(folderpath+'/'+filename+'labels.txt', 'r', newline='') as labelfile:
                label_line = labelfile.readline()
                labels = [int(i) for i in label_line.split()]

        ### add each time_window as a labelized value in Xs and its label in ys
        while len(labels):
            if len(labels) < TIME_WINDOW_SIZE + (STEP/2): # +(step/2) is to minimalise the amount of frames compressed or extended (think about it, you'll get it)
                ### Resize data to have equal lengths in case
                arr = df.to_numpy()
                arr = arr.transpose()
                arr = cv2.resize(arr, (TIME_WINDOW_SIZE, nb_features))
                ys.append(labels[len(labels)-1])
                classes.add(labels[len(labels)-1])
                labels=[]

            else:
                arr = df.to_numpy()
                arr = arr[-TIME_WINDOW_SIZE:] # Getting the last frames corresponding to a time_window
                arr = arr.transpose()
                ys.append(labels[len(labels)-1]) # Labels of this TW is the last label
                classes.add(labels[len(labels)-1])

                # Get rid of the last values considering the overlap_ratio (induced by STEP)
                df.drop(df.tail(STEP).index,inplace=True) # drop last n rows
                labels = labels[:-STEP]
            Xs.append(arr)
    print("CSV file loaded: ", k, '/', len(filepaths))
    print("DATA LOADING COMPLETED")
    print()
    return Xs, ys, classes
