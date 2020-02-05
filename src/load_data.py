import pandas as pd
import cv2
import os
import glob
import numpy as np
import random as rn

from param import *

def load_data():
    filepaths =  glob.glob("../data/*/*.csv")
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
    for filepath in filepaths:
        # Preprocess data
        df = pd.read_csv(filepath, header=None, names=features, skiprows=[0])
        df.dropna(inplace=True) # Remove missing values
        
        # Getting the label of the current csv file
        folderpath, filename = os.path.split(filepath)

        # to remove "data.csv" at the end of the file
        filename = filename[:-8] 
        _, foldername = os.path.split(folderpath)
        with open(folderpath+'/'+filename+'labels.txt', 'r', newline='') as labelfile:
                label_line = labelfile.readline()
                labels = [int(i) for i in label_line.split()]

        # add each time_window as a labelized value in Xs and its label in ys
        # TODO, atm we take a timevalue and remove it, some timewindow are contingent
        # (no overlap)
        # 
        # TODO intersting values are at the end (kaboom), 
        # we might wanna get timewindow starting from the end
        # so the "adapted" values are in the beginning
        while len(labels):
            if len(labels) < TIME_WINDOW_SIZE + STEP:
                ### Resize data to have equal lengths in case 
                arr = df.to_numpy() 
                arr = arr.transpose()
                arr = cv2.resize(arr, (TIME_WINDOW_SIZE, nb_features)) 
                ys.append(labels[len(labels)-1])
                classes.add(labels[len(labels)-1])  
                labels=[]

            else:
                arr = df.to_numpy()
                arr = arr[:TIME_WINDOW_SIZE]
                arr = arr.transpose()
                ys.append(labels[TIME_WINDOW_SIZE-1])
                classes.add(labels[TIME_WINDOW_SIZE-1])  
                # Get rid of the first values
                # df = df.iloc[STEP:]
                df.drop(df.index[:STEP], inplace=True)
                labels = labels[STEP:]
            Xs.append(arr)  
    return Xs, ys, classes