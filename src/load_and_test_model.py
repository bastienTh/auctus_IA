import pandas as pd
import tensorflow as tf
import seaborn as sn
import cv2
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import random as rn
import json
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

# Show existing models
print("============================")
print("     Existing models:       ")
print("============================")

os.system("tree -L 2 -- ../models/")

# Chosing and loading the model
value = input("Please enter the model type and name:\nexemple: <cartesian_coord/my_model>\n")
model_name = '../models/'+str(value)+'/model.h5'
model = tf.keras.models.load_model(model_name)
# model.summary()

# Reload the data
Xs, ys, classes = load_data()

print(model.evaluate(np.array([Xs[0]]),np.array([ys[0]])))
print(model.predict(np.array([Xs[0]])))