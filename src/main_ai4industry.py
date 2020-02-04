import numpy as np
np.random.seed(1)

import random as rn
rn.seed(12345)

import tensorflow as tf

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed

tf.random.set_random_seed(12345)
# tf.random.set_seed(12345) # other version of tensorflow

import pandas as pd
import seaborn as sn
import cv2
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from IPython.display import display

# -------------------------------------------------
filepaths =  glob.glob("../my_data/*/*/*.csv")
# glob.glob("../my_data/*/*/*.csv")
rn.shuffle(filepaths)

# Real columns names are long in the data file format, we prefer to rename them.
ft = ["time","acc_x","acc_y","acc_z",
      "rot_x","rot_y","rot_z",
      "mag_x","mag_y","mag_z",
      "w","qua_x","qua_y","qua_z"
     ]
selected_ft = ["time","acc_x","acc_y","acc_z","rot_z"] 
model_output_file = './model.h5'
nb_features = len(selected_ft) if selected_ft is not None else 13


# -------------------------------------------------
Xs = []
ys = []
classes = set()

for filepath in filepaths:
    ### Preprocess data
    df = pd.read_csv(filepath, header=None, names=selected_ft, skiprows=[0])
    df = df[df['time'] != 0]
    df.dropna(inplace=True)

    if len(df) > 30:
        folderpath, filename = os.path.split(filepath)
        _, foldername = os.path.split(folderpath)

        ### Remove time column
        df.drop(['time'], axis=1)

        ### Resize data to have equal lengths
        arr = df.to_numpy() 
        arr = arr.transpose()
        arr = cv2.resize(arr, (80, nb_features)) 
        #arr.shape = 80, nb_features

        Xs.append(arr)
        ys.append(int(foldername)-1)
        classes.add(foldername)  

nb_classes = len(list(classes))

# -------------------------------------------------
Xs = np.array(Xs)
ys = np.array(ys)

classes = set(ys)
ys = to_categorical(ys, nb_classes)

X_train, X_val, y_train, y_val = train_test_split(Xs, ys, 
                                                  stratify=ys, 
                                                  test_size=0.4, 
                                                  random_state=1234, 
                                                  shuffle=True )

for x in X_val:
    print(x.shape)
for y in y_val:
    print(y)
# -------------------------------------------------
# # Dense model
# model = Sequential()
# model.add(Flatten())
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(nb_classes, activation='softmax'))

# LTSM
model = Sequential()
model.add(LSTM(512))
model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'] )


# -------------------------------------------------
callbacks_list = [
    ModelCheckpoint(
        filepath = model_output_file,
        monitor = 'val_loss', save_best_only=True, mode="min", verbose=1
    ),
    EarlyStopping(monitor='val_accuracy', patience=1, verbose=1)
]

# -------------------------------------------------
# history=model.fit(  
#     X_train,
#     y_train,
#     validation_split=0.33,
#     nb_epoch=200, 
#     batch_size=5, 
#     verbose=0
# )

history = model.fit(
    X_train,
    y_train,
    validation_data = (X_val, y_val),
    # validation_split=0.33,
    batch_size = 64,
    epochs = 50,
    #callbacks = callbacks_list,
    verbose = 1
)

# -------------------------------------------------
fig, (ax1,ax2) = plt.subplots(1, 2)

ax1.plot(history.history['val_loss'], label='val_loss')
ax1.plot(history.history['loss'], label ='loss')
ax1.set_ylim(bottom=0)
ax1.set_title("Loss")

ax2.plot(history.history['acc'], label = 'acc')
ax2.plot(history.history['val_acc'], label='val_acc')
# ax2.plot(history.history['accuracy'], label = 'acc') # other version
# ax2.plot(history.history['val_accuracy'], label='val_acc') # other version
ax2.set_ylim([0, 1])
ax2.set_title("Accuracy")

fig.legend()
plt.show()

# model.predict(X_train[0], use_multiprocessing=False, batch_size=1)


# -------------------------------------------------
predicted = model.predict_classes(X_val)
target = np.argmax(y_val, axis=1)

fail = [x!=y for x,y in zip(predicted,target)]
print("False predictions: %d/%d" % (np.sum(fail), len(y_val)))
print("Pred:", predicted[fail])
print("Real:", target[fail])

cm = confusion_matrix(target,predicted)

df_cm = pd.DataFrame(cm, index = np.array(list(classes))+1,
                     columns = np.array(list(classes))+1)
sn.heatmap(df_cm, annot=True)
plt.show()
print(model.predict(np.array([Xs[0]])))
# print(model.evaluate(np.array([Xs[0]]),np.array([ys[0]])))


my_Xs = []
my_ys = []
# my_classes = set()

my_filepaths = glob.glob("../my_data/*/*/*.csv")
rn.shuffle(my_filepaths)

for filepath in my_filepaths:
    ### Preprocess data
    df = pd.read_csv(filepath, header=None, names=selected_ft, skiprows=[0])
    df = df[df['time'] != 0]
    df.dropna(inplace=True)

    if len(df) > 30:
        folderpath, filename = os.path.split(filepath)
        _, foldername = os.path.split(folderpath)

        ### Remove time column
        df.drop(['time'], axis=1)

        ### Resize data to have equal lengths
        arr = df.to_numpy() 
        arr = arr.transpose()
        arr = cv2.resize(arr, (80, nb_features)) 
        #arr.shape = 80, nb_features

        my_Xs.append(arr)
        my_ys.append(int(foldername)-1)
#         my_classes.add(foldername)  

# my_nb_classes = len(list(my_classes))
# # print(my_classes)

my_Xs = np.array(my_Xs)
my_ys = np.array(my_ys)

# my_classes = set(my_ys)
my_ys = to_categorical(my_ys, nb_classes)

predicted = model.predict_classes(my_Xs)
target = np.argmax(my_ys, axis=1)

fail = [x!=y for x,y in zip(predicted,target)]
print("False predictions: %d/%d" % (np.sum(fail), len(my_Xs)))
print("Pred:", predicted[fail]+1)
print("Real:", target[fail]+1)

cm = confusion_matrix(target,predicted)

df_cm = pd.DataFrame(cm, index = np.array(list(classes))+1,
                     columns = np.array(list(classes))+1)

sn.heatmap(df_cm, annot=True)
plt.show()
print(model.evaluate(my_Xs,my_ys))