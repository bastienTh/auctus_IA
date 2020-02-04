import pandas as pd
import tensorflow as tf
import seaborn as sn
import cv2
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from IPython.display import display

### Setting seeds
# tf.random.set_seed(12345) # other version of tensorflow
tf.random.set_random_seed(12345)
np.random.seed(12345)
rn.seed(12345)

TIME_WINDOW_SIZE = 20
OVERLAP_RATIO = 2
STEP = int(TIME_WINDOW_SIZE/OVERLAP_RATIO)


# ---------------------------------------------------------------------------------------------------------------------------
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

value = input("Please enter the model name:\n")
model_output_file = '../models/'+str(value)+'.h5'
nb_features = len(features)

# ---------------------------------------------------------------------------------------------------------------------------
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


nb_classes = len(list(classes))

# ---------------------------------------------------------------------------------------------------------------------------
Xs = np.array(Xs)
ys = np.array(ys)

classes = set(ys)
ys = to_categorical(ys, nb_classes)

X_train, X_val, y_train, y_val = train_test_split(Xs, ys, 
                                                  stratify=ys, 
                                                  test_size=0.5, 
                                                  random_state=12345, 
                                                  shuffle=True )

# print("----- val -----")
# print(len(X_val))
# print(X_val.shape)
# print(len(y_val))
# print(y_val.shape)
# print(len(X_train))
# print(X_train.shape)
# print(len(y_train))
# print(y_train.shape)
# with open('x_val.txt', 'w') as f:
#     for x in X_val:
#         print(x, 'x_val.txt', file=f)
# with open('x_train.txt', 'w') as f:
#     for x in X_train:
#         print(x, 'x_train.txt', file=f)
# with open('y_val.txt', 'w') as f:
#     for y in y_val:
#         print(y, 'y_val.txt', file=f)
# with open('y_train.txt', 'w') as f:
#     for y in y_train:
#         print(y, 'y_train.txt', file=f)

# for y in y_val:
#     print(y)
# print("----- train -----")
# for x in X_train:
#     print(x.shape)
# for y in y_train:
#     print(y)



# ---------------------------------------------------------------------------------------------------------------------------
# # Dense model
# model = Sequential()
# model.add(Flatten())
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(nb_classes, activation='softmax'))

# LTSM
model = Sequential()
model.add(LSTM(512,return_sequences=True))
model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# ---------------------------------------------------------------------------------------------------------------------------
callbacks_list = [
    ModelCheckpoint(
        filepath = model_output_file,
        monitor = 'val_loss', save_best_only=True, mode="min", verbose=1
    ),
    EarlyStopping(monitor='val_accuracy', patience=1, verbose=1)
]

# ---------------------------------------------------------------------------------------------------------------------------
print("history-----------------")
history = model.fit(
    X_train,
    y_train,
    validation_data = (X_val, y_val),
    # validation_split=0.33,
    batch_size = 64,
    epochs = 10,
    #callbacks = callbacks_list,
    verbose = 1
)

# ---------------------------------------------------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------------------------------------------------------
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