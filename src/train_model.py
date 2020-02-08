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
import sys
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
# This might be useless
modes=['cartesian_coord','distance_mat'] # we could had anything, using speed or whatever
MODE=modes[0]
# ---------------------------------------------------------------------------------------------------------------------------
### Usage
if len(sys.argv) != 10 :
    print("=============================")
    print("Usage:",sys.argv[0],"takes 9 arguments (given",len(sys.argv)-1,") as listed:")
    print("$1 <data_path>                       Path to the data. This folder must have folder and subfoler as follow:")
    print("   <data_path>/<scenario_names>/collision/*.csv")
    print("              /no_collision/*.csv")
    print("$2 <model_name>                      Model name, must be new, otherwise the program ask for another name which")
    print("                                     will block the program and wait for user input")
    print("$3 <bool (0 or 1) collision_only>    To only get the collision csv")
    print("$4 <TIME_WINDOW_SIZE>                How much frames per time_window")
    print("$5 <OVERLAP_RATIO>                   Ratio in ]0;1[, a ratio=0.2 means step will be 20% of the TW_SIZE")
    print("$6 <BATCH_SIZE>                      Batch size for training")
    print("$7 <EPOCHS>                          Epochs for training")
    print("$8 <TEST_SIZE>                       Size of the data used for test in ]0;1[")
    print("$9 <IA_number>                       Which of the IA in the train_model file we want to train (integer in {0,1,2,3, ... ,n} with n last implemented network)")
    exit()
# ---------------------------------------------------------------------------------------------------------------------------
### Setting parameters from command input name of the model (should not exist already)
DATA_PATH = sys.argv[1]
MODEL_NAME = sys.argv[2]
BOOL_COLLISION_ONLY = (sys.argv[3] == '1')
TIME_WINDOW_SIZE = int(sys.argv[4])
OVERLAP_RATIO = float(sys.argv[5])
STEP = int(TIME_WINDOW_SIZE*OVERLAP_RATIO)
BATCH_SIZE = int(sys.argv[6])
EPOCHS = int(sys.argv[7])
TEST_SIZE = float(sys.argv[8])
IA_NUMBER = int(sys.argv[9])

output_path_memo='../models/'+MODE+'/'+str(MODEL_NAME)
output_path=output_path_memo
i=1
while (os.path.exists(output_path)):
    output_path=output_path_memo+'('+str(i)+')'
    i=i+1
os.makedirs(output_path)



# ---------------------------------------------------------------------------------------------------------------------------
### Setting seeds
tf.random.set_random_seed(12345)
# tf.random.set_seed(12345) # other version of tensorflow
np.random.seed(12345)
rn.seed(12345)


# ---------------------------------------------------------------------------------------------------------------------------
Xs, ys, classes = load_data(path=DATA_PATH,collision_only=BOOL_COLLISION_ONLY, TIME_WINDOW_SIZE=TIME_WINDOW_SIZE, STEP=STEP)
nb_classes = len(list(classes))
Xs = np.array(Xs)
ys = np.array(ys)

classes = set(ys)
ys = to_categorical(ys, nb_classes)

X_train, X_val, y_train, y_val = train_test_split(Xs, ys, 
                                                  stratify=ys, 
                                                  test_size=TEST_SIZE, 
                                                  random_state=12345, 
                                                  shuffle=True )


# ---------------------------------------------------------------------------------------------------------------------------
### IA models
if (IA_NUMBER == 0):
    ### Dense model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
    model.add(Dense(nb_classes, activation='softmax'))
elif (IA_NUMBER == 1):
    ### LTSM
    model = Sequential()
    model.add(LSTM(256,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
elif (IA_NUMBER == 2):
    ### LTSM
    model = Sequential()
    model.add(LSTM(512,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
elif (IA_NUMBER == 3):
    ### LTSM
    model = Sequential()
    model.add(LSTM(1024,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
else:
    print("There are no IA number:",IA_NUMBER," implemented yet, see the files train_moddel.py")
    exit(0)



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# ---------------------------------------------------------------------------------------------------------------------------
# Save the entire model to a HDF5 file after each epoch (save the best version)
# Other way to save the model: model.save(output_path+'model.h5') NOTE : not needed because its done automatically in the callback
callbacks_list = [
    ModelCheckpoint(
        filepath = output_path+'/model.h5',
        monitor = 'val_loss', save_best_only=True, mode="min", verbose=1
    ),
    EarlyStopping(monitor='val_acc', patience=1, verbose=1)
]


# ---------------------------------------------------------------------------------------------------------------------------
### model training
history = model.fit(
    X_train,
    y_train,
    validation_data = (X_val, y_val),
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    callbacks = callbacks_list,
    verbose = 1
)


# ---------------------------------------------------------------------------------------------------------------------------
### Ploting the acc and loss
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
plt.savefig(output_path+'acc_loss.png')
# plt.show()


# ---------------------------------------------------------------------------------------------------------------------------
### Ploting the results over the entire set of data
plt.clf() # Clear figure
# model.predict(X_train[0], use_multiprocessing=True, batch_size=1)
predicted = model.predict_classes(X_val)
target = np.argmax(y_val, axis=1)

fail = [x!=y for x,y in zip(predicted,target)]
cm = confusion_matrix(target,predicted)
df_cm = pd.DataFrame(cm, index = np.array(list(classes)),
                     columns = np.array(list(classes)))
sn.heatmap(df_cm, annot=True)
plt.savefig(output_path+'heatmap.png')

# ---------------------------------------------------------------------------------------------------------------------------
### Put the model overall summary in a txt file
stdout = sys.stdout # get stdout

sys.stdout = open(output_path+'summary.txt', 'w') # Changing stdout to the file.txt
print("========================================")
print("      OVERALL RECAP OF THE MODEL        ")
print("========================================")
print()
print("---------------------------------------------")
print(">>     Parameters     <<")
print()
print("> DATA_PATH:            ", DATA_PATH)
print("> MODEL_NAME:           ", MODEL_NAME)
print("> BOOL_COLLISION_ONLY:  ", BOOL_COLLISION_ONLY)
print("> TIME_WINDOW_SIZE:     ", TIME_WINDOW_SIZE)
print("> OVERLAP_RATIO:        ", OVERLAP_RATIO)
print("> BATCH_SIZE:           ", BATCH_SIZE)
print("> EPOCHS:               ", EPOCHS)
print("> TEST_SIZE:            ", TEST_SIZE)
print("> IA_NUMBER:            ", IA_NUMBER)
print("---------------------------------------------")
print()
model.summary()
print()
print("False predictions: %d/%d" % (np.sum(fail), len(y_val)))
print("Pred:", predicted[fail])
print("Real:", target[fail])

sys.stdout = stdout # Reset stdout
model.summary() # print thee summary in the terminal as well
