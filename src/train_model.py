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

from param import *

### Setting seeds
# tf.random.set_seed(12345) # other version of tensorflow
tf.random.set_random_seed(12345)
np.random.seed(12345)
rn.seed(12345)

name = input("Please enter the model name:\n")
output_path='../models/'+str(name)+'/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print("Error: this model already exist, pick another name")
    exit(0)

Xs, ys, classes = load_data()
nb_classes = len(list(classes))

# ---------------------------------------------------------------------------------------------------------------------------
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
# # Dense model
# model = Sequential()
# model.add(Flatten())
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(100, activation='relu', input_shape=(nb_features, 80)))
# model.add(Dense(nb_classes, activation='softmax'))

# LTSM
model = Sequential()
model.add(LSTM(LSTM_SIZE,return_sequences=True))
model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# ---------------------------------------------------------------------------------------------------------------------------
# Save the entire model to a HDF5 file after each epoch (save the best version)
# model.save(output_path+'model.h5') NOTE : not needed because its done automatically in the callback
callbacks_list = [
    ModelCheckpoint(
        filepath = output_path+'/model.h5',
        monitor = 'val_loss', save_best_only=True, mode="min", verbose=1
    ),
    EarlyStopping(monitor='val_acc', patience=1, verbose=1)
]

# ---------------------------------------------------------------------------------------------------------------------------
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
# model.predict(X_train[0], use_multiprocessing=True, batch_size=1)
plt.clf() # Clear figure
predicted = model.predict_classes(X_val)
target = np.argmax(y_val, axis=1)

fail = [x!=y for x,y in zip(predicted,target)]
print("False predictions: %d/%d" % (np.sum(fail), len(y_val)))
print("Pred:", predicted[fail])
print("Real:", target[fail])

cm = confusion_matrix(target,predicted)

df_cm = pd.DataFrame(cm, index = np.array(list(classes)),
                     columns = np.array(list(classes)))
sn.heatmap(df_cm, annot=True)
plt.savefig(output_path+'heatmap.png')
# plt.show()

# Save the history as .csv
# NOTE : note really working
# pd.DataFrame.from_dict(model.history.history).to_csv('history.csv',index=False)