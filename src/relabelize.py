### ===================================
#
#   File used to change the generated labels
#
### ===================================
import sys
import os
import glob

# ---------------------------------------------------------------------------------------------------------------------------
### Usage
if len(sys.argv) != 2 :
    print("=============================")
    print("Usage:",sys.argv[0],"<path to data folder>")

# ---------------------------------------------------------------------------------------------------------------------------
### Setting parameters from command input name of the model (should not exist already)
DATA_PATH = sys.argv[1]
if not os.path.exists(DATA_PATH):
    print("ERROR: folder",DATA_PATH,"does not exist")
    exit(0)
DATA_PATH = glob.glob(DATA_PATH+"/*/collision/*labels.txt")
for data in DATA_PATH:
    labels=[]
    with open(data, 'r', newline='') as labelfile:
        label_line = labelfile.readline()
        labels = [int(i) for i in label_line.split()]
    new_labels=''
    for i in range(len(labels)):
        if (len(labels)-i) < 30: 
            new_labels = new_labels+' 0'
        elif (len(labels)-i) < 60: 
            new_labels = new_labels+' 1'
        elif (len(labels)-i) < 90: 
            new_labels = new_labels+' 2'
        else: 
            new_labels = new_labels+' 3'

    with open(data, 'w', newline='') as labelfile:
        labelfile.writelines(new_labels)