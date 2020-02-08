#!/bin/bash

# Usage loads of parameters:
# python3 train_model.py 
# <data_path>                       Path to the data. This folder must have folder and subfoler as follow:
#                                   <data_path>/<scenario_names>/collision/*.csv
#                                                               /no_collision/*.csv
# <model_name>                      Model name, must be new, otherwise the program ask for another name which will block the program and wait for user input
# <bool (0 or 1) collision_only>    To only get the collision csv
# <TIME_WINDOW_SIZE>                How much frames per time_window
# <OVERLAP_RATIO>                   Ratio in ]0;1[, a ratio=0.2 means step will be 20% of the TW_SIZE
# <BATCH_SIZE>                      Batch size for training
# <EPOCHS>                          Epochs for training
# <TEST_SIZE>                       Size of the data used for test in ]0;1[
# <IA_number>                       Which of the IA in the train_model file we want to train (integer in {0,1,2,3, ... ,n} with n last implemented network)

python3 train_model.py ~/Bureau/Valides_5-5/ dense_1 0 30 0.5 128 5 0.3 0
python3 train_model.py ~/Bureau/Valides_5-5/ lstm_256 0 30 0.5 128 5 0.3 1
python3 train_model.py ~/Bureau/Valides_5-5/ lstm_512 0 30 0.5 128 5 0.3 2
python3 train_model.py ~/Bureau/Valides_5-5/ lstm_1024 0 30 0.5 128 5 0.3 3

