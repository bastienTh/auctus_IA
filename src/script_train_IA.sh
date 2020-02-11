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

python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_dense_100 0 30 0.5 256 5 0.3 0
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_dense_500 0 30 0.5 256 5 0.3 1
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_dense_300_300 0 30 0.5 256 5 0.3 2
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_dense_200_300_200 0 30 0.5 256 5 0.3 3
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_dense_200_200_200_200_200 0 30 0.5 256 7 0.3 4
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_LSTM_256 0 30 0.5 256 5 0.3 5
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_LSTM_1024 0 30 0.5 256 5 0.3 6
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_30_LSTM_4096 0 30 0.5 256 5 0.3 7

python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_dense_300 0 50 0.5 256 5 0.3 8
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_dense_100_100 0 50 0.5 256 5 0.3 9
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_dense_500_500 0 50 0.5 256 5 0.3 10
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_dense_100_100_100 0 50 0.5 256 5 0.3 11
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_dense_100_300_100 0 50 0.5 256 5 0.3 12
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_dense_100_100_100_100_100 0 50 0.5 256 7 0.3 13
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_LSTM_512 0 50 0.5 256 5 0.3 14
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_50_LSTM_2048 0 50 0.5 256 5 0.3 15

python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_dense_100 0 70 0.5 256 5 0.3 16
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_dense_500 0 70 0.5 256 5 0.3 17
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_dense_300_300 0 70 0.5 256 5 0.3 18
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_dense_200_300_200 0 70 0.5 256 5 0.3 19
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_dense_200_200_200_200_200 0 70 0.5 256 7 0.3 20
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_LSTM_256 0 70 0.5 256 5 0.3 21
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_LSTM_1024 0 70 0.5 256 5 0.3 22
python3 train_model.py ~/Bureau/Valides_5-5_new/ tw_70_LSTM_4096 0 70 0.5 256 5 0.3 23

