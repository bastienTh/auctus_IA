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
#
# net name = <00 net number>_<000 TW>_<00 epoch>_<0000 Batch size>_<0-0 overlap>

python3 train_model.py ../data/ label30_03_30_05_0256_0-5 0 30 0.5 256 5 0.3 3
python3 train_model.py ../data/ label30_06_30_05_0256_0-5 0 30 0.5 256 5 0.3 6
python3 train_model.py ../data/ label30_14_30_05_0256_0-5 0 30 0.5 256 5 0.3 14
python3 train_model.py ../data/ label30_07_30_05_0256_0-5 0 30 0.5 256 5 0.3 7

python3 train_model.py ../data/ label30_03_20_05_0256_0-5 0 20 0.5 256 5 0.3 3
python3 train_model.py ../data/ label30_06_20_05_0256_0-5 0 20 0.5 256 5 0.3 6
python3 train_model.py ../data/ label30_14_20_05_0256_0-5 0 20 0.5 256 5 0.3 14
python3 train_model.py ../data/ label30_07_20_05_0256_0-5 0 20 0.5 256 5 0.3 7

python3 train_model.py ../data/ label30_03_05_05_0256_0-5 0 05 0.5 256 5 0.3 3
python3 train_model.py ../data/ label30_06_05_05_0256_0-5 0 05 0.5 256 5 0.3 6
python3 train_model.py ../data/ label30_14_05_05_0256_0-5 0 05 0.5 256 5 0.3 14
python3 train_model.py ../data/ label30_07_05_05_0256_0-5 0 05 0.5 256 5 0.3 7


# python3 train_model.py ../Valides_5-5/ 00_30_05_0256_0-5 0 30 0.5 256 5 0.3 0
# python3 train_model.py ../Valides_5-5/ 01_30_05_0256_0-5 0 30 0.5 256 5 0.3 1
# python3 train_model.py ../Valides_5-5/ 02_30_05_0256_0-5 0 30 0.5 256 5 0.3 2
# python3 train_model.py ../Valides_5-5/ 03_30_05_0256_0-5 0 30 0.5 256 5 0.3 3
# python3 train_model.py ../Valides_5-5/ 04_30_05_0256_0-5 0 30 0.5 256 5 0.3 4
#
# python3 train_model.py ../Valides_5-5/ 05_30_07_0256_0-5 0 30 0.5 256 7 0.3 5
#
# python3 train_model.py ../Valides_5-5/ 06_30_05_0256_0-5 0 30 0.5 256 5 0.3 6
# python3 train_model.py ../Valides_5-5/ 07_30_05_0256_0-5 0 30 0.5 256 5 0.3 7
#
#
#
# python3 train_model.py ../Valides_5-5/ 08_50_05_0256_0-5 0 50 0.5 256 5 0.3 8
# python3 train_model.py ../Valides_5-5/ 09_50_05_0256_0-5 0 50 0.5 256 5 0.3 9
# python3 train_model.py ../Valides_5-5/ 10_50_05_0256_0-5 0 50 0.5 256 5 0.3 10
# python3 train_model.py ../Valides_5-5/ 03_50_05_0256_0-5 0 50 0.5 256 5 0.3 3
# python3 train_model.py ../Valides_5-5/ 11_50_05_0256_0-5 0 50 0.5 256 5 0.3 11
#
# python3 train_model.py ../Valides_5-5/ 12_50_07_0256_0-5 0 50 0.5 256 7 0.3 12
#
# python3 train_model.py ../Valides_5-5/ 13_50_05_0256_0-5 0 50 0.5 256 5 0.3 13
# python3 train_model.py ../Valides_5-5/ 14_50_05_0256_0-5 0 50 0.5 256 5 0.3 14
# python3 train_model.py ../Valides_5-5/ 15_50_05_0256_0-5 0 50 0.5 256 5 0.3 15
#
#
#
# python3 train_model.py ../Valides_5-5/ 00_70_05_0256_0-5 0 70 0.5 256 5 0.3 0
# python3 train_model.py ../Valides_5-5/ 01_70_05_0256_0-5 0 70 0.5 256 5 0.3 1
# python3 train_model.py ../Valides_5-5/ 02_70_05_0256_0-5 0 70 0.5 256 5 0.3 2
# python3 train_model.py ../Valides_5-5/ 03_70_05_0256_0-5 0 70 0.5 256 5 0.3 3
# python3 train_model.py ../Valides_5-5/ 04_70_05_0256_0-5 0 70 0.5 256 5 0.3 4
#
# python3 train_model.py ../Valides_5-5/ 05_70_07_0256_0-5 0 70 0.5 256 7 0.3 5
#
# python3 train_model.py ../Valides_5-5/ 06_70_05_0256_0-5 0 70 0.5 256 5 0.3 6
# python3 train_model.py ../Valides_5-5/ 07_70_05_0256_0-5 0 70 0.5 256 5 0.3 7


# echo "Doing"
#
# if [ $1 -eq "1" ]
# then
#   echo $1
#   python3 train_model.py ../Valides_5-5/ 00_30_05_0256_0-5 0 30 0.5 256 5 0.3 0
#   python3 train_model.py ../Valides_5-5/ 01_30_05_0256_0-5 0 30 0.5 256 5 0.3 1
#   python3 train_model.py ../Valides_5-5/ 02_30_05_0256_0-5 0 30 0.5 256 5 0.3 2
#   python3 train_model.py ../Valides_5-5/ 03_30_05_0256_0-5 0 30 0.5 256 5 0.3 3
#   python3 train_model.py ../Valides_5-5/ 04_30_05_0256_0-5 0 30 0.5 256 5 0.3 4
# fi
#
# if [ $1 -eq "2" ]
# then
#   echo $1
#   python3 train_model.py ../Valides_5-5/ 05_30_07_0256_0-5 0 30 0.5 256 7 0.3 5
#
#   python3 train_model.py ../Valides_5-5/ 06_30_05_0256_0-5 0 30 0.5 256 5 0.3 6
#   python3 train_model.py ../Valides_5-5/ 07_30_05_0256_0-5 0 30 0.5 256 5 0.3 7
# fi
#
# if [ $1 -eq "3" ]
# then
#   echo $1
#   python3 train_model.py ../Valides_5-5/ 08_50_05_0256_0-5 0 50 0.5 256 5 0.3 8
#   python3 train_model.py ../Valides_5-5/ 09_50_05_0256_0-5 0 50 0.5 256 5 0.3 9
#   python3 train_model.py ../Valides_5-5/ 10_50_05_0256_0-5 0 50 0.5 256 5 0.3 10
#   python3 train_model.py ../Valides_5-5/ 03_50_05_0256_0-5 0 50 0.5 256 5 0.3 3
#   python3 train_model.py ../Valides_5-5/ 11_50_05_0256_0-5 0 50 0.5 256 5 0.3 11
# fi
#
# if [ $1 -eq "4" ]
# then
#   echo $1
#   python3 train_model.py ../Valides_5-5/ 12_50_07_0256_0-5 0 50 0.5 256 7 0.3 12
#
#   python3 train_model.py ../Valides_5-5/ 13_50_05_0256_0-5 0 50 0.5 256 5 0.3 13
#   python3 train_model.py ../Valides_5-5/ 14_50_05_0256_0-5 0 50 0.5 256 5 0.3 14
#   python3 train_model.py ../Valides_5-5/ 15_50_05_0256_0-5 0 50 0.5 256 5 0.3 15
# fi
#
# if [ $1 -eq "5" ]
# then
#   echo $1
#   python3 train_model.py ../Valides_5-5/ 00_70_05_0256_0-5 0 70 0.5 256 5 0.3 0
#   python3 train_model.py ../Valides_5-5/ 01_70_05_0256_0-5 0 70 0.5 256 5 0.3 1
#   python3 train_model.py ../Valides_5-5/ 02_70_05_0256_0-5 0 70 0.5 256 5 0.3 2
#   python3 train_model.py ../Valides_5-5/ 03_70_05_0256_0-5 0 70 0.5 256 5 0.3 3
#   python3 train_model.py ../Valides_5-5/ 04_70_05_0256_0-5 0 70 0.5 256 5 0.3 4
# fi
#
# if [ $1 -eq "6" ]
# then
#   echo $1
#   python3 train_model.py ../Valides_5-5/ 05_70_07_0256_0-5 0 70 0.5 256 7 0.3 5
#
#   python3 train_model.py ../Valides_5-5/ 06_70_05_0256_0-5 0 70 0.5 256 5 0.3 6
#   python3 train_model.py ../Valides_5-5/ 07_70_05_0256_0-5 0 70 0.5 256 5 0.3 7
# fi
# python3 train_model.py ../Valides_5-5/ 07_30_05_0256_0-5 0 30 0.5 256 5 0.3 7
# python3 train_model.py ../Valides_5-5/ 07_70_05_0256_0-5 0 70 0.5 256 5 0.3 7
# python3 train_model.py ../Valides_5-5/ 15_50_05_0256_0-5 0 50 0.5 256 5 0.3 15
#
#
# echo "Done"
