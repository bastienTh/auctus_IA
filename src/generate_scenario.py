import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from datetime import datetime
from robot_control import Arm
from intersection import segments_distance, LineSegment, Point 

def mvt_collision(mvt):
    for i in range(3):
        for j in range(4,7):
            if segments_distance(   LineSegment(Point(mvt[i][0],mvt[i][1]), Point(mvt[i+1][0],mvt[i+1][1])), 
                                    LineSegment(Point(mvt[j][0],mvt[j][1]), Point(mvt[j+1][0],mvt[j+1][1]))) < (arm1.get_radius() + arm2.get_radius()):
                return True
    return False

# ------------------------------------------------------
# Generate a mouvement and structure the data for the CSV
arm1 = Arm()
arm2 = Arm()
mvts = []
collision = False
for i in range(100):
    mvt1 = arm1.goto_random_pos(100)
    mvt2 = arm2.goto_random_pos(100)
    mvt  = [mvt1[i] + mvt2[i] for i in range(len(mvt1))]
    mvts = mvts + mvt
    for j in range(100):
        if mvt_collision(mvt[j]):
            collision = True
            break

# ------------------------------------------------------
# Name of the CSV file (diferent for every new generation)
if collision:
    folder='../data/collision/'
else:
    folder='../data/no_collision/'
now = datetime.now()
name=str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'__'+str(now.hour)+':'+str(now.minute)+':'+str(now.second)+'::'+str(now.microsecond)

# ------------------------------------------------------
# Create and complete the descriptor file (.txt) of this mouvement
with open(folder+name+'_labels.txt', 'w', newline='') as descriptor_file:
    L0 = 0
    L1 = 0
    L2 = 0
    L3 = 0
    frame_labels=''
    if collision:
        label = len(mvts)-1
        for i in range(len(mvts)):
            if   (label < 30):
                L3 += 1
                frame_labels += '3 '
            elif (label < 60):
                L2 += 1
                frame_labels += '2 '
            elif (label < 90):
                L1 += 1
                frame_labels += '1 '
            else:
                L0 += 1
                frame_labels += '0 '
            label -= 1
    else:
        L1 = len(mvts)
        for mvt in mvts:
            frame_labels += '0 '

    descriptor_file.write(frame_labels + '\n')
    descriptor_file.write('label_0: ' + str(L0) + '\n')
    descriptor_file.write('label_1: ' + str(L1) + '\n')
    descriptor_file.write('label_2: ' + str(L2) + '\n')
    descriptor_file.write('label_3: ' + str(L3) + '\n')

# ------------------------------------------------------
# Create and complete the CSV representing this mouvement
with open(folder+name+'_data.csv', 'w', newline='') as csvfile:
    fieldnames =[   
                    'arm_1_x0','arm_1_y0',
                    'arm_1_x1','arm_1_y1',
                    'arm_1_x2','arm_1_y2',
                    'arm_1_x3','arm_1_y3',

                    'arm_2_x0','arm_2_y0',
                    'arm_2_x1','arm_2_y1',
                    'arm_2_x2','arm_2_y2',
                    'arm_2_x3','arm_2_y3'
                ]   

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for mvt in mvts:
        writer.writerow({
                            'arm_1_x0': mvt[0][0], 'arm_1_y0': mvt[0][1], 
                            'arm_1_x1': mvt[1][0], 'arm_1_y1': mvt[1][1],
                            'arm_1_x2': mvt[2][0], 'arm_1_y2': mvt[2][1],
                            'arm_1_x3': mvt[3][0], 'arm_1_y3': mvt[3][1],

                            'arm_2_x0': mvt[4][0], 'arm_2_y0': mvt[4][1],
                            'arm_2_x1': mvt[5][0], 'arm_2_y1': mvt[5][1],
                            'arm_2_x2': mvt[6][0], 'arm_2_y2': mvt[6][1],
                            'arm_2_x3': mvt[7][0], 'arm_2_y3': mvt[7][1]  
                        })
