import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from datetime import datetime
from robot_control import Arm
from intersection import segments_distance, LineSegment, Point
from infinity import inf

# We labelize each frame according to these thresholds. If the collision happens
# under thresholds[i] frames, the frame will be labelized 'i'.
label_thresholds = [30, 60, 90, inf]
assert label_thresholds[-1] == inf

# We crop that many last frames from all videos with no collisions to avoid
# mislabelling in case of collision right after the end of the video. This
# should be greater than label_thresholds[-2].
no_collision_cropping = 120


def mvt_collision(mvt, arm1, arm2):
    for i in range(3):
        for j in range(4,7):
            # print(segments_distance(LineSegment(Point(mvt[i][0],mvt[i][1]), Point(mvt[i+1][0],mvt[i+1][1])),
            #                         LineSegment(Point(mvt[j][0],mvt[j][1]), Point(mvt[j+1][0],mvt[j+1][1]))))
            # print(arm1.get_radius() + arm2.get_radius())
            if segments_distance(   LineSegment(Point(mvt[i][0],mvt[i][1]), Point(mvt[i+1][0],mvt[i+1][1])),
                                    LineSegment(Point(mvt[j][0],mvt[j][1]), Point(mvt[j+1][0],mvt[j+1][1]))) < (arm1.get_radius() + arm2.get_radius()):
                return True
    return False

# ------------------------------------------------------
# Generate a mouvement and structure the data for the CSV
def generate_mvt():
    arm1 = Arm(base_pos=[0,0])
    arm2 = Arm(base_pos=[4,0])
    mvts = []
    collision = False
    i = 0
    while ((i < 80) and (collision == False)):
        mvt1 = arm1.goto_random_pos(100)
        mvt2 = arm2.goto_random_pos(100)
        mvt  = [mvt1[i] + mvt2[i] for i in range(len(mvt1))]
        for j in range(len(mvt)):
            if mvt_collision(mvt[j], arm1, arm2):
                mvt = mvt[:j+1]
                collision = True
                break
        i+=1
        mvts = mvts + mvt
    return mvts, collision

# ------------------------------------------------------
# Name of the CSV file (diferent for every new generation)
def filenames(collision):
    if collision:
        folder='../data/collision/'
    else:
        folder='../data/no_collision/'
    now = datetime.now()
    name=str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'__'+str(now.hour)+':'+str(now.minute)+':'+str(now.second)+'::'+str(now.microsecond)
    return folder + name + '_data.csv', folder + name + '_labels.txt'


# ------------------------------------------------------
# Create and complete the descriptor file (.txt) of this mouvement
def labelize(mvt, collision, file):
    with open(file, 'w', newline='') as descriptor_file:
        nb_labels = [0]*len(label_thresholds)
        frame_labels=''
        if collision:
            label = len(mvt)-1
            for i in range(len(mvt)):
                for j in range(len(label_thresholds)):
                    if label < label_thresholds[j]:
                        nb_labels[j] += 1
                        frame_labels += str(j)+' '
                        break
                label -= 1
        else:
            mvt = mvt[:-no_collision_cropping]
            nb_labels[-1] = len(mvt)
            for i in range(len(mvt)):
                frame_labels += str(len(label_thresholds)-1)+' '

        descriptor_file.write(frame_labels + '\n')
        for i in range(len(nb_labels)):
            descriptor_file.write('label_%d: %d\n' % (i, nb_labels[i]))

# ------------------------------------------------------
# Create and complete the CSV representing this mouvement
def write_data(mvt, file):
    with open(file, 'w', newline='') as csvfile:
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
        for pos in mvt:
            writer.writerow({
                'arm_1_x0': pos[0][0], 'arm_1_y0': pos[0][1],
                'arm_1_x1': pos[1][0], 'arm_1_y1': pos[1][1],
                'arm_1_x2': pos[2][0], 'arm_1_y2': pos[2][1],
                'arm_1_x3': pos[3][0], 'arm_1_y3': pos[3][1],

                'arm_2_x0': pos[4][0], 'arm_2_y0': pos[4][1],
                'arm_2_x1': pos[5][0], 'arm_2_y1': pos[5][1],
                'arm_2_x2': pos[6][0], 'arm_2_y2': pos[6][1],
                'arm_2_x3': pos[7][0], 'arm_2_y3': pos[7][1]
            })

def main():
    mvt, collision = generate_mvt()
    data_file, labels_file = filenames(collision)
    labelize(mvt, collision, labels_file)
    write_data(mvt, data_file)

if __name__ == '__main__':
    main()
