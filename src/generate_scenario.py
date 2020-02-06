import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from datetime import datetime
from robot_control import Arm
from intersection import segments_distance, LineSegment, Point
from infinity import inf
import sys, getopt # for cli args parsing
import os # for directories
from math import pi, ceil
import random

# We labelize each frame according to these thresholds. If the collision happens
# under thresholds[i] frames, the frame will be labelized 'i'.
# label_thresholds = [30, 60, 90, inf]
label_thresholds = [30, 60, 90, 120]

# We crop that many last frames from all videos with no collisions to avoid
# mislabelling in case of collision right after the end of the video. This
# should be greater than label_thresholds[-2].
no_collision_cropping = label_thresholds[-2]

arm_radius = 0.05

movement_types = {
    '3_2H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, 0, 0], 'config2':[pi, 0, random.uniform(-pi, pi)],
        'movable1':[False, False, False], 'movable2':[False, False, True]
    },
    '3_2': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, -pi/4, 0], 'config2':[3*pi/4, pi/4, random.uniform(-pi, pi)],
        'movable1':[False, False, False], 'movable2':[False, False, True]
    },
    '3_1H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, 0, 0], 'config2':[pi, random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, False], 'movable2':[False, True, True]
    },
    '3_1': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, -pi/4, 0], 'config2':[3*pi/4, random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, False], 'movable2':[False, True, True]
    },
    '3_0H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, 0, 0], 'config2':[random.uniform(-pi, 0), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, False], 'movable2':[True, True, True]
    },
    '3_0': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, -pi/4, 0], 'config2':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, False], 'movable2':[True, True, True]
    },
    '2_2H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, 0, random.uniform(-pi, pi)], 'config2':[pi, 0, random.uniform(-pi, pi)],
        'movable1':[False, False, True], 'movable2':[False, False, True]
    },
    '2_2': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, -pi/4, random.uniform(-pi, pi)], 'config2':[3*pi/4, pi/4, random.uniform(-pi, pi)],
        'movable1':[False, False, True], 'movable2':[False, False, True]
    },
    '2_1H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, 0, random.uniform(-pi, pi)], 'config2':[pi, random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, True], 'movable2':[False, True, True]
    },
    '2_1': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, -pi/4, random.uniform(-pi, pi)], 'config2':[3*pi/4, random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, True], 'movable2':[False, True, True]
    },
    '2_0H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, 0, random.uniform(-pi, pi)], 'config2':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, True], 'movable2':[True, True, True]
    },
    '2_0': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, -pi/4, random.uniform(-pi, pi)], 'config2':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, False, True], 'movable2':[True, True, True]
    },
    '1_1H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, random.uniform(-pi, pi), random.uniform(-pi, pi)], 'config2':[pi, random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, True, True], 'movable2':[False, True, True]
    },
    '1_1': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, random.uniform(-pi, pi), random.uniform(-pi, pi)], 'config2':[3*pi/4, random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, True, True], 'movable2':[False, True, True]
    },
    '1_0H': {
        'base1':[0,0], 'base2':[5.5,0],
        'config1':[0, random.uniform(-pi, pi), random.uniform(-pi, pi)], 'config2':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, True, True], 'movable2':[True, True, True]
    },
    '1_0': {
        'base1':[0,0], 'base2':[5,0],
        'config1':[pi/4, random.uniform(-pi, pi), random.uniform(-pi, pi)], 'config2':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, True, True], 'movable2':[True, True, True]
    },
    '0_0H': {
        'base1':[0,0], 'base2':[4.5,0],
        'config1':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)], 'config2':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, True, True], 'movable2':[True, True, True]
    },
    '0_0': {
        'base1':[0,0], 'base2':[4,0],
        'config1':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)], 'config2':[random.uniform(0, pi), random.uniform(-pi, pi), random.uniform(-pi, pi)],
        'movable1':[False, True, True], 'movable2':[True, True, True]
    },
}

def randomize_config():
    for type in movement_types:
        for i in (1, 2):
            for j, b in enumerate(movement_types[type]['movable'+str(i)]):
                if b:
                    movement_types[type]['config'+str(i)][j] = random.uniform(-pi, pi) if j else random.uniform(0, pi)

# How to generate positions procedurally
# 1) Choose starting config
#   a) 15 angles per joint, 7 for the base
#   b) Possibly randomized a bit
# 2) Choose fized joints
#   Choose randomly to freeze the first 0-3 joints, with emphasis on 0
#   proportions: ~[0.5, 0.3, O.15, 0.05]
# 3) Choose bases
#   a) First one in the plane (or in 0,0, parameterable)
#   b) Second one at a short, randomized distance (arm_size, 3*arm_size)
# 4) Move! (many times)

def get_partition(max_pos):
    assert max_pos != 0

    max_joint_blocks = 5
    less_joint_blocks = 2
    joint1_positions = 4
    joint2_positions = 5
    max_joint_pos = joint1_positions * joint2_positions * joint2_positions
    # batch_size = 100 # a batch is composed of scenarios with the same configuration
    batch_size = 10 # a batch is composed of scenarios with the same configuration

    nb_in_joint_blocks, nb_in_joint_pos = 0, 0

    if max_pos//max_joint_blocks >= batch_size:
        nb_blocks = max_joint_blocks
        nb_in_joint_blocks = max_pos//max_joint_blocks
    elif max_pos//less_joint_blocks >= batch_size:
        nb_blocks = less_joint_blocks
        nb_in_joint_blocks = max_pos//less_joint_blocks
    else:
        nb_blocks = 1
        nb_in_joint_blocks = max_pos

    if nb_in_joint_blocks//max_joint_pos >= batch_size:
        nb_pos = max_joint_pos
        nb_in_joint_pos = nb_in_joint_blocks//max_joint_pos
    else:
        nb_pos = ceil(nb_in_joint_blocks/batch_size)
        nb_in_joint_pos = nb_in_joint_blocks//nb_pos # we slice in y sections of ~100

    return nb_blocks, nb_in_joint_blocks, nb_pos, nb_in_joint_pos

def get_start_parameters(current_pos, max_pos):
    assert max_pos >= current_pos
    parameters = {
        'base1':[0,0], 'base2':[4,0],
        'config1':[pi/2, 0, 0], 'config2':[pi/2, 0, 0],
        'movable1':[True, True, True], 'movable2':[True, True, True]
    }

    nb_blocks, nb_in_joint_blocks, nb_pos, nb_in_joint_pos = get_partition(max_pos)

    joint_blocks = [
        {'movable1':[True, True, True], 'movable2':[True, True, True]},
        {'movable1':[False, False, True], 'movable2':[True, True, True]},
        {'movable1':[False, False, True], 'movable2':[False, False, True]},
        {'movable1':[False, False, False], 'movable2':[True, True, True]},
        {'movable1':[False, False, False], 'movable2':[False, False, True]},
    ]

    joint_pos1 = [[a, b, c] for a in (0 , pi/4, pi/2, 3*pi/4) for b in (0, -pi/3, -2*pi/3, pi/3, 2*pi/3) for c in (0, -pi/3, -2*pi/3, pi/3, 2*pi/3)]
    joint_pos2 = [[a, b, c] for a in (pi, 3*pi/4, pi/2, pi/4) for b in (0, pi/3, 2*pi/3, -pi/3, -2*pi/3) for c in (0, pi/3, 2*pi/3, -pi/3, -2*pi/3)]

    current_joint_block = current_pos//nb_in_joint_blocksmovable2
    current_in_block = current_pos % nb_in_joint_blocks

    current_joint_pos = current_in_block//nb_in_joint_pos
    current_in_pos = current_in_block % nb_in_joint_pos

    parameters['movable1'] = joint_blocks[current_joint_block]['movable1']
    parameters['movable2'] = joint_blocks[current_joint_block]['movable2']

    parameters['config1'] = joint_pos1[current_joint_pos]
    parameters['config2'] = joint_pos2[current_joint_pos]

    return parameters

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

# ------------------------------------------------------
# Create and complete the descriptor file (.txt) of this mouvement
def labelize(mvt, collision, file):
    with open(file, 'w', newline='') as descriptor_file:
        nb_labels = [0]*len(label_thresholds)
        frame_labels=''
        if collision:
            label = len(mvt)-1
            for pos in range(len(mvt)):
                for label_index in range(len(label_thresholds)):
                    if label < label_thresholds[label_index]:
                        nb_labels[label_index] += 1
                        frame_labels += str(label_index)+' '
                        break
                label -= 1
        else:
            nb_labels[-1] = len(mvt)
            for i in range(len(mvt)):
                frame_labels += str(len(label_thresholds)-1)+' '

        descriptor_file.write(frame_labels + '\n')
        for i in range(len(nb_labels)):
            descriptor_file.write('label_%d: %d\n' % (i, nb_labels[i]))

# ------------------------------------------------------
# Name of the CSV file (diferent for every new generation)
def filenames(file, type, collision):
    if file is None:
        folder = '../data/' + type + '/'
        if collision:
            folder+='collision/'
        else:
            folder+='no_collision/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        now = datetime.now()
        name=folder+str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'__'+str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'--'+str(now.microsecond)
    else:
        name = file[:-len('_data.csv')]
    return name + '_data.csv', name + '_labels.txt'

def labelize_and_save(mvt, collision, type, file, label):
    data_file, labels_file = filenames(file, type, collision)
    if label:
        labelize(mvt, collision, labels_file)
    if file is None:
        write_data(mvt, data_file)

def mvt_collision(mvt, arm1_radius, arm2_radius):
    for i in range(3):
        for j in range(4,7):
            if segments_distance(   LineSegment(Point(mvt[i][0],mvt[i][1]), Point(mvt[i+1][0],mvt[i+1][1])),
                                    LineSegment(Point(mvt[j][0],mvt[j][1]), Point(mvt[j+1][0],mvt[j+1][1]))) < (arm1_radius + arm2_radius):
                return True
    return False

def read_mvt(file):
    mvt = []
    collision = False
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        spamreader.__next__()
        for row in spamreader:
            pos = [[float(row[2*i+j]) for j in (0, 1)] for i in range(len(row)//2)]

            mvt.append(pos)
            if mvt_collision(pos, arm_radius, arm_radius):
                collision = True
                break
    return mvt, collision

# ------------------------------------------------------
# Generate a movement and structure the data for the CSV
def generate_mvt(start_config, nb_mvts=1):
    arm1 = Arm(base_pos=start_config['base1'], joints_config=start_config['config1'], radius=arm_radius)
    arm1.with_constraints(movable=start_config['movable1'])
    arm2 = Arm(base_pos=start_config['base2'], joints_config=start_config['config2'], radius=arm_radius)
    arm2.with_constraints(movable=start_config['movable2'])

    collision = False
    mvt_number = 0
    mvt1 = []
    mvt2 = []
    while mvt_number < nb_mvts:
        mvt1 += arm1.goto_random_pos()
        mvt2 += arm2.goto_random_pos()
        mvt_number += 1
    mvt  = [mvt1[min(i, len(mvt1)-1)] + mvt2[min(i, len(mvt2)-1)] for i in range(max(len(mvt1), len(mvt2)))]
    for j in range(len(mvt)):
        if mvt_collision(mvt[j], arm1.get_radius(), arm2.get_radius()):
            mvt = mvt[:j+1]
            collision = True
            break

    if not collision:
        mvt = mvt[:-no_collision_cropping]

    # keep only the last moves (near the collision)
    if label_thresholds[-1] not in (-inf, inf) and len(mvt) > label_thresholds[-1]:
        mvt = mvt[-label_thresholds[-1]:]

    assert len(mvt) <= label_thresholds[-1]
    return mvt, collision

def swap_config(config):
    conf1 = np.array([pi, 0, 0]) - np.array(config['config1'])
    conf2 = np.array([pi, 0, 0]) - np.array(config['config2'])
    config['config1'] = conf2.tolist()
    config['config2'] = conf1.tolist()
    config['movable1'], config['movable2'] = config['movable2'], config['movable1']
    return config

def make_and_save_mvt(type, nb_mvts, label):
    start_config = movement_types[type]
    if random.choice([True, False]):
        start_config = swap_config(start_config)

    mvt = []
    while len(mvt) < no_collision_cropping:
        mvt, collision = generate_mvt(start_config, nb_mvts=nb_mvts)
        randomize_config()
    labelize_and_save(mvt, collision, type, None, label)

def main(n, file = None, type = '0_0', multi = None, nb_mvts = 1):
    if multi is None:
        if type == 'all':
            type = '0_0'
        if file is None:
            make_and_save_mvt(type, nb_mvts, not(n))
        else:
            mvt, collision = read_mvt(file)
            labelize_and_save(mvt, collision, type, file, not(n))

    else:
        for i in range(multi):
            if type == 'all':
                print(i, end="\r")
                for current_type in movement_types:
                    make_and_save_mvt(current_type, nb_mvts, not(n))
            else:
                if i%10 == 0:
                    print(i, end="\r")
                make_and_save_mvt(current_type, nb_mvts, not(n))

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],"hni:t:m:M:",["help", "nolabel", "ifile=", "type=", "multi=", "mvts="])
    file = None
    n = False
    type = '0_0'
    multi = None
    nb_mvts = 10
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: % [-h, --help] [-n, --nolabel] [-i, --ifile=<input_file>] [-t, --type=<movement type>] [-m, --multi=<nb_files>] [-M, --mvts=<number of movements per simulation>]")
            exit()
        elif opt in ("-i", "--ifile"):
            file = arg
        elif opt in ("-n", "--nolabel"):
            n = True
        elif opt in ("-t", "--type"):
            type = arg if arg in movement_types else 'all'
        elif opt in ("-m", "--multi"):
            multi = int(arg)
        elif opt in ("-M", "--mvts"):
            nb_mvts = int(arg)
    if file is not None and n:
        exit()
    if multi is not None:
        file = None
    if type == 'all' and multi is None:
        type = '0_0'
    main(n, file, type, multi, nb_mvts)
