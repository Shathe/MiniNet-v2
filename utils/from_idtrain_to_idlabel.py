import tensorflow as tf
import numpy as np
import random
import math
import os
import argparse
import time
import cv2
import math
import sys
import glob

files = glob.glob(os.path.join('../results/final/*'))
label_to_train={
    '0': 255,
    '1': 255,
    '2': 255,
    '3': 255,
    '4': 255,
    '5': 255,
    '6': 255,
    '7': 0,
    '8': 1,
    '9': 255,
    '10': 255,
    '11': 2,
    '12': 3,
    '13': 4,
    '14': 255,
    '15': 255,
    '16': 255,
    '17': 5,
    '18': 255,
    '19': 6,
    '20': 7,
    '21': 8,
    '22': 9,
    '23': 10,
    '24': 11,
    '25': 12,
    '26': 13,
    '27': 14,
    '28': 15,
    '29': 255,
    '30': 255,
    '31': 16,
    '32': 17,
    '33': 18,
    '-1': 255}

train_to_label={
    '255': 0,
    '0': 7,
    '1': 8,
    '2': 11,
    '3': 12,
    '4': 13,
    '5': 17,
    '6': 19,
    '7': 20,
    '8': 21,
    '9': 22,
    '10': 23,
    '11': 24,
    '12': 25,
    '13': 26,
    '14': 27,
    '15': 28,
    '16': 31,
    '17': 32,
    '18': 33}

def transform(image):

    return np.array([ [train_to_label[str(j)] for j in i] for i in image])

for file in files:

    print(file)
    img = cv2.imread(file, 0)
    print(np.unique(img))
    
    img = np.array(transform(img))
    print(np.unique(img))

    img = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(file, img)
    print(np.unique(img))
