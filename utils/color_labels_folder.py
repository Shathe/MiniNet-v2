from __future__ import print_function
import os
import numpy as np
import glob
import cv2
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Dataset to train", default='./test100')
parser.add_argument("--n_labels", help="Dataset to train", default=12)
args = parser.parse_args()


labels_files = glob.glob(os.path.join(args.folder,'*'))
print(len(labels_files))


out_dir = os.path.join(args.folder,'colored')
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

label_to_color = {}

r = lambda: random.randint(0, 255)

for label_i in xrange(args.n_labels):
    color_i = tuple([r(), r(), r()])
    label_to_color[str(label_i)] = color_i

#label_to_color ={'11': (225, 210, 158), '10': (78, 195, 159), '1': (129, 57, 88), '0': (60, 54, 60), '3': (189, 130, 28), '2': (132, 196, 248), '5': (207, 180, 202), '4': (199, 202, 212), '7': (193, 241, 4), '6': (168, 25, 169), '9': (251, 22, 253), '8': (67, 64, 235)}



print (label_to_color)

for file in labels_files:

    img = cv2.imread(file, 1)

    for label_i in xrange(args.n_labels):
        color_i = label_to_color[str(label_i)]
        cond = img[:,:,0]==label_i
        img[cond,:]=color_i
    new_file = file.replace(args.folder,out_dir)
    cv2.imwrite(new_file, img)


print('Finished')
