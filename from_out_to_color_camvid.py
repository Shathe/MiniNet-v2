from __future__ import print_function
import os
import numpy as np
import glob
import cv2
import random
import scipy
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--input_dir", help="Dataset to train", default='./out_dir/Datasets/camvid') 
parser.add_argument("--output_dir", help="Dataset to train", default='./out_dir/Datasets/camvid_colored') 
args = parser.parse_args()
from collections import namedtuple


input_dir = args.input_dir
output_dir = args.output_dir

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #     name  id  trainId  category     catId hasInstances   ignoreInEval   color
    Label('sky', 0, 0, 'sky', 0, False, True, (70, 130, 180)),

    Label('building', 1, 1, 'sky', 0, False, True, (70, 70, 70)),

    Label('column_pole', 2, 2, 'column_pole', 0, False, True, (153, 153, 153)),
    Label('road', 3, 3, 'road', 0, False, True, (128, 64, 128)),

    Label('sidewalk', 4, 4, 'sidewalk', 0, False, True, (244, 35, 232)),
    Label('Tree', 5, 5, 'Tree', 0, False, True, (107, 142, 35)),

    Label('SignSymbol', 6, 6, 'SignSymbol', 0, False, True, (220, 220, 0)),
    Label('Fence', 7, 7, 'Fence', 0, False, True, (190, 153, 153)),
    Label('Car', 8, 8, 'Car', 0, False, True, (0, 0, 142)),

    Label('Pedestrian', 9, 9, 'Pedestrian', 0, False, True, (220, 20, 60)),
    Label('Bicyclist', 10, 10, 'Bicyclist', 0, False, True, (119, 11, 32)),
    Label('Void', 11, 11, 'Void', 0, False, True, (0, 0, 0)),
]





#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }



def fromIdTraintoColor(imgin, imgout):
    for id in id2label:

        color = (id2label[id].color[2], id2label[id].color[1], id2label[id].color[0])
        imgout[imgin==id2label[id].trainId] = color

    imgout[imgin > 10] = 0
    return imgout


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


outputs = glob.glob(input_dir + '/*')
for output in outputs:
    name = output.split('/')[-1]
    output_name = output_dir + '/' + name
    print(output_name)
    img = cv2.imread(output, 0)
    imgout = cv2.imread(output, 1)
    imgout = fromIdTraintoColor(img, imgout)
    cv2.imwrite(output_name, imgout)

