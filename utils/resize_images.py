
import numpy as np
import random
import cv2
import os
import argparse
import glob
from PIL import Image, ImageOps

random.seed(os.urandom(9))

from glob import glob

for file in glob("./dataset_classif/*/*/*"):

    print(file)
    
    desired_size = 224
    im = Image.open(file)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    im = im.resize(new_size, Image.BILINEAR)# NEAREST BILINEAR
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    new_im.save(file)
    
