
import numpy as np
import random
import cv2
import os
import argparse
import glob
from PIL import Image, ImageOps

random.seed(os.urandom(9))

from glob import glob

for file in glob("./*.png"):

    print(file)
    
    desired_size = 224
    im = Image.open(file)
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    im = im.resize((5520, 3680), Image.BILINEAR)# NEAREST BILINEAR
    im.save(file)
    
