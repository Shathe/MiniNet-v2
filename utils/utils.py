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

def get_parameters():
	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1

		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	print("Total parameters of the net: " + str(total_parameters)+ " == " + str(total_parameters/1000000.0) + "M")

def preprocess(x):
	return x.astype(np.float32) / 255.0 - 0.5

