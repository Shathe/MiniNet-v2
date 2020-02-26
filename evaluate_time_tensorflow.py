import tensorflow as tf
import numpy as np
import random
import math
import os
from Mininet import MiniNet2, MiniNet2_cpu
import argparse
import time
import sys
import cv2

random.seed(os.urandom(9))

parser = argparse.ArgumentParser()
parser.add_argument("--width", help="width", default=1024)
parser.add_argument("--height", help="height", default=512)
parser.add_argument("--cpu_version", help="Whether to use the cpu version", default=0)
parser.add_argument("--cpu", help="Whether to use the cpu", default=0)

args = parser.parse_args()



n_gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)
# Hyperparameter
width = int(args.width)
height = int(args.height)
cpu_version = bool(int(args.cpu_version))
cpu = bool(int(args.cpu))

# For Batch_norm or dropout operations: training or testing
training_flag = tf.placeholder(tf.bool)
 
x = tf.placeholder(tf.float32, shape=[1, height, width, 3], name='input')
label = tf.placeholder(tf.float32, shape=[1, height, width, 11], name='output')
mask_label = tf.placeholder(tf.float32, shape=[1, height, width, 11], name='mask')

# Network
if cpu_version:
	output = MiniNet2_cpu(x, 11, is_training=training_flag)
else:
	output = MiniNet2(x, 11, is_training=training_flag)



flops = tf.profiler.profile(
	tf.get_default_graph(),
	options=tf.profiler.ProfileOptionBuilder.float_operation())


# Count parameters
total_parameters = 0
for variable in tf.trainable_variables():
	# shape is an array of tf.Dimension
	shape = variable.get_shape()
	variable_parameters = 1

	for dim in shape:
		variable_parameters *= dim.value
	total_parameters += variable_parameters


var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
			 for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]



shape_output = output.get_shape()
label_shape = label.get_shape()

predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])
output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
mask_labels = tf.reshape(mask_label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])
from tensorflow.python.client import device_lib


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	img = np.random.randint(0, high=255, size=(1, height, width, 3))

	image_salida2 = output.eval(feed_dict={x: img, training_flag: False})
	image_salida2 = output.eval(feed_dict={x: img, training_flag: False})
	sol = 0.
	for i in range(1000):
		print(device_lib.list_local_devices())
		img = np.random.randint(0, high=255, size=(1, height, width, 3))
		first = time.time()
		image_salida2 = output.eval(feed_dict={x: img, training_flag: False})
		sol += time.time() - first

		print(str(sol/(i+1)*1000) + " ms to inference")
		print(sum(var_sizes) / (1024. ** 2), 'MB')
		print('GFLOP = ', str(flops.total_float_ops/1000000000.))
		print("Total parameters of the net: " + str(total_parameters)+ " == " + str(total_parameters/1000000.0) + "M")
