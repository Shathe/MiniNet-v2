import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
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

def export_to_pb(sess, name='model.pb'):
    # Export to .pb file for tensorflowLite
    # get the tf graph and retrieve operation names
    graph = tf.get_default_graph()
    op_names = [op.name for op in graph.get_operations()]
    # convert the protobuf GraphDef to a GraphDef that has no variables but just constants with the
    # current values.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(), op_names)

    # dump GraphDef to file
    graph_io.write_graph(output_graph_def, './', name, as_text=False)
