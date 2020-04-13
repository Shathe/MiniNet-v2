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

import os

def export_to_pb2(sess, name='model.pb'):
	export_path = os.path.join(
		tf.compat.as_bytes(name)
	)
	builder = tf.saved_model.builder.SavedModelBuilder(export_path)
	feature_configs = {
		'x': tf.FixedLenFeature(shape=[], dtype=tf.string),
		'y': tf.FixedLenFeature(shape=[], dtype=tf.string)
	}
	serialized_example = tf.placeholder(tf.string, name="tf_example")
	tf_example = tf.parse_example(serialized_example, feature_configs)
	x = tf.identity(tf_example['x'], name='x')
	y = tf.identity(tf_example['y'], name='y')
	predict_input = x
	predict_output = y
	predict_signature_def_map = tf.saved_model.signature_def_utils.predict_signature_def(
		inputs={
			tf.saved_model.signature_constants.PREDICT_INPUTS: predict_input
		},
		outputs={
			tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_output
		}
	)

	legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
	builder.add_meta_graph_and_variables(
		sess=sess,
		tags=[tf.saved_model.tag_constants.SERVING],
		signature_def_map={
			"predict_signature_map": predict_signature_def_map
		},
		legacy_init_op=legacy_init_op,
		assets_collection=None
	)
	builder.save()