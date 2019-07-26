from __future__ import print_function, division, unicode_literals
import tensorflow as tf



# USEFUL LAYERS
fc = tf.layers.dense
conv = tf.layers.conv2d
deconv = tf.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = tf.layers.max_pooling2d
dropout_layer = tf.layers.dropout
batchnorm = tf.layers.batch_normalization
winit = tf.contrib.layers.xavier_initializer()
repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope
l2_regularizer = tf.contrib.layers.l2_regularizer


'''
#################################
########### OPERATIONS ##########
#################################
'''
def residual_separable(input, n_filters,  is_training, dropout=0.3, dilation=1, l2=None, name="down"):
    x = tf.layers.separable_conv2d(input, n_filters, (3, 3), strides=1, padding='same', activation=None,
                        dilation_rate=dilation, use_bias=False, depthwise_initializer=winit, pointwise_initializer=winit,
                        pointwise_regularizer=l2_regularizer(0.00004))
    x = tf.layers.batch_normalization(x, training=is_training)
    x = dropout_layer(x, rate=dropout)
    if input.shape[3] == x.shape[3]:
        x = tf.add(x, input)
    x = tf.nn.relu(x)
    return x

def residual_separable_multi(input, n_filters,  is_training, dropout=0.3, dilation=1, l2=None, name="down"):
    input_b = tf.identity(input)
    d = tf.keras.layers.DepthwiseConv2D(3, strides=(1, 1), depth_multiplier=1, padding='same')
    x = d(input)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)

    d2= tf.keras.layers.DepthwiseConv2D(3, strides=(1, 1), depth_multiplier=1, padding='same')
    d2.dilation_rate = (dilation, dilation)
    x2 = d2(input)
    x2 = tf.layers.batch_normalization(x2, training=is_training)
    x2 = tf.nn.relu(x2)

    x +=x2

    x = tf.layers.conv2d(x, n_filters, 1, strides=1, padding='same', activation=None, kernel_initializer=winit,
                         dilation_rate=1, use_bias=False, kernel_regularizer=l2_regularizer(0.00004))
    x = tf.layers.batch_normalization(x, training=is_training)

    x = dropout_layer(x, rate=dropout)

    if input.shape[3] == x.shape[3]:
        x = tf.add(x, input_b)

    x = tf.nn.relu(x)
    return x


def encoder_module(input, n_filters,  is_training, dropout=0.3, dilation=[1,1], l2=None, name="down"):
    x = residual_separable(input, n_filters,  is_training, dropout=dropout, dilation=dilation[0], l2=l2, name=name)
    x = residual_separable(x, n_filters,  is_training, dropout=dropout, dilation=dilation[1], l2=l2, name=name)
    return x

def encoder_module_multiti(input, n_filters,  is_training, dropout=0.3, dilation=[1,1], l2=None, name="down"):
    x = residual_separable_multi(input, n_filters,  is_training, dropout=dropout, dilation=dilation[0], l2=l2, name=name)
    x = residual_separable_multi(x, n_filters,  is_training, dropout=dropout, dilation=dilation[1], l2=l2, name=name)
    return x

def upsample(x, n_filters, is_training=False, last=False, l2=None, name="down"):
    x = tf.layers.conv2d_transpose(x, n_filters, 3, strides=2, padding='same',use_bias=True,
                                   kernel_initializer=winit, activation=None,
                                   kernel_regularizer=l2_regularizer(0.00004))
    if not last:
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)

    return x

def downsample(input, n_filters_in, n_filters_out, is_training, bn=False, use_relu=False, l2=None, name="down"):
    
    maxpool_use = n_filters_in < n_filters_out

    if not maxpool_use:
        filters_conv = n_filters_out
    else:
        filters_conv = n_filters_out - n_filters_in

    x = tf.layers.conv2d(input, filters_conv, 3, strides=2, padding='same', activation=None, kernel_initializer=winit,
                         dilation_rate=1, use_bias=False, kernel_regularizer=l2_regularizer(0.00004))
    if maxpool_use:
        y = maxpool(input, pool_size=2, strides=2)
        x = tf.concat([x, y], axis=-1, name="concat")

    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    return x


'''
#################################
############ MININET ############
#################################
'''

def MiniNet(input_x=None, n_classes=20, is_training=True, upsampling=1):

    x = downsample(input_x, n_filters_in=3, n_filters_out=8, is_training=is_training, l2=l2_regularizer, name="d1")
    x = downsample(x, n_filters_in=8, n_filters_out=24, is_training=is_training, l2=l2_regularizer, name="d2")
    x = encoder_module(x,n_filters=24, is_training=is_training, dilation=[1, 1], l2=l2_regularizer, name="fres3", dropout=0.0)
    x = downsample(x, n_filters_in=24, n_filters_out=64, is_training=is_training, l2=l2_regularizer, name="d2")
    x = encoder_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2_regularizer, name="fres4", dropout=0.0)
    x = downsample(x,  n_filters_in=64, n_filters_out=128, is_training=is_training, l2=l2_regularizer, name="d8")

    x = encoder_module(x, n_filters=128,is_training=is_training, dilation=[1, 2], l2=l2_regularizer, name="fres9", dropout=0)
    x = encoder_module(x,n_filters=128, is_training=is_training, dilation=[1, 2], l2=l2_regularizer, name="fres10", dropout=0)
    x = encoder_module(x, n_filters=128,is_training=is_training, dilation=[1, 4], l2=l2_regularizer, name="fres11", dropout=0)

    x = upsample(x, n_filters=64, is_training=is_training, l2=l2_regularizer, name="up17")
    x = encoder_module(x,n_filters=64, is_training=is_training, dilation=[1, 1], l2=l2_regularizer, name="fres12", dropout=0)
    x = upsample(x, n_filters=24, is_training=is_training, l2=l2_regularizer, name="up17")
    x3 = downsample(input_x, n_filters_in=3, n_filters_out=8, is_training=is_training, l2=l2_regularizer, name="d7")
    x3 = downsample(x3, n_filters_in=8, n_filters_out=24, is_training=is_training, l2=l2_regularizer, name="d7")
    x = x+x3

    x = encoder_module(x, n_filters=24,is_training=is_training, dilation=[1, 1], l2=l2_regularizer, name="fres19", dropout=0)
    x = upsample(x, n_filters=n_classes, is_training=is_training, l2=l2_regularizer, name="up23", last=True)

    if upsampling > 1:
            x = tf.image.resize_bilinear(x, size=[x.shape[1] * upsampling, x.shape[2] * upsampling], align_corners=True)

    return x


'''
####################################
############ MININET-v2 ############
####################################
'''

def MiniNet2(input_x,  n_classes, l2=None, is_training=False, upsampling=1):
    
    x = downsample(input_x, n_filters_in=3, n_filters_out=16, is_training=is_training, l2=l2, name="d1")

    x = downsample(x, n_filters_in=16, n_filters_out=64, is_training=is_training, l2=l2, name="d2")
    x = encoder_module(x,n_filters=64, is_training=is_training, dilation=[1, 1], l2=l2, name="fres3", dropout=0.0)
    x = encoder_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres4", dropout=0.0)
    x = encoder_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres5", dropout=0.0)
    x = encoder_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres6", dropout=0.0)
    x = encoder_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres7", dropout=0.0)

    x = downsample(x,  n_filters_in=64, n_filters_out=128, is_training=is_training, l2=l2, name="d8")
    x = encoder_module_multi(x, n_filters=128,is_training=is_training, dilation=[1, 2], l2=l2, name="fres9", dropout=0.25)
    x = encoder_module_multi(x,n_filters=128, is_training=is_training, dilation=[1, 4], l2=l2, name="fres10", dropout=0.25)
    x = encoder_module_multi(x, n_filters=128,is_training=is_training, dilation=[1, 8], l2=l2, name="fres11", dropout=0.25)
    x = encoder_module_multi(x,n_filters=128, is_training=is_training, dilation=[1, 16], l2=l2, name="fres12", dropout=0.25)

    x = encoder_module_multi(x, n_filters=128,is_training=is_training, dilation=[1, 1], l2=l2, name="fres13", dropout=0.25)
    x = encoder_module_multi(x,n_filters=128, is_training=is_training, dilation=[1, 2], l2=l2, name="fres14", dropout=0.25)
    x = encoder_module_multi(x, n_filters=128,is_training=is_training, dilation=[1, 8], l2=l2, name="fres15", dropout=0.25)
    x = encoder_module_multi(x,n_filters=128, is_training=is_training, dilation=[1, 16], l2=l2, name="fres16", dropout=0.25)

    x = upsample(x, n_filters=64, is_training=is_training, l2=l2, name="up17")
    x3 = downsample(input_x, n_filters_in=3, n_filters_out=16, is_training=is_training, l2=l2, name="d7")
    x3 = downsample(x3, n_filters_in=16, n_filters_out=64, is_training=is_training, l2=l2, name="d7")
    x = x+x3

    x = encoder_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres19", dropout=0)
    x = encoder_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres19", dropout=0)
    x = upsample(x, n_filters=n_classes, is_training=is_training, l2=l2, name="up23", last=True)

    if upsampling > 1:
            x = tf.image.resize_bilinear(x, size=[x.shape[1] * upsampling, x.shape[2] * upsampling], align_corners=True)

    return x
