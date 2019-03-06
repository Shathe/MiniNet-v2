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
############ MININET ############
#################################
'''

def module_mininet(inputs, filters, name, dilation_rate=(1, 1), training=True):


    with tf.variable_scope('module_' + name): # Name scope

        x = tf.layers.separable_conv2d(inputs, int(filters), (1,3), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=tf.nn.selu, dilation_rate=(1,1),
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x2 = tf.layers.separable_conv2d(x, int(filters), (3,1), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=None, dilation_rate=(1,1),
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x = tf.nn.selu(x2)
        x = tf.layers.separable_conv2d(x, int(filters), (3,1), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=tf.nn.selu, dilation_rate=dilation_rate,
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x = tf.layers.separable_conv2d(x, int(filters), (1,3), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=None, dilation_rate=dilation_rate,
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x = tf.add(x, x2)
        x = tf.contrib.nn.alpha_dropout(x, keep_prob=0.75)
        x = tf.add(x, inputs[:, :, :, :filters])
        x = tf.nn.selu(x)

    return x




def downsampling_mininet(inputs, filters, name, strides=(2, 2), kernels=(3, 3), training=True):
    with tf.variable_scope('downsampling_' + name):
        x = tf.layers.separable_conv2d(inputs, filters, kernels, strides=strides, padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=tf.nn.selu,
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        return x


def upsampling_mininet(inputs, filters, name, strides=(2, 2), kernels=(3, 3), training=True, last=False):
     with tf.variable_scope('upsampling_' + name):

        activation = tf.nn.selu
        if last:
            activation=None
        x = tf.layers.conv2d_transpose(inputs, filters, kernels, strides=strides, padding='same', kernel_initializer=tf.initializers.lecun_normal(),  activation=activation,
            kernel_regularizer=l2_regularizer(0.0001)) # there is also dilation_rate!

        return x


def MiniNet(input_x=None, n_classes=20, training=True):
    d1 = downsampling_mininet(input_x, 12, 'd1', strides=(2, 2), kernels=(3, 3), training=training)
    d2 = downsampling_mininet(d1, 24, 'd2', strides=(2, 2), kernels=(3, 3), training=training)
    d3 = downsampling_mininet(d2, 48, 'd2_1', strides=(2, 2), kernels=(3, 3), training=training)
    d4 = downsampling_mininet(d3, 96, 'd3', strides=(2, 2), kernels=(3, 3), training=training)
    m4 = module_mininet(d4, 96, 'm3', dilation_rate=(1, 1), training=training)
    m5 = module_mininet(m4, 96, 'm4', dilation_rate=(1, 2), training=training)
    m6 = module_mininet(m5, 96, 'm5', dilation_rate=(1, 4), training=training)
    m7 = module_mininet(m6, 96, 'm6', dilation_rate=(1, 8), training=training)

    d5 = downsampling_mininet(d4, 192, 'd5', strides=(2, 2), kernels=(3, 3), training=training)
    d5 = module_mininet(d5, 192, 'm7d5', dilation_rate=(1, 1), training=training)
    d6 = downsampling_mininet(d5, 386, 'd6', strides=(2, 2), kernels=(3, 3), training=training)
    d6 = module_mininet(d6, 386, 'm7d6', dilation_rate=(1, 1), training=training)
    d6 = module_mininet(d6, 386, 'm7d6d6', dilation_rate=(1, 1), training=training)
    up_1 = upsampling_mininet(d6, 192, '_up11', training=training)
    up_1 = module_mininet(up_1, 192, 'm7d5up_1', dilation_rate=(1, 1), training=training)
    up_2 = upsampling_mininet(tf.concat([up_1, d5], axis=3), 96, 'new_1', training=training)

    up_concat = upsampling_mininet(tf.concat([m7, up_2, d4], axis=3), 96, 'new_2', training=training)

    m9 = module_mininet(up_concat, 48, 'm8', dilation_rate=(1, 1), training=training)

    up2 = upsampling_mininet(tf.concat((m9, d3), axis=3), 32, 'up2', training=training)
    up3 = upsampling_mininet(tf.concat((up2, d2), axis=3), 16, 'up22', training=training)

    out = upsampling_mininet(tf.concat((up3, d1), axis=3), n_classes, 'up3', training=training, last=True)
    return out




'''
####################################
############ MININET-v2 ############
####################################
'''

def factorized_res_module_one(input, n_filters,  is_training, dropout=0.3, dilation=1, l2=None, name="down"):
    x = tf.layers.separable_conv2d(input, n_filters, (3, 3), strides=1, padding='same', activation=None,
                         dilation_rate=dilation, use_bias=True,depthwise_initializer=winit, pointwise_initializer=winit,
                                   depthwise_regularizer=l2_regularizer(0.0002),
                                   pointwise_regularizer=l2_regularizer(0.0002))
    x = tf.layers.batch_normalization(x, training=is_training)
    x = dropout_layer(x, rate=dropout)
    if input.shape[3] == x.shape[3]:
        x = tf.add(x, input)
    x = tf.nn.relu(x)
    return x


def factorized_res_module(input, n_filters,  is_training, dropout=0.3, dilation=[1,1], l2=None, name="down"):
    x = factorized_res_module_one(input, n_filters,  is_training, dropout=dropout, dilation=dilation[0], l2=l2, name=name)

    x = factorized_res_module_one(x, n_filters,  is_training, dropout=dropout, dilation=dilation[1], l2=l2, name=name)

    return x


def downsample(input, n_filters_in, n_filters_out, is_training, bn=False, use_relu=False, l2=None, name="down"):
    filters_conv = n_filters_out - n_filters_in
    maxpool_use= True

    if n_filters_in >= n_filters_out:
        filters_conv = n_filters_out
        maxpool_use = False

    x = tf.layers.conv2d(input, filters_conv, 3, strides=2, padding='same', activation=None, kernel_initializer=winit,
                         dilation_rate=1, use_bias=True, kernel_regularizer=l2_regularizer(0.0002))
    if maxpool_use:
        y = maxpool(input, pool_size=2, strides=2)
        x = tf.concat([x, y], axis=-1, name="concat")
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.nn.relu(x)
    return x


def upsample(x, n_filters, is_training=False, last=False, l2=None, name="down"):
    x = tf.layers.conv2d_transpose(x, n_filters, 3, strides=2, padding='same',
                                   kernel_initializer=winit, activation=None,
                                   kernel_regularizer=l2_regularizer(0.0002))
    if not last:
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)

    return x





def MiniNet2(input_x,  n_classes, l2=None, is_training=False, upsampling=1):
    print('input shape')
    print(input_x.shape)
    x = downsample(input_x, n_filters_in=3, n_filters_out=16, is_training=is_training, l2=l2, name="d1")

    x = downsample(x, n_filters_in=16, n_filters_out=64, is_training=is_training, l2=l2, name="d2")
    x = factorized_res_module(x,n_filters=64, is_training=is_training, dilation=[1, 1], l2=l2, name="fres3", dropout=0.0)
    x = factorized_res_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres4", dropout=0.0)
    x = factorized_res_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres5", dropout=0.0)
    x = factorized_res_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres5", dropout=0.0)
    x = factorized_res_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres5", dropout=0.0)


    x = downsample(x,  n_filters_in=64, n_filters_out=128, is_training=is_training, l2=l2, name="d8")
    x = factorized_res_module(x, n_filters=128,is_training=is_training, dilation=[1, 1], l2=l2, name="fres9", dropout=0)
    x = factorized_res_module(x,n_filters=128, is_training=is_training, dilation=[1, 4], l2=l2, name="fres10", dropout=0)
    x = factorized_res_module(x, n_filters=128,is_training=is_training, dilation=[1, 8], l2=l2, name="fres11", dropout=0)
    x = factorized_res_module(x,n_filters=128, is_training=is_training, dilation=[1, 16], l2=l2, name="fres12", dropout=0)

    x = factorized_res_module(x, n_filters=128,is_training=is_training, dilation=[1, 1], l2=l2, name="fres9", dropout=0)
    x = factorized_res_module(x,n_filters=128, is_training=is_training, dilation=[1, 2], l2=l2, name="fres10", dropout=0)
    x = factorized_res_module(x, n_filters=128,is_training=is_training, dilation=[1, 8], l2=l2, name="fres11", dropout=0)
    x = factorized_res_module(x,n_filters=128, is_training=is_training, dilation=[1, 16], l2=l2, name="fres12", dropout=0)


    x = upsample(x, n_filters=64, is_training=is_training, l2=l2, name="up17")
    x3 = downsample(input_x, n_filters_in=3, n_filters_out=16, is_training=is_training, l2=l2, name="d7")
    x3 = downsample(x3, n_filters_in=16, n_filters_out=64, is_training=is_training, l2=l2, name="d7")
    x = x+x3

    x = factorized_res_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres19", dropout=0)
    x = factorized_res_module(x, n_filters=64,is_training=is_training, dilation=[1, 1], l2=l2, name="fres19", dropout=0)
    x = upsample(x, n_filters=n_classes, is_training=is_training, l2=l2, name="up23", last=True)

    if upsampling>1:
            x = tf.image.resize_bilinear(x, size=[x.shape[1] * upsampling, x.shape[2] * upsampling], align_corners=True)

    print('output shape')
    print(x.shape)
    return x