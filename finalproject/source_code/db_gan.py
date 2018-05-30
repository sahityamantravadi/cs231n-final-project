from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from data_io import Dataset_Pipeline, _get_data

# import glob
# import random
# import collections
# import math
# import time



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def leaky_relu(x, a=0.1):
    return tf.maximum(a*x, x)

def G_conv(batch_input, out_channels):
    return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding="same")

def D_conv(batch_input, out_channels, stride):
    return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding="same")

def D_max_pool(batch_input, stride):
    return tf.layers.max_pooling3d(batch_input, 2, 2)

def G_conv_transpose(batch_input, out_channels):
    return tf.layers.conv3d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding="same", kernel_initializer=initializer)

def batchnorm(batch_input):
    return tf.layers.batch_normalization(batch_input, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def generator(G_in, G_out_channels):
    layers = []

    # encoder_1
    with tf.variable_scope("encoder_1"):
        conv_out = G_conv(G_in, ngf)
        
#        output = gen_conv(G_in, ngf)
        layers.append(conv_out)

    layer_nfilters = [
        ngf * 2, # encoder_2
        ngf * 4, # encoder_3
        ngf * 8, # encoder_4
        ngf * 8, # encoder_5
        ngf * 8, # encoder_6
        ngf * 8, # encoder_7
        ngf * 8, # encoder_8
    ]

    for out_n in layer_nfilters:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.1)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = G_conv(G_in, out_n)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8
        (ngf * 8, 0.5),   # decoder_7
        (ngf * 8, 0.5),   # decoder_6
        (ngf * 8, 0.0),   # decoder_5
        (ngf * 4, 0.0),   # decoder_4
        (ngf * 2, 0.0),   # decoder_3
        (ngf, 0.0),       # decoder_2
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                decoder_input = layers[-1]
            else:
                decoder_input = tf.concat([layers[-1], layers[skip_layer]], axis=4)

            rectified = tf.nn.relu(decoder_input)

            decoder_output = G_conv_transpose(rectified, out_channels)

            output = batchnorm(decoder_output)
            
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1
    with tf.variable_scope("decoder_1"):
        dec_1_input = tf.concat([layers[-1], layers[0]], axis=4)
        rectified = tf.nn.relu(dec_1_input)
        output = G_conv_transpose(rectified, G_out_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def site_discriminator(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []

    D_input = tf.concat([discrim_inputs, discrim_targets], axis=4)

    # layer_1:
    with tf.variable_scope("layer_1"):
        convolved = D_conv(D_input, ndf, stride=2)
        rectified = lrelu(convolved, 0.1)
        layers.append(rectified)

    # layer_2:
    # layer_3:
    # layer_4:
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            #stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = D_conv(layers[-1], out_channels, stride=1)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.1)
            layers.append(rectified)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = D_conv(rectified, out_channels=1, stride=1)
        output = tf.layers.dense(fc1, 17)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]

def qc_discriminator(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []

    D_input = tf.concat([discrim_inputs, discrim_targets], axis=4)

    # layer_1:
    with tf.variable_scope("layer_1"):
        convolved = D_conv(D_input, ndf, stride=2)
        rectified = lrelu(convolved, 0.1)
        layers.append(rectified)

    # layer_2:
    # layer_3:
    # layer_4:
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            #stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = D_conv(layers[-1], out_channels, stride=1)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.1)
            layers.append(rectified)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = D_conv(rectified, out_channels=1, stride=1)
        output = tf.layers.dense(fc1, 2)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]

