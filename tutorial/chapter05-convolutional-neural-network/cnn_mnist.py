# encoding:utf8

import tensorflow as tf


def cnn_model_fn(feature, lables, mode):
    """Model function for CNN"""
    # input layer
    input_layer = tf.reshape(feature['x'], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5],
                             padding='same', activation=tf.nn.relu)
