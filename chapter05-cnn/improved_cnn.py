#encoding:utf8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='wight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

if __name__ == "__main__":
    max_steps = 3000
    batch_size = 138
    data_dir = '/tmp/cifar10_data/cifar-10-batchs-bin'

    cifar10.maybe_download_and_extract()
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])

    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1,1,1,1], padding='SAME')
    bias1 = tf.Variable(tf.float32, shape=[64])
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2,1 ], padding='SAME')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


