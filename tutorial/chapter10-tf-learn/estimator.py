#encoding:utf8
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
