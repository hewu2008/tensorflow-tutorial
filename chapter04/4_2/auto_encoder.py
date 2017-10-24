# encoding:utf8

import numpy as np
import sklearn.preprocessing as prep 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in, fan_out, constant = 1):
	
