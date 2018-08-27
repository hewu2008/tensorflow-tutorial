# encoding:utf8

import tensorflow as tf


class GPUTester(object):
    def __init__(self):
        pass

    def test_cpu_gpu(self):
        with tf.device("/cpu:0"):
            a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
            b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
        with tf.device("/gpu:0"):
            c = a + b
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(sess.run(c))


if __name__ == "__main__":
    tester = GPUTester()
    tester.test_cpu_gpu()
