import tensorflow as tf
import numpy as np
import threading
import time


def fifo_queue():
    # RandonShuffleQueue
    q = tf.FIFOQueue(2, "int32")

    init = q.enqueue_many(([0, 10],))

    x = q.dequeue()
    y = x + 1

    q_inc = q.enqueue([y])

    with tf.Session() as sess:
        init.run()
        for _ in range(5):
            v, _ = sess.run([x, q_inc])
            print(v)


def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d" % worker_id)
            coord.request_stop()
        else:
            print("Working on id: %d" % worker_id)
        time.sleep(1)


def test_coordinator():
    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]
    for t in threads: t.start()
    coord.join(threads)


def test_queue_runner():
    queue = tf.FIFOQueue(100, "float")
    enqueue_op = queue.enqueue([tf.random_normal([1])])
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

    tf.train.add_queue_runner(qr)
    out_tensor = queue.dequeue()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in range(3):
            print(sess.run(out_tensor)[0])
        coord.request_stop()
        coord.join(threads)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_record_files():
    num_shared = 2
    instances_per_shared = 2
    for i in range(num_shared):
        filename = '/path/to/data.tfrecords-%.5d-of-%.5d' % (i, num_shared)
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(instances_per_shared):
            example = tf.train.Example(features=tf.train.Features(feature={
                'i': _int64_feature(i),
                'j': _int64_feature(j)
            }))
            writer.write(example.SerializeToString())


def test_file_queue():
    files = tf.train.match_filenames_once("/path/to/data.tfrecord-*")
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'i': tf.FixedLenFeature([], tf.int64),
        "j": tf.FixedLenFeature([], tf.int64),
    })

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        print(sess.run(files))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(6):
            print(sess.run([features['i'], features['j']]))
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    test_queue_runner()
