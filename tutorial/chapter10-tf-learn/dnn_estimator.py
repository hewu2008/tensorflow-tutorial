# encoding:utf8
import tensorflow as tf


def _input_fn(num_epochs=None):
    features = {'age': tf.train.limit_epochs(tf.constant([[.8], [.2], [.1]]),
                                             num_epochs=num_epochs),
                'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                            indices=[[0, 0], [0, 1], [2, 0]],
                                            shape=[3, 2])}
    return features, tf.constant([[1], [0], [0]], dtype=tf.int32)


if __name__ == "__main__":
    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(language_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]
