# encoding:utf8
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = "http://mattmahoney.net/dc/"
vocabulary_size = 50000
data_index = 0

def maybe_download(filename, expected_bytes):
    dest_filename = '../text_data/' + filename
    if not os.path.exists(dest_filename):
        dest_filename, _ = urllib.request.urlretrieve(url + filename, dest_filename)
    statinfo = os.stat(dest_filename)
    # print(dest_filename, ' file_size: ', statinfo.st_size)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with browser?')
    return dest_filename


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_directory = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_directory

# skip_window 单词最远可以联系的距离
# num_skip 每个单词生成的样本数量
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    # 读入span个单词作为初始值
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

if __name__ == "__main__":
    filename = maybe_download('text8.zip', 31344016)
    words = read_data(filename)
    print('Data size:', len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    del words
    print('Most common words +(UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])