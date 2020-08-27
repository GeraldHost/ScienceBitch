#! /usr/bin/env python
# creates a dataset from imdb/train/{neg|pos}
import tensorflow as tf

files = ['neg', 'pos']

def labeler(v, label):
    return v, tf.cast(label, tf.int64)

datasets = []
for i, path in enumerate(files):
    list_files = tf.data.Dataset.list_files("./imdb/train/%s/*.txt" % path)
    dataset = tf.data.TextLineDataset(list_files)
    labeled_dataset = dataset.map(lambda x: labeler(x, i))
    datasets.append(labeled_dataset)

# print example of the data set
for dataset in datasets:
    for v, l in dataset.take(1):
        print(v)
