#!/bin/python
import tensorflow as tf

from data_io import _get_data
import numpy as np

with tf.Session() as sess:
    train_dataset = _get_data(nthreads=4,
                              batch_size=10,
                              src_folder="/home/smantra/finalproject/data_small",
                              n_epochs=1,
                              cache_prefix=None,
                              shuffle=True,
                              target_shape=(32, 32, 32))
    
    ds_iterator = train_dataset.make_one_shot_iterator()
    filenames = []
    count = 0

    import time

    start = time.time()
    print("hello")

    while True:
        try:
            print(count)
            features, labels = sess.run(ds_iterator.get_next())
            qc, site = labels
            print(qc)
            print(site)
            count += 1
        except tf.errors.OutOfRangeError:
            break
    end = time.time()
    print(end - start)
    print(count)
