import datetime
from glob import glob
import math
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import image
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import timeit
import warnings
#%matplotlib inline

import os
import warnings

import tensorflow as tf
from data_io import Dataset_Pipeline, _get_data
import models.basic_cnn as model
import datetime

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    log_dir = "logs"
    current_run_subdir = os.path.join(
        "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    model_dir = os.path.join(log_dir, model.name, '5272018')

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_secs=1800
    )


    params = tf.contrib.training.HParams(
        target_shape=(106, 128, 110),
        model_dir=model_dir
    )

    
    ds = Dataset_Pipeline(target_shape=params.target_shape,
                        n_epochs=10,
                        train_src_folder="/home/smantra/finalproject/data/",
                        train_cache_prefix="/home/smantra/finalproject/cache_train_hires/",
                        eval_src_folder="/home/smantra/finalproject/eval/",
                        eval_cache_prefix="/home/smantra/finalproject/cache_eval_hires/",
                        batch_size=4
                       )

    # Workaround for cache iterator concurency issues. Iterate over the whole
    # training dataset without counterbalancing to make sure everything is
    # preprocessed and cached


    if not os.path.exists(ds.train_cache_prefix + ".index"):
        print("Preprocessing the training set")
        with tf.Session() as sess:
            train_dataset = _get_data(batch_size=ds.batch_size,
                                      src_folder=ds.train_src_folder,
                                      n_epochs=1,
                                      cache_prefix=ds.train_cache_prefix,
                                      shuffle=False,
                                      target_shape=params.target_shape,
                                      balance_dataset=False)

            train_dataset = train_dataset.make_one_shot_iterator()
            while True:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        features, labels = sess.run(train_dataset.get_next())
                except tf.errors.OutOfRangeError:
                    break
        print("Finished preprocessing the training set")