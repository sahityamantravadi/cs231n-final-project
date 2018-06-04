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

from source_code.data_io import Dataset_Pipeline, _get_data

devices = ['/gpu:0', '/gpu:1']

class_type=True #true if QC, false if site

if class_type:
    import source_code.models.basic_qc_cnn as model
    train_cache_prefix="/home/smantra/finalproject/cache_train_qc/"
    eval_cache_prefix="/home/smantra/finalproject/cache_eval_qc//"
    d = devices[0]
else:
    import source_code.models.basic_site_cnn as model
    train_cache_prefix="/home/smantra/finalproject/cache_train_sites/"
    eval_cache_prefix="/home/smantra/finalproject/cache_eval_sites/"
    d = devices[1]

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    log_dir = "../logs"
    current_run_subdir = os.path.join(
        "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#    model_dir = os.path.join(log_dir, model.name, "106x128x110")#current_run_subdir)
    model_dir = os.path.join(log_dir, model.name, '5292018')

    run_config = tf.estimator.RunConfig(model_dir=model_dir)

    params = tf.contrib.training.HParams(
        target_shape=(106, 128, 110),
        model_dir=model_dir
    )

    ds = Dataset_Pipeline(target_shape=params.target_shape,
                          n_epochs=10,
                          train_src_folder="/home/smantra/finalproject/data/",
                          train_cache_prefix=train_cache_prefix,
                          eval_src_folder="/home/smantra/finalproject/eval/",
                          eval_cache_prefix=eval_cache_prefix,
                          batch_size=4,
                          class_type=class_type
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
                                      balance_dataset=False,
                                      class_type=True
                                     )

            train_dataset = train_dataset.make_one_shot_iterator()
            while True:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        features, labels = sess.run(train_dataset.get_next())
                except tf.errors.OutOfRangeError:
                    break
        print("Finished preprocessing the training set")

    train_spec = tf.estimator.TrainSpec(input_fn=ds.train_input_fn,
                                        max_steps=20000,
                                       )
    eval_spec = tf.estimator.EvalSpec(input_fn=ds.eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=600)

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       params=params,
                                       config=run_config)
    
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
        with tf.device(d):
            sess.run(tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec))