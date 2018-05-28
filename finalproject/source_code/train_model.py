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
%matplotlib inline

import os
import warnings

import tensorflow as tf
import models.basic_cnn as model
from src.data_io import Dataset_Pipeline, _get_data
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

    ds = Dataset_Pipeline(
        target_shape=params.target_shape,
        n_epochs=10,
        train_src_folder="/home/smantra/finalproject/data/",
        train_cache_prefix="/home/smantra/finalproject/cache_train_hires/",
        eval_src_folder="/home/smantra/finalproject/eval/",
        eval_cache_prefix="/home/smantra/finalproject/cache_eval_hires/",
        batch_size=4
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=ds.train_input_fn,
        steps=200
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=ds.eval_input_fn,steps=None,
        reuse=True,
        start_delay_secs=0,
        throttle_secs=1200
    )

    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        params=params,
        config=run_config
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)