import os
import warnings

import tensorflow as tf
import models.basic_cnn as model
from data_io import InputFnFactory, _get_data
import datetime

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    log_dir = "logs"
    current_run_subdir = os.path.join(
        "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_dir = os.path.join(log_dir, model.name, "106x128x110")#current_run_subdir)

    run_config = tf.estimator.RunConfig(model_dir=model_dir)

    params = tf.contrib.training.HParams(
        target_shape=(106, 128, 110),
        model_dir=model_dir
    )

    ds = InputFnFactory(target_shape=params.target_shape,
                        n_epochs=100,
                        train_src_folder="D:/data/PAC_Data/PAC_data/train",
                        train_cache_prefix="D:/drive/workspace/marvin/cache_train_hires",
                        eval_src_folder="D:/data/PAC_Data/PAC_data/eval",
                        eval_cache_prefix="D:/drive/workspace/marvin/cache_eval_hires",
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

    train_spec = tf.estimator.TrainSpec(input_fn=ds.train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=ds.eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=1200)

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       params=params,
                                       config=run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)