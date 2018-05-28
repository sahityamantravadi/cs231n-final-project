import tensorflow as tf

name = "resnet_3d"

def model_fn(features, labels, mode, params):
    num_classes = 17
    net = {}
    
    net['features'] = tf.expand_dims(features, -1)

    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv3d(features, 32, 5,
                             activation=tf.nn.relu,
                             padding='valid')
    conv1 = tf.layers.dropout(conv1, rate=0.3,
                              training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv1 = tf.layers.max_pooling3d(conv1, 2, 2)

    # Convolution Layer with 64 filters and a kernel size of 3
    conv2 = tf.layers.conv3d(conv1, 52, 5,
                             activation=tf.nn.relu,
                             padding='valid')
    conv2 = tf.layers.dropout(conv2, rate=0.3,
                              training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv2 = tf.layers.max_pooling3d(conv2, 2, 2)

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv2)

    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 300)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate=0.3,
                            training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Output layer, class prediction
    out = tf.layers.dense(fc1, 17)

    pred_classes = tf.argmax(out, axis=1)
    pred_probas = tf.nn.softmax(out)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes)
    else:
        loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=out, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op}
        )
