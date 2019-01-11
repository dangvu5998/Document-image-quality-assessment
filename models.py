'''
DNN model for quality document image assessment
'''
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf

def build_cnn_model():
    model = Sequential()
    model.add(layers.Conv2D(filters=40, kernel_size=(5, 5), padding='valid', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))
    model.add(layers.Conv2D(filters=80, kernel_size=(5, 5), padding='valid'))
    def min_max_pool2d(x):
        max_x = K.max(K.max(x, axis=1), axis=1)
        min_x = K.min(K.min(x, axis=1), axis=1)
        return K.concatenate([max_x, min_x]) # concatenate on channel

    def min_max_pool2d_output_shape(input_shape):
        return (None, input_shape[-1]*2)

    # replace maxpooling layer
    model.add(layers.Lambda(min_max_pool2d, output_shape=min_max_pool2d_output_shape))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1))
    return model

def cnn_model_fn(features, labels, mode):
    '''Build model and integrate with estimator'''
    input_layer = tf.reshape(features, [-1, 48, 48, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=40,
        kernel_size=[5, 5],
        padding='valid'
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=80,
        kernel_size=[5, 5],
        padding='valid',
    )

    max_pool2 = tf.reduce_max(tf.reduce_max(conv2, axis=1), axis=1)
    min_pool2 = tf.reduce_min(tf.reduce_min(conv2, axis=1), axis=1)
    minmax_pool = tf.concat([max_pool2, min_pool2], axis=1)
    dense1 = tf.layers.dense(minmax_pool, 1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, 1024, activation=tf.nn.relu)
    output = tf.layers.dense(dense2, 1)
    output = tf.squeeze(output)

    predictions = {
        'quality': output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions)

    loss = tf.losses.absolute_difference(labels, output)
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = 0.0001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'mean_absolute_error': tf.metrics.mean_absolute_error(labels, output)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
