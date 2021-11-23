"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)
IMAGE_SIZE = 40
dropout_rate = 0.5

class CNNModel:

    def __init__(self, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), name="images")
        self.distance_ph = tf.placeholder(tf.float32, shape=(None, 1), name="distance")

        with tf.name_scope("CONV"):
            out = tf.layers.conv2d(self.input_ph, 5, 5, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            self.out1 = tf.nn.relu(out)
            out = tf.layers.conv2d(self.out1, 5, 5, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            self.out2 = tf.nn.relu(out)
            out = tf.layers.conv2d(self.out2, 5, 5, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
            self.out3 = out

        with tf.name_scope("FC"):
            conv_out = tf.cast(out, tf.float32)
            flattened = tf.layers.flatten(conv_out)
            flattened = tf.concat([flattened, self.distance_ph], 1)
#             fc = tf.layers.dense(flattened, 1, activation=None)
#             fc = tf.nn.leaky_relu(fc)
#             fc = tf.nn.batch_normalization(fc)
            fc = tf.layers.dense(flattened, 1, activation=None)
            fc = tf.nn.leaky_relu(fc)

        self.logits = tf.layers.dropout(fc, rate=dropout_rate)

        self.label_ph = tf.placeholder(tf.float32, shape=(None, 1), name="label")
        self.loss = tf.sqrt(
            tf.losses.mean_squared_error(self.logits, self.label_ph))
        self.predictions = self.logits
        self.grad = tf.gradients(self.predictions, self.out3)[0]

        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)
