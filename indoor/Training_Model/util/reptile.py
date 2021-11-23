"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

from skimage.transform import resize
from skimage import io
import os
import math
from .variables import (interpolate_vars, average_vars, VariableState)
from .omniglot import read_dataset
import random
import tensorflow.compat.v1 as tf

import numpy as np

PATH = "data/Beam"


class Reptile:

    def __init__(self, session, variables=None, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._pre_step_op = pre_step_op

    def train_step(self, input_ph, label_ph, distance_ph, minimize_op, losses, meta_step_size, meta_batch_size,):

        old_vars = self._model_state.export_variables()
        new_vars = []
        averloss = 0
        for t in range(meta_batch_size):
            taskBeam = t + 1
            task_set = read_dataset(PATH + str(taskBeam) + "/")
            inputs = np.array([i[0] for i in task_set])  # 这种就是取出一个list中指定的一列
            labels = np.array([i[1] for i in task_set]).astype('float32').reshape([-1, 1])
            distances = np.array([i[4] for i in task_set]).astype('float32').reshape([-1, 1])
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            op, loss = self.session.run([minimize_op, losses],
                                        feed_dict={input_ph: inputs, label_ph: labels, distance_ph: distances})
            averloss = averloss + loss
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))
        return averloss / 16

    '''define a function to plot conv output layer'''
    def evaluate(self,dataset,input_ph, label_ph, distance_ph, minimize_op, losses,
                 predictions, layers, grads):

        random.shuffle(dataset)
        old_vars = self._full_state.export_variables()
        inputs = np.array([i[0] for i in dataset])  # 这种就是取出一个list中指定的一列
        # 主要是我要计算loss #其实在test阶段是没有label的（原代码也是计算了准确率根据label，所以本质上还是利用了label）
        labels = np.array([i[1] for i in dataset]).astype('float32').reshape([-1, 1])
        distances = np.array([i[4] for i in dataset]).astype(
            'float32').reshape([-1, 1])
        row = np.array([i[2] for i in dataset]).reshape([-1, 1])
        col = np.array([i[3] for i in dataset]).reshape([-1, 1])

        loss, prediction, layer, grad = self.session.run(
            [losses, predictions, layers, grads],
            feed_dict={input_ph: inputs, label_ph: labels,
                       distance_ph: distances})
        self._full_state.import_variables(old_vars)

        
        return loss
    
