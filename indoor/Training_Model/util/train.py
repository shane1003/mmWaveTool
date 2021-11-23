"""
Training helpers for supervised meta-learning.
"""

from .reptile import Reptile
from .variables import weight_decay
import numpy as np
import os
import time
import tensorflow.compat.v1 as tf


def train(sess, model, dataset, save_dir,meta_step_size=0.1,
          meta_batch_size=1, meta_iters=2000, eval_interval=10, weight_decay_rate=1, time_deadline=None,
          reptile_fn=Reptile, log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):  # 文件夹不存在时创建相应的文件夹
        os.makedirs(save_dir)
    
    saver = tf.train.Saver()
    reptile = reptile_fn(sess, pre_step_op=weight_decay(weight_decay_rate))
    aver_loss_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('average loss', aver_loss_ph)
    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(os.path.join(save_dir, 'train step loss'), sess.graph)  # save_dir+"/"+TIMESTAMP
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)  # save_dir+"/"+TIMESTAMP
    
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())
    for i in range(meta_iters):
        # 开始训练过程
        aver_loss = 0
        loss = reptile.train_step(model.input_ph, model.label_ph, model.distance_ph, model.minimize_op, model.loss,
                                  meta_step_size=meta_step_size, meta_batch_size=meta_batch_size)
        if np.isnan(loss):
            print("end to train... loss is nan")
            return
        if i % eval_interval == 0:
            aver_loss= reptile.evaluate(dataset, model.input_ph,model.label_ph, model.distance_ph,model.minimize_op, model.loss, model.predictions,[model.out1,model.out2,model.out3],model.grad)

            summary = sess.run(merged, feed_dict={aver_loss_ph: aver_loss})
            train_writer.add_summary(summary, i)
            train_writer.flush()

            log_fn('batch %d: average train_loss=%f ' % (i, aver_loss))
        if i % 100 == 0 or i == meta_iters - 1:
            print("training------%d, loss: %f" % (i, loss))

            summary = sess.run(merged, feed_dict={aver_loss_ph: loss})
            trainWriter.add_summary(summary, i)
            trainWriter.flush()
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break
