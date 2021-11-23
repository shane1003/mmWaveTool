# """
# Helpers for evaluating models.
# """
# import numpy as np
# from .reptile import Reptile
# from .variables import weight_decay


# # pylint: disable=R0913,R0914
# def evaluate(sess, datetime_,tag, model, dataset,
#              weight_decay_rate=1,reptile_fn=Reptile,filename=""):
#     """
#     Evaluate a model on a dataset.
#     """
#     reptile = reptile_fn(sess,pre_step_op=weight_decay(weight_decay_rate))

#     loss, label, row, column, distance, prediction, fine_tune_vars, test_set = \
#         reptile.evaluate(dataset, model.input_ph, model.label_ph, model.distance_ph, model.minimize_op, model.loss,
#                          model.predictions, )

#     if np.isnan(loss):
#         return

#     return loss
