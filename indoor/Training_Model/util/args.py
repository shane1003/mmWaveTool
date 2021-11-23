"""
Command-line argument parsing.
"""
#设置了一堆的参数
import argparse
from functools import partial

import tensorflow.compat.v1 as tf

from .reptile import Reptile
def argument_parser(datetime):
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='model_checkpoint/'+datetime)
    parser.add_argument('--train-shots', help='shots in a training batch', default=1, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=8, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=10, type=int)
    parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=16, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=50000, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=1, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=100, type=int)
    parser.add_argument('--weight-decay', help='weight decay rate', default=1, type=float)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    return parser

def model_kwargs(parsed_args):
    """
    Build the kwargs for model constructors from the
    parsed command-line arguments.
    """
    res = {'learning_rate': parsed_args.learning_rate}
    if parsed_args.sgd:
        res['optimizer'] = tf.train.GradientDescentOptimizer
    return res

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'meta_step_size': parsed_args.meta_step,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'reptile_fn': _args_reptile()
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'weight_decay_rate': parsed_args.weight_decay,
        'reptile_fn': _args_reptile(),
        'filename':parsed_args.checkpoint
    }

def _args_reptile():
    return Reptile
