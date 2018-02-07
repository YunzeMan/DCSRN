'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def conv3d(x, W):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
    return conv_3d

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)
 