'''
author: MANYZ
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
import util
from layers import (weight_variable, weight_variable_devonc, conv3d, pixel_wise_softmax_2, cross_entropy)

def create_conv_net(x, channels=1, layers=4, growth_rate=24, filter_size=3, summaries=True):
    """
    Creates a new convolutional dcsrn for the given parametrization.
    
    :param x: input tensor, shape [?,depth,height,width,channels]
    :param channels: number of channels in the input image, default is 1
    :param layers: number of layers in the dense unit, default is 4
    :param growth_rate: number of features in the dense layers
    :param filter_size: size of the convolution filter, default is [3 x 3 x 3]
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers}, growth_rate {growth_rate}, filter size {filter_size}x{filter_size}x{filter_size}".format(layers=layers, growth_rate=growth_rate, filter_size=filter_size))
    
    # nx = tf.shape(x)[1]
    # ny = tf.shape(x)[2]
    # x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    # in_node = x_image
    # batch_size = tf.shape(x_image)[0]
    stddev = np.sqrt(2 / (filter_size**3 * growth_rate))


    # Placeholder for the input image, of pattern "NDHWC"
    in_node = x

    # first 2k layer
    w0 = weight_variable([filter_size, filter_size, filter_size, channels, 2 * growth_rate], stddev)
    in_node = conv3d(in_node, w0)
    
    # Use batch normalization, bias is not needed
    weights = []
    convs = []
    weights.append(w0)
    # densely-connected block with 4 units
    for layer in range(0, layers):
        in_features = (2 + layer) * growth_rate
        w1 = weight_variable([filter_size, filter_size, filter_size, in_features, growth_rate], stddev)
        last_node = in_node
        # concat the dense layers
        for conv in convs:
            in_node = tf.concat([in_node, conv], 4)
        
        # Batch Normalization layer
        bn = tf.layers.batch_normalization(in_node, 0)
        # ELU layer
        elu = tf.nn.elu(bn)
        # 3D conv layer
        conv1 = conv3d(elu, w1)
        in_node = conv1
        # Add last layer into convs, for dense connection use
        convs.append(last_node)
        
        weights.append(w1)


    # Last layer, output the SR image   
    in_features = (2 + layers) * growth_rate
    w2 = weight_variable([filter_size, filter_size, filter_size, in_features, channels], stddev)
    for conv in convs:
        in_node = tf.concat([in_node, conv], 4)
    output = conv3d(in_node, w2)

    weights.append(w2)
    output_map = output;
    
    return output, weights


class DCSRN(object):
    """
    A dcsrn implementation
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """
    
    def __init__(self, channels=1, cost_kwargs={}, **kwargs):
        tf.reset_default_graph()
        
        self.summaries = kwargs.get("summaries", True)
        
        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        
        logits, self.variables = create_conv_net(self.x, channels)
        
        self.cost = tf.reduce_sum(tf.pow(logits - self.x, 2))/(n_instances)
        
        self.gradients_node = tf.gradients(self.cost, self.variables)
        
        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
        
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)
            
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))

        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
            
        return prediction
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)