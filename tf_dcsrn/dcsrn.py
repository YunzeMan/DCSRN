'''
author: MANYZ
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
sys.path.append('../')
import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
from tf_dcsrn import util
from tf_dcsrn.layers import (weight_variable, conv3d, pixel_wise_softmax_2)
from ssim import tf_SSIM

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

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
        # in_node = tf.reshape(in_node, [-1, 64, 64, 64, in_features])
        bn = tf.contrib.layers.batch_norm(in_node, 0)
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
    """
    
    def __init__(self, channels=1, **kwargs):
        tf.reset_default_graph()
        
        self.summaries = kwargs.get("summaries", True)
        self.channels = channels
        # x is LR image, y is HR image
        self.x = tf.placeholder("float", shape=[None, None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, None, channels])
        self.logits, self.variables = create_conv_net(self.x, channels = channels)
        
        instance_number = tf.shape(self.logits)[0] * tf.shape(self.logits)[1] * tf.shape(self.logits)[2] * tf.shape(self.logits)[3]
        instance_number = tf.cast(instance_number, tf.float32)
        # L2 loss (Mean squared error)
        self.cost = tf.reduce_sum(tf.pow(self.logits - self.y, 2)) / instance_number
                
        
        logits2D = tf.reshape(self.logits, [1, 64, -1, 1])
        HR2D = tf.reshape(self.y, [1, 64, -1, 1])
        self.mean_ssim =  self._get_ssim(logits2D, HR2D)

        '''        
        batch_size = tf.shape(self.logits)[0]
        slice_number = tf.shape(self.logits)[1]
        num = batch_size * slice_number
        for i in range(0, batch_size):
            for j in range(0, slice_number):
                self.mean_ssim += self._get_ssim(self.logits[i, j:j+1, :, :, :], self.y[i, j:j+1, :, :, :]) / num
        '''
        
    def _get_ssim(self, result, hr_image):
        """
        Constructs the SSIM index, which measures the similarity of 2 input images

        :param result: final result generated from the low resolution image
        :param hr_image: high resolution image, serves as ground truth 
        """
        ssim_result = tf_SSIM.tf_ssim(result, hr_image)

        return ssim_result
    
    def predict(self, model_path, x_test):
        """
        Uses the model to get a HR for the test LR data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to perform super-resolution on.
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], self.channels))
            result_image = sess.run(self.logits, feed_dict={self.x: x_test, self.y: y_dummy})
            
        return result_image

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


class Trainer(object):
    """
    Trains a dcsrn instance
    
    :param net: the dcsrn instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (default adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    """
    
    verification_batch_size = 2
    
    def __init__(self, net, batch_size=2, optimizer="adam", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.00001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)
                
        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('mean ssim', self.net.mean_ssim)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init

    def train(self, data_provider, output_path, training_iters=3500, epochs=10, display_step=10, restore=False, write_graph=False, prediction_path = 'prediction'):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, output_path, restore, prediction_path)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            logging.info("Init graph over")
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            logging.info("Begin to fetch test data")
            test_x, test_y = data_provider(self.verification_batch_size)
            logging.info("Test data fetch over")
            self.store_prediction(sess, test_x, test_y, "_init")
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")
            
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)
                     
                    # Run optimization op (backprop)
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node), 
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y})
                    
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, batch_y)
                        
                    total_loss += loss
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, "epoch_%s"%epoch)
                    
                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")
            
            return save_path
        
    def store_prediction(self, sess, batch_x, batch_y, name):
        loss, ssim = sess.run((self.net.cost, self.net.mean_ssim), feed_dict={self.net.x: batch_x, 
                                                                                self.net.y: batch_y})
        logging.info("\n***********Begin testing**************\n")                                                                                
        logging.info("Testing data: Average L2 loss: {:.4f}, Average SSIM: {:.4f}".format(loss, ssim))
        
        logging.info("\n**************************************\n")                     

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("\n")
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, ssim = sess.run([self.summary_op, 
                                            self.net.cost,
                                            self.net.mean_ssim], 
                                            feed_dict={self.net.x: batch_x,
                                                        self.net.y: batch_y})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Training SSIM= {:.4f}".format(step,
                                                                                      loss,
                                                                                      ssim))
