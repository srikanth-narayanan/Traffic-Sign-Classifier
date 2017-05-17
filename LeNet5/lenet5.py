#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:26:36 2017

@author: srikanthnarayanan

This module contains the LeNet5 implementation as python class.
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten


class neuralLeNet(object):
    """
    This class contains the definition of a LeNet object which has all methods
    and defintions based on tensor flow to make a piple of LeNet5 convolution
    neural network
    """
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        """
        Constructor for the LeNet5 architecture
        """
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        self.X_test = X_test
        self.y_test = y_test
        
    def process_setup(self, EPOCHS=5, BATCH_SIZE=128):
        """
        setup process variables
        """
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
    
    def placeholder_setup(self, x_shape=(None,32,32,1), y_shape=(None), onehot_class=10):
        """
        setup placeholder variables
        """
        self.x = tf.placeholder(tf.float32, x_shape)
        self.y = tf.placeholder(tf.int32, y_shape)
        self.one_hot_y = tf.one_hot(y, onehot_class)
    
    def setup_train_pipe(self, learning_rate=0.001, x):
        """
        Setup training pipeline
        """
        self.learning_rate = learning_rate
        self.logits = self.convnet_setup(x)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)
    
    def _accuracy(self):
        """
        Method to calcuate accuracy of operation
        """
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()
    
    def evaluate(self, X_data, y_data):
        """
        Method to evaluate loss and accuracy of the given dataset
        """
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
        
    def convnet_setup(self, x, mu=0, sigma=0.1):
        """
        Main lenet setup
        """           
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.#
        #############################################################
        
        weights_ly1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma)) # (height, width, input_depth, output_depth)
        bias_ly1 = tf.Variable(tf.zeros(6)) # same as output1 depth
        strides1 = [1, 1, 1, 1] # (batch, height, width, depth)
        conv1 = tf.nn.conv2d(x, weights_ly1, strides = strides1, padding='VALID')
        conv1 = tf.nn.bias_add(conv1, bias_ly1)
    
        # Activation of Layer 1
        conv1_act = tf.nn.relu(conv1) #activation using a RELU function
    
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        ksize1 = [1, 2, 2, 1] # (batch_size, height, width, depth)
        strides_11 = [1, 2, 2, 1] # (batch, height, width, depth)
        max_pool1 = tf.nn.max_pool(conv1_act, ksize=ksize1, strides=strides_11, padding='VALID')
        
    
        # Layer 2: Convolutional. Output = 10x10x16.#
        #############################################
        
        weights_ly2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma)) # (height, width, input_depth, output_depth)
        bias_ly2 = tf.Variable(tf.zeros(16)) # same as output2 depth
        strides2 = [1, 1, 1, 1] # (batch, height, width, depth)
        conv2 = tf.nn.conv2d(max_pool1, weights_ly2, strides = strides2, padding='VALID')
        conv2 = tf.nn.bias_add(conv2, bias_ly2)
        
        # Activation of Layer2
        conv2_act = tf.nn.relu(conv2) #activation using a RELU function
    
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        ksize2 = [1, 2, 2, 1] # (batch_size, height, width, depth)
        strides_22 = [1, 2, 2, 1] # (batch, height, width, depth)
        max_pool2 = tf.nn.max_pool(conv2_act, ksize=ksize2, strides=strides_22, padding='VALID')
    
        # Flatten. Input = 5x5x16. Output = 400.
        flat_layer = flatten(max_pool2)
        
        # Layer 3: Fully Connected. Input = 400. Output = 120.#
        #######################################################
        
        weights_ly3 = tf.Variable(tf.truncated_normal(shape=(400, 120),mean=mu, stddev=sigma)) # (height, width)
        bias_ly3 = tf.Variable(tf.zeros(120)) # same as output2 depth
        full_con_ly3 = tf.add(tf.matmul(flat_layer, weights_ly3), bias_ly3)
        
        # Activation of Layer3
        full_con_ly3_act = tf.nn.relu(full_con_ly3)
    
        # Layer 4: Fully Connected. Input = 120. Output = 84.#
        ######################################################
        
        weights_ly4 = tf.Variable(tf.truncated_normal(shape=(120, 84),mean=mu, stddev=sigma)) # (height, width)
        bias_ly4 = tf.Variable(tf.zeros(84)) # same as output2 depth
        full_con_ly4 = tf.add(tf.matmul(full_con_ly3_act, weights_ly4), bias_ly4)
        
        # Activation of Layer4
        full_con_ly4_act = tf.nn.relu(full_con_ly4)
    
        # Layer 5: Fully Connected. Input = 84. Output = 10.#
        #####################################################
        
        weights_ly5 = tf.Variable(tf.truncated_normal(shape=(84, 10),mean=mu, stddev=sigma)) # (height, width)
        bias_ly5 = tf.Variable(tf.zeros(10)) # same as output2 depth
        logits = tf.add(tf.matmul(full_con_ly4_act, weights_ly5), bias_ly5)
        
        return logits
    
    def train(self, fname):
        """
        Method to train the LeNet Model
        """
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(self.X_train)
            
            print("Training...")
            print()
            for i in range(self.EPOCHS):
                X_train, y_train = shuffle(self.X_train, self.y_train)
                for offset in range(0, num_examples, self.BATCH_SIZE):
                    end = offset + self.BATCH_SIZE
                    batch_x, batch_y = self.X_train[offset:end], self.y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={x: batch_x, y: batch_y})
                    
                validation_accuracy = self.evaluate(X_validation, y_validation)
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()
                
            saver.save(sess, fname)
            print("Model saved")

if __name__ == "__main__":
    pass
            
            