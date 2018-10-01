# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:55:51 2018

@author: wmy
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

class ConvolutionalNeuralNetwork():
    
    def __init__(self, identifier, name = None, description = None):
        self.identifier = identifier
        self.name = name
        self.description = description      
        pass
    
    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        self.io_information = {'n_H0':n_H0, 'n_W0':n_W0, 'n_C0':n_C0, 'n_y':n_y}        
        self.X = tf.placeholder(tf.float32,shape=[None, n_H0, n_W0, n_C0])
        self.Y = tf.placeholder(tf.float32,shape=[None, n_y])       
        return self.X, self.Y
    
    def initialize_filters(self, filters_list):
        tf.set_random_seed(1)   
        self.filters_list = filters_list
        self.filters = {}
        with tf.variable_scope(self.identifier):     
            for i in range(1, len(filters_list) + 1):
                self.filters['W' + str(i)] = \
                tf.get_variable('W' + str(i), list(filters_list[i - 1]), initializer=tf.contrib.layers.xavier_initializer(seed = 0))
                pass
            pass
        return self.filters
    
    def forward_propagation(self, X, maxpool_size_list):
        self.maxpool_size_list = maxpool_size_list
        P_prev = X
        for i in range(1, len(self.filters_list) + 1):
             Wi = self.filters['W' + str(i)]
             Zi = tf.nn.conv2d(P_prev, Wi, strides=[1,1,1,1], padding='SAME')
             Ai = tf.nn.relu(Zi)
             si = maxpool_size_list[i - 1]
             Pi = tf.nn.max_pool(Ai, [1,si,si,1], strides=[1,si,si,1], padding='SAME')
             P_prev = Pi
             pass                 
        n_y = self.io_information['n_y']
        Pi = tf.contrib.layers.flatten(Pi)
        ZL = tf.contrib.layers.fully_connected(Pi, n_y, activation_fn=None)
        self.ZL = ZL
        return ZL             
    
    pass

tf.reset_default_graph()

c = ConvolutionalNeuralNetwork('c')
c.create_placeholders(64, 64, 3, 6)

parameters = c.initialize_filters([(4, 4, 3, 8), (2, 2, 8, 16)])

with tf.Session() as sess_test:
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(c.filters["W1"].eval()[1,1,1]))
    print("W2 = " + str(c.filters["W2"].eval()[1,1,1]))
    pass

with tf.Session() as sess:
    np.random.seed(1)
    c.forward_propagation(c.X, [8, 4])
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(c.ZL, {c.X: np.random.randn(2,64,64,3), c.Y: np.random.randn(2,6)})
    print("ZL = " + str(a))
    
    
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1,[1,8,8,1],strides=[1,8,8,1],padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2,[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn=None)
    ### END CODE HERE ###

    return Z3

tf.reset_default_graph()

# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable('W1',[4, 4, 3, 8],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2',[2, 2, 8, 16],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

# GRADED FUNCTION: create_placeholders

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(tf.float32,shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32,shape=[None, n_y])
    ### END CODE HERE ###
    
    return X, Y

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = " + str(a))
    
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = " + str(a))