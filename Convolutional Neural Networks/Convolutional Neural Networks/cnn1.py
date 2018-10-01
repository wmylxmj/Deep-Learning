# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:52:13 2018

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
        self.filters_list = filters_list
        self.filters = {}
        tf.set_random_seed(1)   
        with tf.variable_scope(self.identifier):     
            for i in range(1, len(filters_list) + 1):
                self.filters['W' + str(i)] = \
                tf.get_variable('W' + str(i), list(filters_list[i - 1]), initializer=tf.contrib.layers.xavier_initializer(seed = 0))
                pass
            pass
        pass
    
    def forward_propagation(self, X, maxpool_window_list):
        self.maxpool_window_list = maxpool_window_list
        P_prev = X
        for i in range(1, len(self.filters_list) + 1):
            print(i)
            Wi = self.filters['W' + str(i)]
            Zi = tf.nn.conv2d(P_prev, Wi, strides=[1,1,1,1], padding='SAME')
            Ai = tf.nn.relu(Zi)
            s = maxpool_window_list[i - 1]
            print(s)
            Pi = tf.nn.max_pool(Ai, [1,s,s,1], strides=[1,s,s,1], padding='SAME')
            P_prev = Pi
            pass
        n_y = self.io_information['n_y']
        Pi = tf.contrib.layers.flatten(Pi)
        ZL = tf.contrib.layers.fully_connected(Pi, n_y, activation_fn=None)
        self.ZL = ZL
        return ZL

    def compute_cost(self, Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ZL,labels=Y))
        self.cost = cost  
        return cost
            
  

tf.reset_default_graph()
c = ConvolutionalNeuralNetwork('c')
c.create_placeholders(64, 64, 3, 6)
print(c.X)
print(c.Y)
with tf.Session() as sess_test:
    filters = c.initialize_filters([(4, 4, 3, 8), (2, 2, 8, 16)])
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(c.filters["W1"].eval()[1,1,1]))
    print("W2 = " + str(c.filters["W2"].eval()[1,1,1]))
    pass

print(c.filters['W1'])
print(c.filters['W2'])

with tf.Session() as sess:
    np.random.seed(1)   
    ZL = c.forward_propagation(c.X, [8, 4])
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(ZL, {c.X: np.random.randn(2,64,64,3), c.Y: np.random.randn(2,6)})
    print("ZL = " + str(a))
    
    
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = c.create_placeholders(64, 64, 3, 6)
    parameters = c.initialize_filters([(4, 4, 3, 8), (2, 2, 8, 16)])
    Z3 = c.forward_propagation(X, [8, 4])
    cost = c.compute_cost(Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
    print("cost = " + str(a))