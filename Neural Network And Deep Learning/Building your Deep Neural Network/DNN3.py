# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:37:10 2018

@author: wmy
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
  
class DeepNeuralNetwork:
    
    def __init__(self, layer_list):
        self.Parameters_Init(self, layer_list)
        pass
    
    def Parameters_Init(self, layer_list):
        self.layer_list = layer_list[:]
        np.random.seed(3)
        self.parameters = {}
        self.L = len(layer_list) - 1
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layer_list[l], \
                       layer_list[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layer_list[l], 1))
            assert(self.parameters['W' + str(l)].shape == (layer_list[l], \
                                   layer_list[l-1]))
            assert(self.parameters['b' + str(l)].shape == (layer_list[l], 1))
        return self.parameters
    
    def Forward_Z(A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        cache = (A_prev, W, b)
        return Z, cache
    
    def Sigmoid(Z):
        A = 1/(1 + np.exp(-Z))
        cache = Z
        return A, cache
    
    def ReLU(Z):
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache

    def Forward_Activation(self, A_prev, W, b, activation_function):
        if activation_function == "sigmoid":
            Z, A_prev_W_b_cache = self.Forward_Z(A_prev, W, b)
            A, Z_cache = self.Sigmoid(Z)
        elif activation_function == "relu":
            Z, A_prev_W_b_cache = self.Forward_Z(A_prev, W, b)
            A, Z_cache = self.ReLU(Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (A_prev_W_b_cache, Z_cache)
        return A, cache

    def Forward_Propagation(self, X):
        self.m = X.shape[1]
        self.caches = []
        A = X
        assert(self.L == len(self.parameters) // 2)
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A, cache = self.Forward_Activation(self, A_prev, W, b, "relu")
            self.caches.append(cache)
        WL = self.parameters['W' + str(self.L)]
        bL = self.parameters['b' + str(self.L)]
        self.AL, cache = self.Forward_Activation(self, A, WL, bL, "sigmoid")    
        self.caches.append(cache)
        assert(self.AL.shape == (1, X.shape[1]))
        return self.AL, self.caches
    
    def Compute_Cost(self, Y):
        assert(self.m == Y.shape[1])
        cost = -1/self.m * np.sum(np.dot(Y, np.log(self.AL).T) + \
                                  np.dot(1 - Y, np.log(1 - self.AL).T))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        self.cost = cost
        return self.cost
        
    def Backward_Z(dZ, A_prev_W_b_cache):
        A_prev, W, b = A_prev_W_b_cache
        m = float(A_prev.shape[1])
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db
    
    def Derivative_Function_Sigmoid(dA, Z_cache):
        Z = Z_cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
    
    def Derivative_Function_ReLU(dA, Z_cache):
        Z = Z_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ
    
    def Backward_Activation(self, dA, cache, activation_function):
        A_prev_W_b_cache, Z_cache = cache
        if activation_function == "relu":   
            dZ = self.Derivative_Function_ReLU(dA, Z_cache)
            dA_prev, dW, db = self.Backward_Z(dZ, A_prev_W_b_cache)       
        elif activation_function == "sigmoid":
            dZ = self.Derivative_Function_Sigmoid(dA, Z_cache)
            dA_prev, dW, db = self.Backward_Z(dZ, A_prev_W_b_cache)
        return dA_prev, dW, db
    
    def Backward_Propagation(self, Y):
        self.grads = {}
        assert(self.L == len(self.caches))
        assert(self.m == self.AL.shape[1])
        L = self.L
        Y = Y.reshape(self.AL.shape)
        dAL = - (np.divide(Y, self.AL) - np.divide(1 - Y, 1 - self.AL))
        current_cache = self.caches[L - 1]
        self.grads["dA" + str(L)], self.grads["dW" + str(L)], \
        self.grads["db" + str(L)] = \
        self.Backward_Activation(self, dAL, current_cache, 'sigmoid')
        for l in reversed(range(L - 1)):
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = \
            self.Backward_Activation(self, self.grads["dA" + str(l + 2)], \
                                           self.caches[l], 'relu')
            self.grads["dA" + str(l + 1)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp
        return self.grads
    
    def Update_Parameters(self, learning_rate):
        assert(self.L == len(self.parameters) // 2)
        L = self.L
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - \
            learning_rate * self.grads["dW" + str(l + 1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - \
            learning_rate * self.grads["db" + str(l + 1)]    
        return self.parameters
    
    def Train(self, X, Y, iterations = 20000, learning_rate = 0.05, print_cost = False):
        for i in range(1, iterations + 1):
            self.Forward_Propagation(self, X)
            self.Backward_Propagation(self, Y)
            self.Update_Parameters(self, learning_rate)
            if print_cost and i % 1000 == 0:
                cost = self.Compute_Cost(self, Y)
                print ("Cost after iteration %i: %f" %(i, cost))
        print('finished!')
        finial_cost = self.Compute_Cost(self, Y)
        print ("Finial cost: %f" %finial_cost)
        return finial_cost
        
        
        
        
    
a = DeepNeuralNetwork
a.__init__(a,[6,6,2,1])
X = np.array([[12, 23],
     [45, 23],
     [11, 11],
     [44, 44],
     [21, 21],
     [12, 22]])
Y = np.array([[0,1]])
a.Train(a, X, Y, print_cost = True)
a.Train(a, X, Y, print_cost = True)
a.Train(a, X, Y, print_cost = True, iterations = 25950)
