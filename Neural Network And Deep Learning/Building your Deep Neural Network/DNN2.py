# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:09:31 2018

@author: wmy
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *

class DeepNeuralNetwork:
    
    def __init__(self):
        print('please input the layer list')
        pass    
    
    def Parameters_Init(self, layer_list):
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
            Z, linear_cache = self.Forward_Z(A_prev, W, b)
            A, activation_cache = self.Sigmoid(Z)
        elif activation_function == "relu":
            Z, linear_cache = self.Forward_Z(A_prev, W, b)
            A, activation_cache = self.ReLU(Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
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
        assert(self.AL.shape == (1,X.shape[1]))
        return self.AL, self.caches
    
    def Compute_Cost(self, Y):
        self.m = Y.shape[1]
        cost = -1/self.m * np.sum(np.dot(Y, np.log(self.AL).T) + \
                                  np.dot(1 - Y, np.log(1 - self.AL).T))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        self.cost = cost
        return self.cost
    
    def Backward_Z(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        #dW = A_prev * dZ
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db
    
    def Derivative_Function_Sigmoid(dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
    
    def Derivative_Function_ReLU(dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ
    
    def Backward_Activation(self, dA, cache, activation_function):
        linear_cache, activation_cache = cache
        if activation_function == "relu":   
            dZ = self.Derivative_Function_ReLU(dA, activation_cache)
            dA_prev, dW, db = self.Backward_Z(dZ, linear_cache)       
        elif activation_function == "sigmoid":
            dZ = self.Derivative_Function_Sigmoid(dA, activation_cache)
            dA_prev, dW, db = self.Backward_Z(dZ, linear_cache)
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
        
a = DeepNeuralNetwork
a.__init__(a)
a.Parameters_Init(a,[5,4,1])
print("W1 = " + str(a.parameters["W1"]))
print("b1 = " + str(a.parameters["b1"]))
print("W2 = " + str(a.parameters["W2"]))
print("b2 = " + str(a.parameters["b2"]))
X = np.array([[12, 23],
     [45, 23],
     [11, 11],
     [44, 44],
     [21, 21]])
Y = np.array([[0,1]])
print(a.Forward_Propagation(a, X))
print(a.Backward_Propagation(a, Y))
print(a.Compute_Cost(a, Y))
print(a.Update_Parameters(a, 0.05))