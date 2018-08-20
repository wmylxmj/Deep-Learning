# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:32:39 2018

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
    
    def Forward_Propagation(self, X, parameters):
        self.caches = []
        A = X
        assert(self.L == len(parameters) // 2)
        for l in range(1, self.L):
            A_prev = A 
            A, cache = self.Forward_Activation(self, A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            self.caches.append(cache)
        self.AL, cache = self.Forward_Activation(self, A, parameters['W' + str(self.L)], parameters['b' + str(self.L)], "sigmoid")    
        self.caches.append(cache)
        assert(self.AL.shape == (1,X.shape[1]))
        return self.AL, self.caches
    
    def Compute_Cost(self, AL, Y):
        self.m = Y.shape[1]
        cost = -1/self.m * np.sum(np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        self.cost = cost
        return self.cost
        
        
a = DeepNeuralNetwork
a.__init__(a)
parameters = a.Parameters_Init(a, layer_list=[5,4,3])
print(parameters)
A, W, b = linear_forward_test_case()

Z, linear_cache = a.Forward_Z(A, W, b)
print("Z = " + str(Z))

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = a.Forward_Activation(a, A_prev, W, b, activation_function = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = a.Forward_Activation(a, A_prev, W, b, activation_function = "relu")
print("With ReLU: A = " + str(A))

X, parameters = L_model_forward_test_case()
AL, caches = a.Forward_Propagation(a, X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
print(a.caches)
print(a.L)
print(a.parameters)

Y, AL = compute_cost_test_case()

print("cost = " + str(a.Compute_Cost(a, AL, Y)))