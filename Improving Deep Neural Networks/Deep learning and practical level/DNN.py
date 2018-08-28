# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 20:13:21 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, \
initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, \
backward_propagation, update_parameters

class DeepNeuralNetwork:
    
    def __init__(self, layer_list):
        self.Parameters_Init(self, layer_list)
        pass
    
    def Parameters_Init(self, layer_list):
        self.layer_list = layer_list[:]
        np.random.seed(3)
        self.parameters = {}
        # number of layers in the network
        self.L = len(layer_list) - 1
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layer_list[l], \
                       layer_list[l-1]) * np.sqrt(2 / layer_list[l-1])
            self.parameters['b' + str(l)] = np.zeros((layer_list[l], 1))
            assert(self.parameters['W' + str(l)].shape == (layer_list[l], \
                                   layer_list[l-1]))
            assert(self.parameters['b' + str(l)].shape == (layer_list[l], 1))
        return self.parameters
    
    def Sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def ReLU(x):
        s = np.maximum(0, x)
        return s

    def Forward_Propagation(self, X):
        self.X = X[:]
        self.m = X.shape[1]
        self.caches = []
        assert(self.L == len(self.parameters) // 2)
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_now = self.ReLU(Z)
            cache = (Z, A_now, W, b)
            self.caches.append(cache)
        WL = self.parameters['W' + str(self.L)]
        bL = self.parameters['b' + str(self.L)]
        ZL = np.dot(WL, A_now)
        self.AL = self.Sigmoid(ZL)
        cache = (ZL, self.AL, WL, bL)
        self.caches.append(cache)
        assert(self.AL.shape == (1, X.shape[1]))
        return self.AL, self.caches
    
    def Compute_Cost(self, Y):
        assert(self.m == Y.shape[1])
        logprobs = np.multiply(-np.log(self.AL),Y) + \
        np.multiply(-np.log(1 - self.AL), 1 - Y)
        cost = 1.0/self.m * np.nansum(logprobs)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        self.cost = cost
        return self.cost
    
    def Backward_Propagation(self, Y):
        self.Y = Y[:]
        self.grads = {}
        assert(self.L == len(self.caches))
        assert(self.m == self.AL.shape[1])
        m = self.m
        L = self.L
        Y = Y.reshape(self.AL.shape)
        (ZL, AL, WL, bL) = self.caches[-1]
        (ZL_prev, AL_prev, WL_prev, bL_prev) = self.caches[-2]
        self.grads['dZ' + str(L)] = AL - Y
        self.grads['dW' + str(L)] = 1.0/m * \
        np.dot(self.grads['dZ' + str(L)], AL_prev.T)
        self.grads['db' + str(L)] = 1.0/m * \
        np.sum(self.grads['dZ' + str(L)], axis=1, keepdims = True)
        for l in reversed(range(L - 1)):
            current_cache = self.caches[l]
            (Z_current, A_current, W_current, b_current) = current_cache
            if l != 0:
                before_cache = self.caches[l - 1]
                (Z_before, A_before, W_before, b_before) = before_cache
            else:
                A_before = self.X
            behind_cache = self.caches[l + 1]
            (Z_behind, A_behind, W_behind, b_behind) = behind_cache
            dA = np.dot(W_behind.T, self.grads['dZ' + str(l + 2)])
            dZ = np.multiply(dA, np.int64(A_current > 0))
            dW = 1.0/m * np.dot(dZ, A_before.T)
            db = 1.0/m * np.sum(dZ, axis=1, keepdims = True)
            self.grads['dA' + str(l + 1)] = dA
            self.grads['dZ' + str(l + 1)] = dZ
            self.grads['dW' + str(l + 1)] = dW
            self.grads['db' + str(l + 1)] = db
        return self.grads
    
    def Update_Parameters(self, learning_rate):
        assert(self.L == len(self.parameters) // 2)
        L = self.L
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l + 1)] - \
            learning_rate * self.grads["dW" + str(l + 1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l + 1)] - \
            learning_rate * self.grads["db" + str(l + 1)]    
        return self.parameters
    
    def Train(self, X, Y, iterations = 3000, learning_rate = 0.0075, print_cost = False):
        np.random.seed(1)
        self.learning_rate = learning_rate
        self.costs = []
        self.Forward_Propagation(self, X)
        cost = self.Compute_Cost(self, Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(self, X, Y)
        self.costs.append(cost)
        for i in range(1, iterations+1):
            self.Forward_Propagation(self, X)
            cost = self.Compute_Cost(self, Y)
            self.Backward_Propagation(self, Y)
            self.Update_Parameters(self, learning_rate)
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.Query(self, X, Y)
            if i % 100 == 0:
                self.costs.append(cost)
        print('finished!')
        self.PlotCosts(self)
        return self.costs
    
    def Query(self, X, Y):
        m = X.shape[1]
        p = np.zeros((1,m))
        probs, caches = self.Forward_Propagation(self, X)
        for i in range(0, probs.shape[1]):
            if probs[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        print("Accuracy: "  + str(100*np.sum((p == Y)/m)) + '%')    
        return p
    
    def Predict(self, X):
        m = X.shape[1]
        p = np.zeros((1,m))
        probs, caches = self.Forward_Propagation(self, X)
        for i in range(0, probs.shape[1]):
            if probs[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p   
    
    def PlotCosts(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
            
train_X, train_Y, test_X, test_Y = load_2D_dataset()
plt.show()
n = DeepNeuralNetwork
n.__init__(n, [train_X.shape[0], 20, 3, 1])
n.Train(n, train_X, train_Y, learning_rate = 0.3, iterations = 30000, print_cost = True)

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: n.Predict(n, x.T), train_X, np.squeeze(train_Y))     
        
        
        
