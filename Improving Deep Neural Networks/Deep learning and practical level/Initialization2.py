# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 21:46:53 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
from init_utils import load_dataset, plot_decision_boundary

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
        cost = (1.0/self.m) * (-np.dot(Y,np.log(self.AL).T) - \
                np.dot(1-Y, np.log(1-self.AL).T))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        self.cost = cost
        return self.cost
    
    def Backward_Z(dZ, A_prev_W_b_cache):
        A_prev, W, b = A_prev_W_b_cache
        m = A_prev.shape[1]
        dW = float(1.0/m) * np.dot(dZ, A_prev.T)
        db = float(1.0/m) * np.sum(dZ, axis=1, keepdims=True)
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
                                           current_cache, 'relu')
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
            if print_cost and i % 100 == 0:
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
        
# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
plt.show()

n = DeepNeuralNetwork
n.Parameters_Init(n, [2, 10, 5, 1])
print("W1 = " + str(n.parameters["W1"]))
print("b1 = " + str(n.parameters["b1"]))
print("W2 = " + str(n.parameters["W2"]))
print("b2 = " + str(n.parameters["b2"]))
n.Train(n, train_X, train_Y, learning_rate = 0.01, iterations = 15000, print_cost = True)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: n.Predict(n, x.T), train_X, np.squeeze(train_Y))

plt.show()

n.Query(n, test_X, test_Y)
        