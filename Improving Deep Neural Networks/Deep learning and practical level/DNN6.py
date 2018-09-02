# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 23:22:03 2018

@author: wmy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:02:05 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, \
initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, \
backward_propagation, update_parameters

class DeepNeuralNetwork():
    
    def __init__(self, name, layer_list):
        self.name = name
        self.Parameters_Init(layer_list)
        # it will be used when plot the costs picture
        self.iteration_unit = 1000
        print("You created a deep neural network named '" + self.name + "'")
        print('The layer list is ' + str(self.layer_list))
        pass
    
    def Parameters_Init(self, layer_list):
        self.layer_list = layer_list[:]
        np.random.seed(3)
        self.parameters = {}
        # number of layers in the network
        self.L = len(layer_list) - 1
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layer_list[l], \
                       layer_list[l-1]) * np.sqrt(1 / layer_list[l-1])
            self.parameters['b' + str(l)] = np.zeros((layer_list[l], 1))
            assert(self.parameters['W' + str(l)].shape == (layer_list[l], \
                                   layer_list[l-1]))
            assert(self.parameters['b' + str(l)].shape == (layer_list[l], 1))
        return self.parameters
    
    def Sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def ReLU(self, x):
        s = np.maximum(0, x)
        return s
    
    def Forward_Propagation(self, X, parameters):
        # copy the dataset X (or A0)
        self.dataset = {} 
        self.dataset['X'] = X[:]
        # the number of datasets
        self.m = X.shape[1]
        # the caches for hidden and output layers
        self.caches = []
        assert(self.L == len(parameters) // 2)
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_now = self.ReLU(Z)
            # cache : (Zl, Al, Wl, bl)
            cache = (Z, A_now, W, b)
            self.caches.append(cache)
        WL = parameters['W' + str(self.L)]
        bL = parameters['b' + str(self.L)]
        ZL = np.dot(WL, A_now) + bL
        # the output layer use sigmoid activation function
        self.AL = self.Sigmoid(ZL)
        cache = (ZL, self.AL, WL, bL)
        self.caches.append(cache)
        # check the shape
        assert(self.AL.shape == (1, X.shape[1]))
        return self.AL, self.caches
    
    def Compute_Cost(self, Y):
        assert(self.m == Y.shape[1])
        # - (y * log(a) + (1-y) * log(1-a))
        logprobs = np.multiply(-np.log(self.AL),Y) + \
        np.multiply(-np.log(1 - self.AL), 1 - Y)
        # the average of the loss function
        cost = 1.0/self.m * np.nansum(logprobs)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        self.cost = cost
        return self.cost
    
    def Backward_Propagation(self, Y):
        # copy the dataset Y
        self.dataset['Y'] = Y[:]
        self.grads = {}
        assert(self.L == len(self.caches))
        assert(self.m == self.AL.shape[1])
        m = self.m
        # the number of layers
        L = self.L
        Y = Y.reshape(self.AL.shape)
        (ZL, AL, WL, bL) = self.caches[-1]
        (ZL_prev, AL_prev, WL_prev, bL_prev) = self.caches[-2]
        # compute the grads of layer L
        self.grads['dZ' + str(L)] = AL - Y
        self.grads['dW' + str(L)] = 1.0/m * \
        np.dot(self.grads['dZ' + str(L)], AL_prev.T)
        self.grads['db' + str(L)] = 1.0/m * \
        np.sum(self.grads['dZ' + str(L)], axis=1, keepdims = True)
        for l in reversed(range(L - 1)):
            # the layer l + 1
            current_cache = self.caches[l]
            (Z_current, A_current, W_current, b_current) = current_cache
            if l != 0:
                before_cache = self.caches[l - 1]
                (Z_before, A_before, W_before, b_before) = before_cache
            else:
                # A0
                A_before = self.dataset['X']
            behind_cache = self.caches[l + 1]
            (Z_behind, A_behind, W_behind, b_behind) = behind_cache
            # compute the grads of layer l + 1
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
            # W = W - a * dW
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - \
            learning_rate * self.grads["dW" + str(l + 1)]
            # b = b - a * db
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - \
            learning_rate * self.grads["db" + str(l + 1)]    
        return self.parameters
    
    def Train(self, X, Y, iterations = 3000, learning_rate = 0.0075, print_cost = False):
        self.learning_rate = learning_rate
        self.costs = []
        self.Forward_Propagation(X, self.parameters)
        cost = self.Compute_Cost(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        for i in range(1, iterations+1):
            self.Forward_Propagation(X, self.parameters)
            cost = self.Compute_Cost(Y)
            self.Backward_Propagation(Y)
            self.Update_Parameters(learning_rate)
            if print_cost and i % (10*self.iteration_unit) == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.Query(X, Y)
            if i % self.iteration_unit == 0:
                self.costs.append(cost)
        print('finished!')
        self.PlotCosts()
        return self.costs
    
    def Query(self, X, Y):
        m = X.shape[1]
        p = np.zeros((1,m))
        probs, caches = self.Forward_Propagation(X, self.parameters)
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
        probs, caches = self.Forward_Propagation(X, self.parameters)
        for i in range(0, probs.shape[1]):
            if probs[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p   
    
    def PlotCosts(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per ' + str(self.iteration_unit) + ')')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
    
    def Dropout_Init(self, keep_prob_list):
        self.keep_prob_list = keep_prob_list
        pass
    
    def Forward_Propagation_Dropout(self, X, parameters):
        # choose the random seed
        np.random.seed(1)
        # copy the dataset X (or A0)
        self.dataset = {} 
        self.dataset['X'] = X[:]
        self.m = X.shape[1]
        # the caches for hidden and output layers
        self.caches = []
        self.D = {}
        assert(self.L == len(parameters) // 2)
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_now = self.ReLU(Z)
            # dropout
            self.D['D' + str(l)] = np.random.rand(A_now.shape[0], \
                   A_now.shape[1]) 
            self.D['D' + str(l)] = (self.D['D' + str(l)] < \
                  self.keep_prob_list[l - 1])
            A_now = A_now * self.D['D' + str(l)]
            A_now = A_now / self.keep_prob_list[l - 1]
            # cache : (Zl, Al, Wl, bl)
            cache = (Z, A_now, W, b)
            self.caches.append(cache)
        WL = parameters['W' + str(self.L)]
        bL = parameters['b' + str(self.L)]
        ZL = np.dot(WL, A_now) + bL
        # the output layer use sigmoid activation function
        self.AL = self.Sigmoid(ZL)
        cache = (ZL, self.AL, WL, bL)
        self.caches.append(cache)
        # check the shape
        assert(self.AL.shape == (1, X.shape[1]))
        return self.AL, self.caches
    
    def Backward_Propagation_Dropout(self, Y):
        # copy the dataset Y
        self.dataset['Y'] = Y[:]
        self.grads = {}
        assert(self.L == len(self.caches))
        assert(self.m == self.AL.shape[1])
        m = self.m
        # the number of layers
        L = self.L
        Y = Y.reshape(self.AL.shape)
        (ZL, AL, WL, bL) = self.caches[-1]
        (ZL_prev, AL_prev, WL_prev, bL_prev) = self.caches[-2]
        # compute the grads of layer L
        self.grads['dZ' + str(L)] = AL - Y
        self.grads['dW' + str(L)] = 1.0/m * \
        np.dot(self.grads['dZ' + str(L)], AL_prev.T)
        self.grads['db' + str(L)] = 1.0/m * \
        np.sum(self.grads['dZ' + str(L)], axis=1, keepdims = True)
        for l in reversed(range(L - 1)):
            # the layer l + 1
            current_cache = self.caches[l]
            (Z_current, A_current, W_current, b_current) = current_cache
            if l != 0:
                before_cache = self.caches[l - 1]
                (Z_before, A_before, W_before, b_before) = before_cache
            else:
                # A0
                A_before = self.dataset['X']
            behind_cache = self.caches[l + 1]
            (Z_behind, A_behind, W_behind, b_behind) = behind_cache
            # compute the grads of layer l + 1
            dA = np.dot(W_behind.T, self.grads['dZ' + str(l + 2)])
            # dropout
            dA = dA * self.D['D' + str(l + 1)]
            dA = dA / self.keep_prob_list[l]
            # dropout finished
            dZ = np.multiply(dA, np.int64(A_current > 0))
            dW = 1.0/m * np.dot(dZ, A_before.T)
            db = 1.0/m * np.sum(dZ, axis=1, keepdims = True)
            self.grads['dA' + str(l + 1)] = dA
            self.grads['dZ' + str(l + 1)] = dZ
            self.grads['dW' + str(l + 1)] = dW
            self.grads['db' + str(l + 1)] = db
        return self.grads
    
    def Train_Dropout(self, X, Y, keep_prob_list, iterations = 3000, learning_rate = 0.0075, print_cost = False):
        self.Dropout_Init(keep_prob_list)
        self.learning_rate = learning_rate
        self.costs = []
        self.Forward_Propagation(X, self.parameters)
        cost = self.Compute_Cost(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        for i in range(1, iterations+1):
            self.Forward_Propagation_Dropout(X, self.parameters)
            cost = self.Compute_Cost(Y)
            self.Backward_Propagation_Dropout(Y)
            self.Update_Parameters(learning_rate)
            if print_cost and i % (10*self.iteration_unit) == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.Query(X, Y)
            if i % self.iteration_unit == 0:
                self.costs.append(cost)
        print('finished!')
        self.PlotCosts()
        return self.costs
    
    def L2_Regularization_Init(self, lambda_list):
        self.lambda_list = lambda_list[:]
        pass
    
    def Compute_Cost_L2_Regularization(self, Y):
        m = Y.shape[1]
        cross_entropy_cost = self.Compute_Cost(Y)
        L2_regularization_cost = 0.0
        for l in range(1, self.L + 1):
            Wl = self.parameters['W' + str(l)]
            L2_regularization_cost +=  1.0/m * \
            self.lambda_list[l - 1]/2 * np.sum(np.square(Wl))
        cost = cross_entropy_cost + L2_regularization_cost
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        self.cost = cost
        return self.cost
    
    def Backward_Propagation_L2_Regularization(self, Y):
        # copy the dataset Y
        self.dataset['Y'] = Y[:]
        self.grads = {}
        assert(self.L == len(self.caches))
        assert(self.m == self.AL.shape[1])
        m = self.m
        # the number of layers
        L = self.L
        Y = Y.reshape(self.AL.shape)
        (ZL, AL, WL, bL) = self.caches[-1]
        (ZL_prev, AL_prev, WL_prev, bL_prev) = self.caches[-2]
        # compute the grads of layer L
        self.grads['dZ' + str(L)] = AL - Y
        # L2 regularization
        self.grads['dW' + str(L)] = 1.0/m * \
        np.dot(self.grads['dZ' + str(L)], AL_prev.T)
        self.grads['dW' + str(L)] += self.lambda_list[-1] / m * WL
        # L2 regularization finished
        self.grads['db' + str(L)] = 1.0/m * \
        np.sum(self.grads['dZ' + str(L)], axis=1, keepdims = True)
        for l in reversed(range(L - 1)):
            # the layer l + 1
            current_cache = self.caches[l]
            (Z_current, A_current, W_current, b_current) = current_cache
            if l != 0:
                before_cache = self.caches[l - 1]
                (Z_before, A_before, W_before, b_before) = before_cache
            else:
                # A0
                A_before = self.dataset['X']
            behind_cache = self.caches[l + 1]
            (Z_behind, A_behind, W_behind, b_behind) = behind_cache
            # compute the grads of layer l + 1
            dA = np.dot(W_behind.T, self.grads['dZ' + str(l + 2)])
            dZ = np.multiply(dA, np.int64(A_current > 0))
            # L2 regularization
            dW = 1.0/m * np.dot(dZ, A_before.T)
            dW += self.lambda_list[l] / m * W_current
            # L2 regularization finished
            db = 1.0/m * np.sum(dZ, axis=1, keepdims = True)
            self.grads['dA' + str(l + 1)] = dA
            self.grads['dZ' + str(l + 1)] = dZ
            self.grads['dW' + str(l + 1)] = dW
            self.grads['db' + str(l + 1)] = db
        return self.grads
    
    def Train_L2_Regularization(self, X, Y, lambda_list, iterations = 3000, learning_rate = 0.0075, print_cost = False):
        self.L2_Regularization_Init(lambda_list)
        self.learning_rate = learning_rate
        self.costs = []
        self.Forward_Propagation(X, self.parameters)
        cost = self.Compute_Cost_L2_Regularization(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        for i in range(1, iterations + 1):
            self.Forward_Propagation(X, self.parameters)
            cost = self.Compute_Cost_L2_Regularization(Y)
            self.Backward_Propagation_L2_Regularization(Y)
            self.Update_Parameters(learning_rate)
            if print_cost and i % (10*self.iteration_unit) == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.Query(X, Y)
            if i % self.iteration_unit == 0:
                self.costs.append(cost)
        print('finished!')
        self.PlotCosts()
        return self.costs
    
    pass

train_X, train_Y, test_X, test_Y = load_2D_dataset()
plt.show()

n = DeepNeuralNetwork('Model without regularization', [train_X.shape[0], 20, 3, 1])

n.Train(train_X, train_Y, learning_rate = 0.3, \
        iterations = 30000, print_cost = True)

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: n.Predict(x.T), \
                       train_X, np.squeeze(train_Y))     
n.Query(test_X, test_Y)

b = DeepNeuralNetwork('Model with L2-regularization', [train_X.shape[0], 20, 3, 1])

b.Train_L2_Regularization(train_X, train_Y, [0.7, 0.7, 0.7], \
                          learning_rate = 0.3, iterations = 30000, print_cost = True)

plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: b.Predict(x.T), \
                       train_X, np.squeeze(train_Y))     
b.Query(test_X, test_Y)

a = DeepNeuralNetwork('Model with dropout', [train_X.shape[0], 20, 3, 1])

a.Train_Dropout(train_X, train_Y, [0.86, 0.86], \
                learning_rate = 0.3, iterations = 30000, print_cost = True)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: a.Predict(x.T), \
                       train_X, np.squeeze(train_Y))     
a.Query(test_X, test_Y)

print(n == a)
print(b == a)
