# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:23:53 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, \
load_planar_dataset, load_extra_datasets

class OneHiddenLayerNeuralNetwork:
    
    def __init__(self, X, Y, n_h, learning_rate = 1.2, num_iterations = 10000, print_cost=False):
        self.X = X
        self.Y = Y
        self.n_h = n_h
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        
    def sigmoid(self, x):
        s = 1.0/(1+np.exp(-x))
        return s

    def layer_sizes(self):
        self.n_x = np.shape(self.X)[0]
        self.n_h = self.n_h
        self.n_y = np.shape(self.Y)[0]
        return (self.n_x, self.n_h, self.n_y)
        
    def initialize_parameters(self): 
        np.random.seed(1) 
        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        self.b2 = np.zeros((self.n_y, 1))
        assert (self.W1.shape == (self.n_h, self.n_x))
        assert (self.b1.shape == (self.n_h, 1))
        assert (self.W2.shape == (self.n_y, self.n_h))
        assert (self.b2.shape == (self.n_y, 1))
        self.parameters = {"W1": self.W1,
                           "b1": self.b1,
                           "W2": self.W2,
                           "b2": self.b2}
        return self.parameters
    
    def forward_propagation(self):
        self.W1 = self.parameters['W1']
        self.b1 = self.parameters['b1']
        self.W2 = self.parameters['W2']
        self.b2 = self.parameters['b2']
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        assert(self.A2.shape == (1, self.X.shape[1]))
        self.cache = {"Z1": self.Z1,
                      "A1": self.A1,
                      "Z2": self.Z2,
                      "A2": self.A2}
        return self.A2, self.cache

    def compute_cost(self):
        self.m = self.Y.shape[1] 
        self.logprobs = np.multiply(np.log(self.A2), self.Y) + \
        np.multiply(np.log(1-self.A2), 1-self.Y)
        self.cost = -1/self.m * np.sum(self.logprobs)
        self.cost = np.squeeze(self.cost)                            
        assert(isinstance(self.cost, float))
        return self.cost
    
    def backward_propagation(self):
        self.m = self.X.shape[1]
        self.W1 = self.parameters['W1']
        self.W2 = self.parameters['W2']
        self.A1 = self.cache['A1']
        self.A2 = self.cache['A2']
        self.dZ2 = self.A2 - self.Y
        self.dW2 = 1/self.m * np.dot(self.dZ2, self.A1.T)
        self.db2 = 1/self.m * np.sum(self.dZ2, axis = 1, keepdims = True)
        self.dZ1 = np.dot(self.W2.T, self.dZ2) * (1 - np.power(self.A1, 2))
        self.dW1 = 1/self.m * np.dot(self.dZ1, self.X.T)
        self.db1 = 1/self.m * np.sum(self.dZ1, axis = 1, keepdims = True)
        self.grads = {"dW1": self.dW1,
                      "db1": self.db1,
                      "dW2": self.dW2,
                      "db2": self.db2}
        return self.grads
    
    def update_parameters(self):
        self.W1 = self.parameters['W1']
        self.b1 = self.parameters['b1']
        self.W2 = self.parameters['W2']
        self.b2 = self.parameters['b2']
        self.dW1 = self.grads['dW1']
        self.db1 = self.grads['db1']
        self.dW2 = self.grads['dW2']
        self.db2 = self.grads['db2']
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
        self.parameters = {"W1": self.W1,
                           "b1": self.b1,
                           "W2": self.W2,
                           "b2": self.b2}
        return self.parameters
    
    def nn_model(self):
        np.random.seed(2)
        self.layer_sizes()
        self.initialize_parameters()
        self.W1 = self.parameters['W1']
        self.b1 = self.parameters['b1']
        self.W2 = self.parameters['W2']
        self.b2 = self.parameters['b2']
        for i in range(0, self.num_iterations):
            self.A2, self.cache = self.forward_propagation()    
            self.cost = self.compute_cost()     
            self.grads = self.backward_propagation()    
            self.parameters = self.update_parameters()            
            if self.print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, self.cost))
                self.print_accuracy()
        return self.parameters
    
    def predict(self, X):
        self.W1 = self.parameters['W1']
        self.b1 = self.parameters['b1']
        self.W2 = self.parameters['W2']
        self.b2 = self.parameters['b2']
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        assert(A2.shape == (1, X.shape[1]))
        predictions = (A2 > 0.5)
        return predictions
    
    def print_accuracy(self):
        predictions = self.predict(self.X)
        print ('Accuracy: %d' % \
               float((np.dot(Y,predictions.T) + \
                      np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + \
                      '%')
             
    def print_details(self):
        print('---------------------------------------------\n')
        print('details :')
        print('Neural Network Size : ' + '(' + str(self.n_x) + \
                                          ', ' + str(self.n_h) + \
                                          ', ' + str(self.n_y) + \
                                          ')')
        print('Cost : ' + str(self.cost))
        self.print_accuracy()
        print('W1 : \n' + str(self.W1))
        print('W2 : \n' + str(self.W2))
        print('b1 : \n' + str(self.b1))
        print('b2 : \n' + str(self.b2))
        print('\n---------------------------------------------')
        
X, Y = load_planar_dataset()
a = OneHiddenLayerNeuralNetwork(X, Y, 75, \
                                learning_rate = 1.25, \
                                num_iterations = 20000, print_cost=True)
parameters = a.nn_model()
plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
plt.show()

# Plot the decision boundary
plot_decision_boundary(lambda x: a.predict(x.T), X, Y[0])
plt.title("Decision Boundary for hidden layer size " + str(a.n_h))
plt.show()

# Print accuracy
a.print_accuracy()
a.print_details()
