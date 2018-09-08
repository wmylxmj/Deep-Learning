# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 18:34:47 2018

@author: wmy
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, \
convert_to_one_hot, predict

class DeepNeuralNetwork():
    
    def __init__(self, name, layer_list):
        self.name = name
        self.Parameters_Init(layer_list)
        self.Parameters_Init_TensorFlow(layer_list)
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
                       layer_list[l-1]) * np.sqrt(2 / layer_list[l-1])
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
    
    def Forward_Propagation(self, X):
        # copy the dataset X (or A0)
        self.dataset = {} 
        self.dataset['X'] = X[:]
        # the number of datasets
        self.m = X.shape[1]
        # the caches for hidden and output layers
        self.caches = []
        assert(self.L == len(self.parameters) // 2)
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_now = self.ReLU(Z)
            # cache : (Zl, Al, Wl, bl)
            cache = (Z, A_now, W, b)
            self.caches.append(cache)
        WL = self.parameters['W' + str(self.L)]
        bL = self.parameters['b' + str(self.L)]
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
        self.Forward_Propagation(X)
        cost = self.Compute_Cost(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        for i in range(1, iterations+1):
            self.Forward_Propagation(X)
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
        p = np.zeros((1,m), dtype = np.int)
        probs, caches = self.Forward_Propagation(X)
        for i in range(0, probs.shape[1]):
            if probs[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        #print("Accuracy: " + str(100*np.sum((p == Y)/m)) + '%')
        print("Accuracy: " + str(100 * np.mean((p[0,:] == Y[0,:]))) + '%')
        return p
    
    def Predict(self, X):
        m = X.shape[1]
        p = np.zeros((1,m), dtype = np.int)
        probs, caches = self.Forward_Propagation(X)
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
    
    def Forward_Propagation_Dropout(self, X):
        # choose the random seed
        np.random.seed(1)
        # copy the dataset X (or A0)
        self.dataset = {} 
        self.dataset['X'] = X[:]
        self.m = X.shape[1]
        # the caches for hidden and output layers
        self.caches = []
        self.D = {}
        assert(self.L == len(self.parameters) // 2)
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
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
        WL = self.parameters['W' + str(self.L)]
        bL = self.parameters['b' + str(self.L)]
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
        self.Forward_Propagation(X)
        cost = self.Compute_Cost(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        for i in range(1, iterations+1):
            self.Forward_Propagation_Dropout(X)
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
        self.Forward_Propagation(X)
        cost = self.Compute_Cost_L2_Regularization(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        for i in range(1, iterations + 1):
            self.Forward_Propagation(X)
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
    
    def J(self, X, Y, parameters):
        # the number of datasets
        m = X.shape[1]
        assert(self.L == len(parameters) // 2)
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_now = self.ReLU(Z)
            pass
        WL = parameters['W' + str(self.L)]
        bL = parameters['b' + str(self.L)]
        ZL = np.dot(WL, A_now) + bL
        # the output layer use sigmoid activation function
        AL = self.Sigmoid(ZL)
        # check the shape
        assert(AL.shape == (1, X.shape[1]))
        # Cost
        logprobs = np.multiply(-np.log(AL),Y) + \
        np.multiply(-np.log(1 - AL), 1 - Y)
        cost = 1.0/m * np.sum(logprobs)
        return cost
    
    def Dictionary_To_Vector(self, parameters):
        keys = []
        count = 0
        for l in range(1, self.L + 1):
            for key in ['W' + str(l), 'b' + str(l)]:
                new_vector = np.reshape(parameters[key], (-1,1))
                keys = keys + [key]*new_vector.shape[0]
                if count == 0:
                    theta = new_vector
                else:
                    theta = np.concatenate((theta, new_vector), axis=0)
                count = count + 1
        return theta, keys
    
    def Vector_To_Dictionary(self, theta):
        parameters = {}
        star = 0
        for l in range(1, self.L + 1):
            parameters['W' + str(l)] = \
            theta[star:star + self.parameters['W' + str(l)].shape[0] * \
                                              self.parameters['W' + str(l)].shape[1]].reshape(self.parameters['W' + str(l)].shape)
            star = star + self.parameters['W' + str(l)].shape[0] * \
            self.parameters['W' + str(l)].shape[1]
            parameters['b' + str(l)] = \
            theta[star:star + self.parameters['b' + str(l)].shape[0] * \
                                              self.parameters['b' + str(l)].shape[1]].reshape(self.parameters['b' + str(l)].shape)
            star = star + self.parameters['b' + str(l)].shape[0] * \
            self.parameters['b' + str(l)].shape[1]
        return parameters
   
    def Gradients_To_Vector(self, gradients):
        count = 0
        for l in range(1, self.L + 1):
            for key in ['dW' + str(l), 'db' + str(l)]:
                new_vector = np.reshape(gradients[key], (-1,1))
                if count == 0:
                    theta = new_vector
                else:
                    theta = np.concatenate((theta, new_vector), axis=0)
                count = count + 1
                pass
            pass
        return theta
    
    def Gradient_Check(self, parameters, gradients, X, Y, epsilon = 1e-7):
        # Set-up variables
        parameters_values, _ = self.Dictionary_To_Vector(parameters)
        grad = self.Gradients_To_Vector(gradients)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        # Compute gradapprox
        for i in range(num_parameters):
            # J plus
            thetaplus = np.copy(parameters_values)              
            thetaplus[i][0] += epsilon                           
            J_plus[i]= self.J(X, Y, self.Vector_To_Dictionary(thetaplus)) 
            # J minus
            thetaminus = np.copy(parameters_values)                                  
            thetaminus[i][0] -= epsilon                            
            J_minus[i] = self.J(X, Y, self.Vector_To_Dictionary(thetaminus))
            # Compute gradapprox[i]
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
            pass
        numerator = np.linalg.norm(grad - gradapprox)                      
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)               
        difference = numerator / denominator   
        if difference > 1e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
        return difference  

    def Check_The_Gradient(self, X, Y, epsilon = 1e-7):
        print('Gradient checking...please wait.')
        self.Forward_Propagation(X)
        self.Backward_Propagation(Y)
        parameters = self.parameters
        gradients = self.grads
        difference = self.Gradient_Check(parameters, gradients, X, Y, epsilon = epsilon)
        return difference
    
    def Random_Mini_Batches(self, X, Y, mini_batch_size = 64, seed = 0):
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches
    
    def Initialize_Velocity(self):
        self.v = {}
        for l in range(1, self.L + 1):
            self.v["dW" + str(l)] = np.zeros((self.parameters['W' + str(l)].shape[0], \
                  self.parameters['W' + str(l)].shape[1]))
            self.v["db" + str(l)] = np.zeros((self.parameters['b' + str(l)].shape[0], \
                  self.parameters['b' + str(l)].shape[1]))
        return self.v
    
    def Update_Parameters_Momentum(self, beta, learning_rate):
        for l in range(1, self.L + 1):
            # compute velocities
            self.v["dW" + str(l)] = beta * self.v['dW' + str(l)] + \
            (1 - beta) * self.grads['dW' + str(l)]
            self.v["db" + str(l)] = beta * self.v['db' + str(l)] + \
            (1 - beta) * self.grads['db' + str(l)]
            # update parameters
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - \
            learning_rate * self.v["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - \
            learning_rate * self.v["db" + str(l)]
        return self.parameters, self.v
    
    def Initialize_Adam(self): 
        self.v = {}
        self.s = {}
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(1, self.L + 1):
            self.v["dW" + str(l)] = np.zeros((self.parameters["W" + str(l)].shape[0], \
                  self.parameters["W" + str(l)].shape[1]))
            self.v["db" + str(l)] = np.zeros((self.parameters["b" + str(l)].shape[0], \
                  self.parameters["b" + str(l)].shape[1]))
            self.s["dW" + str(l)] = np.zeros((self.parameters["W" + str(l)].shape[0], \
                  self.parameters["W" + str(l)].shape[1]))
            self.s["db" + str(l)] = np.zeros((self.parameters["b" + str(l)].shape[0], \
                  self.parameters["b" + str(l)].shape[1]))
        return self.v, self.s
    
    def Update_Parameters_Adam(self, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):   
        self.v_corrected = {}                         
        self.s_corrected = {}                       
        # Perform Adam update on all parameters
        for l in range(1, self.L + 1):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            self.v["dW" + str(l)] = beta1 * self.v["dW" + str(l)] + (1 - beta1) * self.grads['dW' + str(l)]
            self.v["db" + str(l)] = beta1 * self.v["db" + str(l)] + (1 - beta1) * self.grads['db' + str(l)]
            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            self.v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - beta1 ** t)
            self.v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - beta1 ** t)
            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            self.s["dW" + str(l)] = beta2 * self.s["dW" + str(l)] + (1 - beta2) * (self.grads['dW' + str(l)] ** 2)
            self.s["db" + str(l)] = beta2 * self.s["db" + str(l)] + (1 - beta2) * (self.grads['db' + str(l)] ** 2)
            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            self.s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - beta2 ** t)
            self.s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - beta2 ** t)
            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * self.v_corrected["dW" + str(l)] / (np.sqrt(self.s_corrected["dW" + str(l)]) + epsilon)
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * self.v_corrected["db" + str(l)] / (np.sqrt(self.s_corrected["db" + str(l)]) + epsilon)
        return self.parameters, self.v, self.s
    
    def Train_Gradient_Descent(self, X, Y, learning_rate = 0.0007, mini_batch_size = 64, epsilon = 1e-8, num_epochs = 10000, print_cost = True):
        self.costs = []
        self.learning_rate = learning_rate
        self.Forward_Propagation(X)
        cost = self.Compute_Cost(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        seed = 10 
        for i in range(1, num_epochs + 1):
            seed = seed + 1
            minibatches = self.Random_Mini_Batches(X, Y, mini_batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                self.Forward_Propagation(minibatch_X)
                cost = self.Compute_Cost(minibatch_Y)
                self.Backward_Propagation(minibatch_Y)
                self.Update_Parameters(learning_rate)
            if print_cost and i % (10*self.iteration_unit) == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.Query(X, Y)
            if i % self.iteration_unit == 0:
                self.costs.append(cost)
        print('finished!')
        self.PlotCosts()
        return self.costs
    
    def Train_Momentum(self, X, Y, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9, epsilon = 1e-8, num_epochs = 10000, print_cost = True):
        self.costs = []
        self.learning_rate = learning_rate
        self.Forward_Propagation(X)
        cost = self.Compute_Cost(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        # Initialize the optimizer
        self.Initialize_Velocity()
        seed = 10 
        for i in range(1, num_epochs + 1):
            seed = seed + 1
            minibatches = self.Random_Mini_Batches(X, Y, mini_batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                self.Forward_Propagation(minibatch_X)
                cost = self.Compute_Cost(minibatch_Y)
                self.Backward_Propagation(minibatch_Y)
                self.Update_Parameters_Momentum(beta=beta, learning_rate=learning_rate)
            if print_cost and i % (10*self.iteration_unit) == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.Query(X, Y)
            if i % self.iteration_unit == 0:
                self.costs.append(cost)
        print('finished!')
        self.PlotCosts()
        return self.costs
    
    def Train_Adam(self, X, Y, learning_rate = 0.0007, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
        self.costs = []
        self.learning_rate = learning_rate
        self.Forward_Propagation(X)
        cost = self.Compute_Cost(Y)
        print ("Cost after iteration %i: %f" %(0, cost))
        self.Query(X, Y)
        self.costs.append(cost)
        # Initialize the optimizer
        self.Initialize_Adam()
        t = 0 
        seed = 10 
        # Optimization loop
        for i in range(1, num_epochs + 1):
            seed = seed + 1
            minibatches = self.Random_Mini_Batches(X, Y, mini_batch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                self.Forward_Propagation(minibatch_X)
                cost = self.Compute_Cost(minibatch_Y)
                self.Backward_Propagation(minibatch_Y)
                # Adam counter
                t = t + 1
                self.Update_Parameters_Adam(t, learning_rate, beta1, beta2, epsilon)
            if print_cost and i % (10*self.iteration_unit) == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.Query(X, Y)
            if i % self.iteration_unit == 0:
                self.costs.append(cost)
        print('finished!')
        self.PlotCosts()
        return self.costs
    
    def Create_Placeholders_TensorFlow(self, n_x, n_y):
        X = tf.placeholder(tf.float32, shape=[n_x, None])
        Y = tf.placeholder(tf.float32, shape=[n_y, None])
        return X, Y
    
    def Parameters_Init_TensorFlow(self, layer_list):
        tf.reset_default_graph()
        tf.set_random_seed(1)
        self.layer_list = layer_list[:]
        self.tf_parameters = {}
        self.L = len(layer_list) - 1
        for l in range(1, self.L + 1):
            self.tf_parameters['W' + str(l)] = \
            tf.get_variable("W" + str(l), [layer_list[l], layer_list[l - 1]], \
                            initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            self.tf_parameters['b' + str(l)] = \
            tf.get_variable("b" + str(l), [layer_list[l], 1], \
                            initializer = tf.zeros_initializer())
            pass
        return self.tf_parameters
    
    def Forward_Propagation_TensorFlow_Softmax(self, X):
        # the number of datasets
        self.m = X.shape[1]
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = self.tf_parameters['W' + str(l)]
            b = self.tf_parameters['b' + str(l)]
            Z = tf.add(tf.matmul(W, A_prev), b) 
            A_now = tf.nn.relu(Z) 
            pass
        WL = self.tf_parameters['W' + str(self.L)]
        bL = self.tf_parameters['b' + str(self.L)]
        ZL = tf.add(tf.matmul(WL, A_now), bL)   
        self.tf_ZL = ZL
        return ZL
    
    def Compute_Cost_TensorFlow_Softmax(self, Y):
        logits = tf.transpose(self.tf_ZL)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        self.tf_cost = cost
        return cost
    
    def Train_TensorFlow_Softmax(self, X_train, Y_train, X_test, Y_test, iterations = 1500, learning_rate = 0.0001, minibatch_size = 32, print_cost = True):
        self.learning_rate = learning_rate
        # to keep consistent results
        tf.set_random_seed(1)
        # to keep consistent results
        seed = 3 
        (n_x, m) = X_train.shape  
        n_y = Y_train.shape[0] 
        self.costs = []
        X, Y = self.Create_Placeholders_TensorFlow(n_x, n_y)
        self.Forward_Propagation_TensorFlow_Softmax(X)
        cost = self.Compute_Cost_TensorFlow_Softmax(Y)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        # Initialize all the variables
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print("Now training...")
            for i in range(1, iterations + 1):
                iteration_cost = 0
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1
                minibatches = self.Random_Mini_Batches(X_train, Y_train, minibatch_size, seed)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    iteration_cost += minibatch_cost / num_minibatches
                    pass
                if print_cost and i % (10*self.iteration_unit) == 0:
                    print ("Cost after iteration %i: %f" %(i, iteration_cost))
                    correct_prediction = tf.equal(tf.argmax(self.tf_ZL), tf.argmax(Y))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print ("Train Accuracy:" , 100*accuracy.eval({X: X_train, Y: Y_train}), "%")
                    print ("Test Accuracy:" , 100*accuracy.eval({X: X_test, Y: Y_test}), "%")
                    pass
                if i % self.iteration_unit == 0:
                    self.costs.append(iteration_cost)
                    pass
                pass
            print('finished')
            self.PlotCosts()
            self.parameters = sess.run(self.tf_parameters)
            correct_prediction = tf.equal(tf.argmax(self.tf_ZL), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Train Accuracy:" , 100*accuracy.eval({X: X_train, Y: Y_train}), "%")
            print ("Test Accuracy:" , 100*accuracy.eval({X: X_test, Y: Y_test}), "%")
            sess.close()
        return self.parameters   
    
    pass


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
plt.show()
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

n = DeepNeuralNetwork("TensotFlow Softmax", [12288, 25, 12, 6])

n.iteration_unit = 10
X, Y = n.Create_Placeholders_TensorFlow(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))


with tf.Session() as sess:
    print("W1 = " + str(n.tf_parameters["W1"]))
    print("b1 = " + str(n.tf_parameters["b1"]))
    print("W2 = " + str(n.tf_parameters["W2"]))
    print("b2 = " + str(n.tf_parameters["b2"]))
   
with tf.Session() as sess:
    X, Y = n.Create_Placeholders_TensorFlow(12288, 6)
    Z3 = n.Forward_Propagation_TensorFlow_Softmax(X)
    print("Z3 = " + str(Z3))
    
with tf.Session() as sess:
    X, Y = n.Create_Placeholders_TensorFlow(12288, 6)
    Z3 = n.Forward_Propagation_TensorFlow_Softmax(X)
    cost = n.Compute_Cost_TensorFlow_Softmax(Y)
    print("cost = " + str(cost))
    
n.Train_TensorFlow_Softmax(X_train, Y_train, X_test, Y_test)

    