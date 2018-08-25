# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:18:18 2018

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
        np.random.seed(1)
        self.parameters = {}
        # number of layers in the network
        self.L = len(layer_list) - 1
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layer_list[l], \
                       layer_list[l-1]) / np.sqrt(layer_list[l-1])
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
    
    def PlotCosts(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 100)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
        
    
train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()
index = 7
plt.imshow(train_x_orig[index])
plt.show()
print ("y = " + str(train_y_orig[0,index]) + ". It's a " + \
       classes[train_y_orig[0,index]].decode("utf-8") +  " picture.")

#标签不需要处理
train_y = train_y_orig
test_y = test_y_orig

print ('\n' + "-------------------------------------------------------" + '\n')
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
print('有'+str(m_train)+'张训练图片')
print('有'+str(m_test)+'张测试图片')
height_px = train_x_orig[0].shape[0]
width_px = train_x_orig[0].shape[1]
print('图片宽'+str(width_px)+'像素')
print('图片高'+str(height_px)+'像素')
print ('\n' + "-------------------------------------------------------" + '\n')

#变为一维向量
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

print('一维化后的图片：')
image_train_data_set = train_x_orig.reshape(train_x_orig.shape[0], \
                                                width_px*height_px, 3)
plt.imshow(image_train_data_set[0])
plt.show()
print ('\n' + "-------------------------------------------------------" + '\n')

#RGB通道取值变为0到1
#标准化数据集
train_x = train_x_flatten/255.0
test_x = test_x_flatten/255.0

n = DeepNeuralNetwork
n.__init__(n, [12288, 50, 20, 7, 3, 1])
n.Train(n, train_x, train_y, learning_rate = 0.0075, print_cost = True, iterations = 2500)

n.Query(n, train_x, train_y)
n.Query(n, test_x, test_y)

n.PlotCosts(n)

index = 5
plt.imshow(test_x_orig[index])
plt.show()
print ("y = " + str(test_y_orig[0,index]) + ". It's a " + \
       classes[test_y_orig[0,index]].decode("utf-8") +  " picture.")

test_pic_y_orig = np.squeeze(test_y_orig)   
test_picture_x = np.array([test_x_orig[0],
                           test_x_orig[4],
                           test_x_orig[8],
                           test_x_orig[12],
                           test_x_orig[16],
                           test_x_orig[20],
                           test_x_orig[24],
                           test_x_orig[28],
                           test_x_orig[32],
                           test_x_orig[36],
                           test_x_orig[40]])
test_picture_y = np.array([test_pic_y_orig[0],
                           test_pic_y_orig[4],
                           test_pic_y_orig[8],
                           test_pic_y_orig[12],
                           test_pic_y_orig[16],
                           test_pic_y_orig[20],
                           test_pic_y_orig[24],
                           test_pic_y_orig[28],
                           test_pic_y_orig[32],
                           test_pic_y_orig[36],
                           test_pic_y_orig[40]])
#变为一维向量
image_x_flatten = test_picture_x.reshape(test_picture_x.shape[0], -1).T


result = n.Query(n, image_x_flatten, test_picture_y)
for i in range(0, len(test_picture_x)):
    index = i
    plt.imshow(test_picture_x[index])
    plt.show()
    if int(np.squeeze(result)[i]) == 1:
        print('you predict it a "cat" picture')
    else:
        print('you predict it a "non-cat" picture')
    
    
