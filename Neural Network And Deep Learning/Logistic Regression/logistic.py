# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 13:24:23 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import time 

def LoadDataSet():
    DataMatrix = []
    LabelList = []
    fr = open('testset.txt')
    for line in fr.readlines():
        LineArray = line.strip().split()
        DataMatrix.append([float(LineArray[0]), float(LineArray[1])])
        LabelList.append(int(LineArray[2]))
    DataMatrix = np.mat(DataMatrix)
    DataMatrix = DataMatrix.T
    return DataMatrix, LabelList

DataMatrix, LabelList = LoadDataSet()
print(DataMatrix)
print(LabelList)

def Sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
    
def PlotLine(W, b):
    DataList = []
    LabelList = []
    fr = open('testset.txt')
    for line in fr.readlines():
        LineArray = line.strip().split()
        DataList.append([float(LineArray[0]), float(LineArray[1])])
        LabelList.append(int(LineArray[2]))  
    DataArray = np.array(DataList)
    n = np.shape(DataArray)[0]
    Weights = W.getA()
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(LabelList[i])==1:
            xcord1.append(DataArray[i,0])
            ycord1.append(DataArray[i,1])
        else:
            xcord2.append(DataArray[i,0])
            ycord2.append(DataArray[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x1 = np.arange(-3.0, 3.0, 0.1)
    x2 = - (b + Weights[0]*x1) / Weights[1]
    ax.plot(x1,x2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
def GradientDescent(datamatrix, labellist, iterations):
    X = datamatrix
    Y = np.mat(labellist)
    n,m = np.shape(X)
    b = 0
    Alpha = 0.15
    W = np.random.randn(n,1)
    for number in range(iterations):
        Z = np.dot(W.T,X) + b
        A = Sigmoid(Z)
        dZ = A - Y 
        db = float(1.0/m)*np.sum(dZ)
        dW = float(1.0/m)*np.dot(X,dZ.T)
        b = b - Alpha * db
        W = W - Alpha * dW
        Loss = -(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A)))
        J = float(np.sum(Loss)/m)
        print('迭代次数：'+str(number))
        print('成本函数值：'+str(J))
        PlotLine(W, b)
    return W, b, J
        
W, b, J = GradientDescent(DataMatrix, LabelList, 500)
print(W)
print(b)
    
PlotLine(W, b)
print(J)
