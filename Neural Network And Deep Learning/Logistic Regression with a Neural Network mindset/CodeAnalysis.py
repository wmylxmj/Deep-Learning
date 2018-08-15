# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:57:55 2018

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#导入数据集
#这里采用orig下标，因为后面将要对数据进行加工处理
train_set_x_orig, train_set_y_orig, test_set_x_orig, \
test_set_y_orig, classes = load_dataset()

#查看一张图片
#图片的索引值
index = 50
plt.imshow(train_set_x_orig[index])
#关闭标签
plt.axis('off')
#显示图片
plt.show()

#写成一个函数
def ShowPicture(index, showlabel = False):
    plt.imshow(train_set_x_orig[index])
    plt.axis('off')
    plt.show()
    if showlabel == True:
        print("y = " + str(train_set_y_orig[:,index]) + ", it's a '" + \
              classes[np.squeeze(train_set_y_orig[:,index])].decode('utf-8') + \
              "' picture")
    
ShowPicture(25, showlabel = True)

#标签不需要处理
train_set_y = train_set_y_orig
test_set_y = test_set_y_orig

print ('\n' + "-------------------------------------------------------" + '\n')
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
print('有'+str(m_train)+'张训练图片')
print('有'+str(m_test)+'张测试图片')
height_px = train_set_x_orig[0].shape[0]
width_px = train_set_x_orig[0].shape[1]
print('图片宽'+str(width_px)+'像素')
print('图片高'+str(height_px)+'像素')
print ('\n' + "-------------------------------------------------------" + '\n')
    
#变为一维向量
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print('一维化后的图片：')
image_train_data_set = train_set_x_orig.reshape(train_set_x_orig.shape[0], \
                                                width_px*height_px, 3)
plt.imshow(image_train_data_set[0])
plt.show()
print ('\n' + "-------------------------------------------------------" + '\n')
#RGB通道取值变为0到1
#标准化数据集
train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0

#激活函数：sigmoid函数
def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s

#用0初始化w和b
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    #确保w和b的格式正确，否则AssertionError
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

#传播函数
def propagate(w, b, X, Y):
    #训练数据集的数量
    m = X.shape[1]
    #正向传播
    #z = w.T*x + b
    Z = np.dot(w.T, X) + b
    #激活
    A = sigmoid(Z)
    #成本函数cost
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    #反向传播
    dZ = A - Y
    dw = float(1.0/m) * np.dot(X, dZ.T)
    db = float(1.0/m) * np.sum(dZ)
    #保护措施
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost
    
#优化函数，利用梯度下降法寻求最优解
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        #满100次记录一下
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    #可再次开发利用
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

#预测函数
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        #大于0.5取一，小于0.5取0
        Y_prediction[0, i] = np.rint(A[0,i])
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

#Logistic回归封装
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    #初始化权重值和偏置值
    w, b = initialize_with_zeros(X_train.shape[0])
    #优化权重
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    #打印正确率
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, \
          num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)   
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#测试算法
def ShowPicture2(index, picture_x, picture_y, showlabel = False):
    plt.imshow(picture_x[index])
    plt.axis('off')
    plt.show()
    if showlabel == True:
        print("y = " + str(picture_y[index]) + ", it's a '" + \
              classes[np.squeeze(picture_y[index])].decode('utf-8') + \
              "' picture")
        
def test(X_test_image, Y_test_image, w, b):
    test_x_flatten = X_test_image.reshape(X_test_image.shape[0], -1).T
    X_test = test_x_flatten/255.0
    Y_test = Y_test_image
    Y_prediction_test = predict(w, b, X_test)
    m = X_test_image.shape[0]
    for i in range(m):
        ShowPicture2(i, X_test_image, Y_test_image, showlabel = True)
        if m > 1:
            print ("y = " + str(int(np.squeeze(Y_prediction_test)[i])) + \
                   ', you predicted that it is a "' + \
                   classes[int(np.squeeze(Y_prediction_test)[i])].decode('utf-8') + \
                   '" picture.')
        else:
            print ("y = " + str(int(np.squeeze(Y_prediction_test))) + \
                   ', you predicted that it is a "' + \
                   classes[int(np.squeeze(Y_prediction_test))].decode('utf-8') + \
                   '" picture.')
   
test_pic_y_orig = np.squeeze(test_set_y_orig)   
test_picture_x = np.array([test_set_x_orig[8]])
test_picture_y = np.array([test_pic_y_orig[8]])

test(test_picture_x, test_picture_y, d['w'], d['b'])
    
a = np.zeros((1,1))
print(a)
print(str(np.squeeze(a)))
print(np.squeeze([1]))

test_pic_y_orig = np.squeeze(test_set_y_orig)   
test_picture_x = np.array([test_set_x_orig[8],
                           test_set_x_orig[3],
                           test_set_x_orig[14],
                           test_set_x_orig[35],
                           test_set_x_orig[25],
                           test_set_x_orig[45]])
test_picture_y = np.array([test_pic_y_orig[8],
                           test_pic_y_orig[3],
                           test_pic_y_orig[14],
                           test_pic_y_orig[35],
                           test_pic_y_orig[25],
                           test_pic_y_orig[45]])

print ('\n' + "-------------------------------------------------------" + '\n')
print('测试：')
print('训练中')
d = model(train_set_x, train_set_y, test_set_x, test_set_y, \
          num_iterations = 3000, learning_rate = 0.0055, print_cost = True)
# Plot learning curve (with costs)   
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
print('分类结果')
test(test_picture_x, test_picture_y, d['w'], d['b'])
