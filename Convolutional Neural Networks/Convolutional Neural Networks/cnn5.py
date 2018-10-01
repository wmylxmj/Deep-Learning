# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:01:06 2018

@author: wmy
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
plt.show()
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

tf.reset_default_graph()

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

X = tf.placeholder(tf.float32,shape=[None, 64, 64, 3])
Y = tf.placeholder(tf.float32,shape=[None, 6])       

tf.set_random_seed(1)   

# layer 1
W1 = tf.get_variable('W1', [3, 3, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
A1 = tf.nn.relu(Z1)
P1 = tf.nn.max_pool(A1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 2
W2 = tf.get_variable('W2', [3, 3, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
A2 = tf.nn.relu(Z2)
P2 = tf.nn.max_pool(A2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 3
W3 = tf.get_variable('W3', [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
A3 = tf.nn.relu(Z3)
P3 = tf.nn.max_pool(A3, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 4
P3 = tf.contrib.layers.flatten(P3)
Z4 = tf.contrib.layers.fully_connected(P3, 20)

# layer 5
Z5 = tf.contrib.layers.fully_connected(Z4, 6, activation_fn=None)
tf.summary.histogram('Z', Z5)

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5,labels=Y))
tf.summary.scalar('cost', cost)

optimizer = tf.train.AdamOptimizer(0.009).minimize(cost)

init = tf.global_variables_initializer()

costs = []
seed = 3  
(m, n_H0, n_W0, n_C0) = X_train.shape   
n_y = Y_train.shape[1]  

with tf.Session() as sess:
    
    sess.run(init)
        
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/cnn1/' + 'tensorboard' + "/", sess.graph)
    for epoch in range(100):

        minibatch_cost = 0.
        num_minibatches = int(m / 64) 
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, 64, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
          
            _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
           
                
            minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
        summary_str = sess.run(merged_summary_op, feed_dict={X: minibatch_X, Y: minibatch_Y})
        summary_writer.add_summary(summary_str, epoch)
        if epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if epoch % 1 == 0:
            costs.append(minibatch_cost)
        
        
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(0.009))
    plt.show()

    # Calculate the correct predictions
    predict_op = tf.argmax(Z5, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)