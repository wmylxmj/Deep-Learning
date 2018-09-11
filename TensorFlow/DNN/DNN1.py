# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:21:55 2018

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
    
    def __init__(self, scope_name, layer_list, name = "Deep Neural Network"):
        self.name = name
        self.scope_name = scope_name
        self.costs = []
        self.parameters = {}
        self.Parameters_Init(layer_list)
        self.n_x = layer_list[0]
        self.n_y = layer_list[-1]
        # it will be used when plot the costs picture
        self.iteration_unit = 1000
        print("You created a deep neural network named '" + self.name + "'.")
        print('The layer list is ' + str(self.layer_list) + '.')
        print("You parameters will work under the scope named '" + self.scope_name + "'.")
        print("You checkpoints will be saved under 'log/" + str(self.scope_name) + "/checkpoints'." )
        pass
        
    def Parameters_Init(self, layer_list):
        with tf.variable_scope(self.scope_name):
            tf.set_random_seed(1)
            self.layer_list = layer_list[:]
            self.parameters = {}
            self.L = len(layer_list) - 1
            for l in range(1, self.L + 1):
                self.parameters['W' + str(l)] = \
                tf.get_variable("W" + str(l), [layer_list[l], layer_list[l - 1]], \
                                initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                self.parameters['b' + str(l)] = \
                tf.get_variable("b" + str(l), [layer_list[l], 1], \
                                initializer = tf.zeros_initializer())
                pass
            pass
        return self.parameters
    
    def Forward_Propagation(self, X):
        A_now = X
        for l in range(1, self.L):
            A_prev = A_now
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = tf.add(tf.matmul(W, A_prev), b) 
            A_now = tf.nn.relu(Z) 
            pass
        WL = self.parameters['W' + str(self.L)]
        bL = self.parameters['b' + str(self.L)]
        ZL = tf.add(tf.matmul(WL, A_now), bL)   
        self.ZL = ZL
        return ZL
    
    def Compute_Cost(self, Y):
        logits = tf.transpose(self.ZL)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        self.cost = cost
        return cost
    
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
    
    def Train(self, X_train, Y_train, iterations = 150, learning_rate = 0.0001, minibatch_size = 32, print_cost = True):
        self.learning_rate = learning_rate
        # to keep consistent results
        tf.set_random_seed(1)
        # to keep consistent results
        seed = 3  
        # define the global step
        global_step = tf.train.get_or_create_global_step()
        m = X_train.shape[1]
        # placeholder
        X = tf.placeholder(tf.float32, shape=[self.n_x, None])
        Y = tf.placeholder(tf.float32, shape=[self.n_y, None])
        ZL = self.Forward_Propagation(X)
        tf.summary.histogram('ZL', ZL)
        print('1')
        cost = self.Compute_Cost(Y)
        tf.summary.scalar('cost', cost)
        print('2')
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step = global_step)
        # Initialize all the variables
        init = tf.global_variables_initializer()
        print('3')
        with tf.Session() as sess:
            print('4')
            sess.run(init)
            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter("log/" + str(self.scope_name) + "/tensorboard", sess.graph)
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
                summary_str = sess.run(merged_summary_op, feed_dict={X: minibatch_X, Y: minibatch_Y})
                summary_writer.add_summary(summary_str, i)
                if print_cost and i % (10*self.iteration_unit) == 0:
                    print ("Cost after iteration %i: %f" %(i, iteration_cost))
                    pass
                if i % self.iteration_unit == 0:
                    # save the costs
                    self.costs.append(iteration_cost)
                    pass
                pass
            print('Finished')
            self.PlotCosts()
            self.parameters_array = sess.run(self.parameters)
        self.Query(X_train, Y_train)
        return self.parameters_array
        
    def PlotCosts(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per ' + str(self.iteration_unit) + ')')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
        pass
    
    def Query(self, X_test, Y_test):
        X = tf.placeholder(tf.float32, shape=[self.n_x, None])
        Y = tf.placeholder(tf.float32, shape=[self.n_y, None])
        ZL = self.Forward_Propagation(X)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            ckpt = tf.train.latest_checkpoint("log/" + self.scope_name + "/checkpoints")
            if ckpt != None:
                saver.restore(sess, ckpt)
                correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print ("Accuracy:" , 100*accuracy.eval({X: X_test, Y: Y_test}), "%")
                pass
            else:
                print('No checkpoints found!')
        pass
    
    pass

tf.reset_default_graph()
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

n = DeepNeuralNetwork("DNN1", [12288, 25, 12, 6], name = "Deep Neural Network 1")

n.iteration_unit = 1

n.Train(X_train, Y_train)

n.Query(X_train, Y_train)
n.Query(X_test, Y_test)
