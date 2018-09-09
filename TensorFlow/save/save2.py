# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 12:50:20 2018

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file



train_X = np.linspace(-1, 1, 150)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

plt.plot(train_X, train_Y, 'ro', label='original data')
plt.legend()
plt.show()

tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

Z = tf.multiply(X, W) + b

plotdata = {"batchsize":[], "loss":[]}

def moving_average(a,w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

cost = tf.reduce_mean(tf.square(Y - Z))
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)


init = tf.global_variables_initializer()

num_epochs = 100
display_step = 2



with tf.train.MonitoredTrainingSession(checkpoint_dir="MonitoredTrainingSession_checkpoint/linear_model", save_checkpoint_secs = 2) as sess:
    sess.run(init)
    plotdata = {"batchsize":[], "loss":[]}
    for epoch in range(num_epochs):
        for (x, y) in zip(train_X, train_Y) :
            sess.run(optimizer, feed_dict = {X:x, Y:y})
            pass
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            #print("Epoch:", epoch+1, "cost=", loss, \
            #      "W=",sess.run(W), "b=", sess.run(b))
            if not (loss == 'NA'):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    
    print('Finished')
    print("cost=", sess.run(cost, feed_dict={X:train_X, Y:train_Y}), \
          "W=",sess.run(W), "b=", sess.run(b))
        
    plt.plot(train_X, train_Y, 'ro', label='original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"]=moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.title('minibatch run vs. training loss')
    
    plt.show()    
    
    print("x=0.2, z=", sess.run(Z, feed_dict={X:0.2}))
    
saver = tf.train.Saver()
 
init = tf.global_variables_initializer()
with tf.Session() as sess2:
    sess2.run(init)
    kpt = tf.train.latest_checkpoint("MonitoredTrainingSession_checkpoint/linear_model")
    saver.restore(sess2, kpt)
    print("x=0.2, z=", sess2.run(Z, feed_dict={X:0.2}))
