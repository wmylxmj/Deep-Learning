# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:10:43 2018

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import time

class DCGAN(object):
    
    def __init__(self, image_width=64, image_height=64, batch_size=64, z_dim=100):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.image_width = image_width
        self.image_height = image_height
        pass
    
    def load_dataset(self, floder, max_datas=1000, show_sample=False):
        resize_width = self.image_width
        resize_height = self.image_height
        images = os.listdir(floder)
        num_images = min(len(images), max_datas)
        sample = plt.imread(floder + '/' + random.choice(images[0:max_datas-1]))
        (image_width, image_height, image_channel) = sample.shape
        if show_sample:
            print('One of sample:')
            plt.imshow(sample)
            plt.show()
            pass
        dataset = np.empty((num_images, resize_width, resize_height, image_channel), dtype="float32")
        for i in range(num_images):
            img = Image.open(floder + "/" + images[i])   
            img = img.resize((resize_width, resize_height))             
            img_arr = np.asarray(img, dtype="float32")                  
            dataset[i, :, :, :] = img_arr     
            pass
        print(dataset.shape)
        with tf.Session() as sess:        
            sess.run(tf.initialize_all_variables())
            dataset = tf.reshape(dataset, [-1, resize_width, resize_height, image_channel])
            traindata = dataset * 1.0 / 127.5 - 1.0 
            traindata = tf.reshape(traindata, [-1, resize_width*resize_height*image_channel])
            trainset = sess.run(traindata)
        print('[OK] ' + str(num_images) + ' samples have been loaded')
        return trainset
    
    def generator(self, z, is_training, reuse):
        with tf.variable_scope("GL1_FC", reuse=reuse):
            output = tf.layers.dense(z, 1024*4*4, trainable=is_training)
            output = tf.reshape(output, [self.batch_size, 4, 4, 1024])
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("GL2_DC", reuse=reuse):
            output = tf.layers.conv2d_transpose(output, 512, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("GL3_DC", reuse=reuse):
            output = tf.layers.conv2d_transpose(output, 256, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("GL4_DC", reuse=reuse):
            output = tf.layers.conv2d_transpose(output, 128, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("GL5_DC", reuse=reuse):
            output = tf.layers.conv2d_transpose(output, 3, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            gen_img = tf.nn.tanh(output)
        return gen_img
    
    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope("DL1_CV", reuse=reuse):
            output = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("DL2_CV", reuse=reuse):
            output = tf.layers.conv2d(output, 128, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("DL3_CV", reuse=reuse):
            output = tf.layers.conv2d(output, 256, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("DL4_CV", reuse=reuse):
            output = tf.layers.conv2d(output, 512, [5, 5], strides=(2, 2), padding="SAME", trainable=is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope("DL5_fc", reuse=reuse):
            output = tf.layers.flatten(output)
            disc_img = tf.layers.dense(output, 1, trainable=is_training)
        return disc_img
    
    def plot_and_save(self, order, images):
        batch_size = len(images)
        n = np.int(np.sqrt(batch_size))
        image_size = np.shape(images)[2]
        n_channel = np.shape(images)[3]
        images = np.reshape(images, [-1, image_size, image_size, n_channel])
        canvas = np.empty((n * image_size, n * image_size, n_channel))
        for i in range(n):
            for j in range(n):
                canvas[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, :] = images[n*i+j].reshape(64, 64, 3)
                pass
            pass
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap="gray")
        label = "Epoch: {0}".format(order+1)
        plt.xlabel(label)
        if type(order) is str:
            file_name = order
        else:
            file_name = "face_gen" + str(order)
            pass
        plt.savefig(file_name)
        print(os.getcwd())
        print("Image saved in file: ", file_name)
        plt.close()
        pass
    
    def training(self):
        data = self.load_dataset('./faces', show_sample=True)
        x = tf.placeholder(tf.float32, shape=[None, self.image_width*self.image_height*3], name="Input_data")
        x_img = tf.reshape(x, [-1] + [self.image_width, self.image_height, 3])
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="latent_var")
        G = self.generator(z, is_training=True, reuse=False)
        D_fake_logits = self.discriminator(G, is_training=True, reuse=False)
        D_true_logits = self.discriminator(x_img, is_training=True, reuse=True)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))
        D_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_true_logits, labels=tf.ones_like(D_true_logits)))
        D_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        D_loss = D_loss_1 + D_loss_2
        total_vars = tf.trainable_variables()
        d_vars = [var for var in total_vars if "DL" in var.name]
        g_vars = [var for var in total_vars if "GL" in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_optimization = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.5).minimize(G_loss, var_list=g_vars)
            d_optimization = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.5).minimize(D_loss, var_list=d_vars)
        print("we successfully make the network")
        start_time = time.time()      
        sess = tf.Session()
        sess.run(tf.initialize_all_variables()) 
        for i in range(200):
            total_batch = int(len(data)/self.batch_size)
            d_value = 0
            g_value = 0
            # 逐个batch训练
            for j in range(total_batch):
                batch_xs = data[j*self.batch_size:j*self.batch_size + self.batch_size]
 
                # 训练判别器
                z_sampled1 = np.random.uniform(low=-1.0, high=1.0, size=[self.batch_size, self.z_dim])
                Op_d, d_ = sess.run([d_optimization, D_loss], feed_dict={x: batch_xs, z: z_sampled1})
 
                # 训练生成器
                z_sampled2 = np.random.uniform(low=-1.0, high=1.0, size=[self.batch_size, self.z_dim])
                Op_g, g_ = sess.run([g_optimization, G_loss], feed_dict={x: batch_xs, z: z_sampled2})
 
                # 尝试生成影像并保存
                images_generated = sess.run(G, feed_dict={z: z_sampled2})
                d_value += d_/total_batch
                g_value += g_/total_batch
                self.plot_and_save(i, images_generated)
 
                # 输出时间和损失函数loss
                hour = int((time.time() - start_time)/3600)
                min = int(((time.time() - start_time) - 3600*hour)/60)
                sec = int((time.time() - start_time) - 3600*hour - 60*min)
                print("Time: ", hour, "h", min, "min", sec, "sec", "   Epoch: ", i, "G_loss: ", g_value, "D_loss: ", d_value)
                pass
            pass
        pass
    
    pass
    
tf.reset_default_graph()
dcgan = DCGAN()
dcgan.training()
