# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:43:03 2018

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

class ConvolutionalNeuralNetwork():
    
    def __init__(self, identifier, name = None, description = None):
        self.identifier = identifier
        self.name = name
        self.description = description      
        pass
    
    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        self.io_information = {'n_H0':n_H0, 'n_W0':n_W0, 'n_C0':n_C0, 'n_y':n_y}        
        self.X = tf.placeholder(tf.float32,shape=[None, n_H0, n_W0, n_C0])
        self.Y = tf.placeholder(tf.float32,shape=[None, n_y])       
        return self.X, self.Y
    
    
    pass

a = [((4, 4, 3, 8),(1, 4, 4, 1),(1, 4, 4, 1)),
     ((4, 4, 3, 8),(1, 4, 4, 1),(1, 4, 4, 1))]
    
print(list(a[0]))