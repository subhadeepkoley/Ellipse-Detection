# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 23:41:06 2021

@author: SUBHADEEP
"""

#// Define the classifier.
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import average_precision_score
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy
from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

class RoiPoolingConv(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]   

    def computeOutputShape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)
        img = x[0]
        rois = x[1]
        input_shape = K.shape(img)
        outputs = []
        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')
            #// Resizing the ROIs of to match pooling size 7-by-7
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)
                
        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return K.permute_dimensions(final_output, (0, 1, 2, 3, 4))    
    
    def getConfig(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).getConfig()
        return dict(list(base_config.items()) + list(config.items()))
    

def classifier_layer(baseLayers, inputROI, numROI, nbClasses = 2):

    # // baseLayers is a VGG backbone
    # // inputROI is the list of ROI, ordered as (x0, y0, semiAx1, semiAx2)
    # // numROI is the number of ROIs to be processed at a time
    input_shape = (numROI,7,7,512)
    poolingRegions = 7
    outROIPool = RoiPoolingConv(poolingRegions, numROI)([baseLayers, inputROI])

    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out = Flatten(name='flatten')(outROIPool)
    out = Dense(4096, activation='relu', name='fc1')(out)
    out = Dropout(0.5)(out)
    out = Dense(4096, activation='relu', name='fc2')(out)
    out = Dropout(0.5)(out)

    # two output layer- classifier and regressor
    # for classify the class name of the object
    out_class = Dense(nbClasses, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nbClasses)(out)
    
    #for coordinates regression
    out_regr = Dense(4 * (nbClasses-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nbClasses)(out)

    return [out_class, out_regr]
