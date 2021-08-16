# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 23:45:26 2021

@author: SUBHADEEP
"""
def rpn_layer(base_layers, num_anchors):
    """
    Define the GPN layer.
    
    Input:
        base_layers: VGG layers
        num_anchors: 5

    Output:
        [x_class, x_ellipse, base_layers]
        x_class: Classification for whether it's an object
        x_ellipse: Ellipse regression
        base_layers: VGG layers
    """
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='gpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors * 2, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='gpn_out_class')(x)
    
    x_ellipse = Conv2D(num_anchors * 5, (1, 1), activation='linear', kernel_initializer='zero', name='gpn_out_ellipse')(x)

    return [x_class, x_ellipse, base_layers]