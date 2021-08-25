import numpy as np
import keras
import tensorflow as tf

from keras.layers import Activation

from keras import backend as K

from keras.layers.advanced_activations import LeakyReLU, PReLU

def activation_function(m, activation_cfg, name_prefix=None):
    """
    Adds an activation function to the model, given the activation config
    Args:
        m: the tensor that is currently being built
        activation_cfg: the configuration of the activations.

    Returns:

    """
    if activation_cfg is None:
        if name_prefix is None:
            relu_name = None
        else:
            relu_name = name_prefix + '_relu'
        m = Activation('relu', name=relu_name)(m)
        return m
    if activation_cfg['name'].lower() == 'leakyrelu':
        if name_prefix is None:
            lrelu_name = None
        else:
            lrelu_name = name_prefix + '_lrelu'
        if 'alpha' in activation_cfg:
            m = LeakyReLU(alpha=activation_cfg['alpha'], name=lrelu_name)(m)
        else:
            m = LeakyReLU(name=lrelu_name)(m)
        return m
    elif activation_cfg['name'].lower() == 'prelu':
        if 'alpha_initializer' in activation_cfg:
            alpha_initializer = activation_cfg['alpha_initializer']
        else:
            alpha_initializer = 'zeros'
        if 'alpha_regularizer' in activation_cfg:
            alpha_regularizer = activation_cfg['alpha_regularizer']
        else:
            alpha_regularizer = None
        if 'alpha_constraint' in activation_cfg:
            alpha_constraint = activation_cfg['alpha_constraint']
        else:
            alpha_constraint = None
        if 'shared_axes' in activation_cfg:
            shared_axes = activation_cfg['shared_axes']
        else:
            shared_axes = None
        if name_prefix is None:
            prelu_name = None
        else:
            prelu_name = name_prefix + '_prelu'
        m = PReLU(alpha_initializer=alpha_initializer,
                  alpha_regularizer=alpha_regularizer,
                  alpha_constraint=alpha_constraint,
                  shared_axes=shared_axes,
                  name=prelu_name)(m)
        return m
