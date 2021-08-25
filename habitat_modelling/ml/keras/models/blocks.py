import keras
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, GlobalAveragePooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.layers

from habitat_modelling.ml.keras.models.model_utils import activation_function


def dense_block(tensor, num_neurons, batch_norm=False, activation_cfg=None, dropout=0.0, name_prefix=None, kernel_reg_fn=None, bias_reg_fn=None):
    """
    A dense layer.

    Args:
        tensor:
        num_neurons:
        batch_norm:
        dropout:

    Returns:

    """
    if name_prefix is None:
        dense_name = None
        bn_name = None
        drop_name = None
        activation_name = None
    else:
        dense_name = name_prefix + 'dense_1'
        bn_name = name_prefix + 'bn_1'
        drop_name = name_prefix + 'dropout_1'
        activation_name = name_prefix + 'activation_1'
    tensor = Dense(num_neurons, name=dense_name, kernel_regularizer=kernel_reg_fn, bias_regularizer=bias_reg_fn)(tensor)
    if batch_norm:
        tensor = BatchNormalization(name=bn_name)(tensor)
    if dropout > 0.0:
        tensor = Dropout(dropout, name=drop_name)(tensor)
    tensor = activation_function(tensor, activation_cfg, name_prefix=activation_name)
    return tensor


def inception_block_basic(x, filters, activation_cfg=None, batch_norm=False, name_prefix=None):
    """A basic inception block.

    Args:
        x (Tensor): input tensor
        filters (int): The number of filters.
        activation_cfg (dict): The activation configuration to use.
        batch_norm (bool): Whether to use batch normalisation.

    Returns:
        Tensor: Output tensor

    """
    if name_prefix is None:
        dense_name = None
        bn_name = None
        drop_name = None
        activation_name = None
    else:
        dense_name = name_prefix + 'dense_1'
        bn_name = name_prefix + 'bn_1'
        drop_name = name_prefix + 'dropout_1'
        activation_name = name_prefix + 'act_1'

    prev_filters = int(x.shape[3])
    c_list = []
    for i,k in enumerate([1, 3, 5, 7]):
        if name_prefix:
            conv_name = name_prefix + 'conv2d_k' + str(k) + '_' + str(i)
        else:
            conv_name = None
        c = Conv2D(filters=filters, kernel_size=k, padding='same', name=conv_name)(x)
        c = activation_function(c, activation_cfg, name_prefix=activation_name)
        c_list.append(c)

    x = concatenate(c_list)

    if name_prefix:
        conv_name = name_prefix + 'conv2d_k1_' + str(i+1)
        activation_name = name_prefix + 'act_1'
    else:
        conv_name = None

    x = Conv2D(prev_filters, kernel_size=1, padding='same', name=conv_name)(x)
    x = activation_function(x, activation_cfg, name_prefix=activation_name)
    return x

def inception_block(x, filters, activation_cfg=None, batch_norm=False, name_prefix=None):
    """A basic inception block

    Args:
        x (Tensor): The tensor.
        filters (int): The output filters.
        activation_cfg (dict): An activation config.
        batch_norm (bool): Whether to use batch normalisation.

    Returns:
        Tensor: The output.

    """
    prev_filters = int(x.shape[3])

    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k1_1'
        bn_name = name_prefix + 'bn_1'
        activation_name = name_prefix + 'act_1'

    # Channel 0
    c0 = Conv2D(filters=filters, kernel_size=1, padding='same', name=conv_name)(x)
    if batch_norm:
        c0 = BatchNormalization(name=bn_name)(c0)
    c0 = activation_function(c0, activation_cfg, name_prefix=activation_name)

    if name_prefix is None:
        conv_name = None
        bn_name = None
        bn_name2 = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k1_2'
        conv_name2 = name_prefix + 'conv2d_k3_2'
        bn_name = name_prefix + 'bn_2'
        bn_name2 = name_prefix + 'bn2_2'
        activation_name = name_prefix + 'act_2'
        activation2_name = name_prefix + 'act2_2'

    # Channel 1
    c1 = Conv2D(filters=filters, kernel_size=1, padding='same', name=conv_name)(x)
    if batch_norm:
        c1 = BatchNormalization(name=bn_name)(c1)
    c1 = activation_function(c1, activation_cfg, name_prefix=activation_name)
    c1 = Conv2D(filters=filters, kernel_size=3, padding='same', name=conv_name2)(c1)
    if batch_norm:
        c1 = BatchNormalization(name=bn_name2)(c1)
    c1 = activation_function(c1, activation_cfg, name_prefix=activation2_name)

    if name_prefix is None:
        conv_name = None
        bn_name = None
        bn_name2 = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k1_3'
        conv_name2 = name_prefix + 'conv2d_k5_3'
        bn_name = name_prefix + 'bn_3'
        bn_name2 = name_prefix + 'bn2_3'
        activation_name = name_prefix + 'act_3'
        activation2_name = name_prefix + 'act2_3'

    # Channel 2
    c2 = Conv2D(filters=filters, kernel_size=1, padding='same', name=conv_name)(x)
    if batch_norm:
        c2 = BatchNormalization(name=bn_name)(c2)
    c2 = activation_function(c2, activation_cfg, name_prefix=activation_name)
    c2 = Conv2D(filters=filters, kernel_size=5, padding='same', name=conv_name2)(c2)
    if batch_norm:
        c2 = BatchNormalization(name=bn_name2)(c2)
    c2 = activation_function(c2, activation_cfg, name_prefix=activation2_name)

    x = concatenate([c0, c1, c2])

    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k1_4'
        activation_name = name_prefix + 'act_4'


    x = Conv2D(prev_filters, kernel_size=1, padding='same', name=conv_name)(x)
    x = activation_function(x, activation_cfg, name_prefix=activation_name)
    return x


def resnet_inception_block(x, filters, activation_cfg=None, batch_norm=False, name_prefix=None):
    """ResNet Inception Block

    Args:
        x (Tensor): The input tensor.
        filters (int): The number of input filters.
        activation_cfg (dict): The activation function.
        batch_norm (bool): Whether to use batch normalisation.

    Returns:
        Tensor: The output tensor

    """
    ib = inception_block(x, filters, name_prefix=name_prefix)
    x = keras.layers.add([x, ib])
    return x

def inception_downsample(x, filters, activation_cfg=None, batch_norm=False, name_prefix=None):
    """Inception downsample.

    Args:
        x (Tensor): Input tensor.
        filters (int): .
        activation_cfg (dict): Description of parameter `activation_cfg`.
        batch_norm (bool): Description of parameter `batch_norm`.

    Returns:
        Tensor: The output tensor

    """
    prev_filters = int(x.shape[3])

    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k1_1'
        bn_name = name_prefix + 'bn_1'
        activation_name = name_prefix + 'act_1'
        activation2_name = name_prefix + 'act2_1'

    # Channel 0
    c0 = MaxPooling2D(2)(x)
    c0 = Conv2D(filters=filters, kernel_size=1, padding='same', name=conv_name)(c0)
    if batch_norm:
        c0 = BatchNormalization()(c0)
    c0 = activation_function(c0, activation_cfg, name_prefix=activation_name)

    # print("c0", c0.shape)
    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k2_2'
        bn_name = name_prefix + 'bn_2'
        activation_name = name_prefix + 'act_2'

    # Channel 1
    c1 = Conv2D(filters=filters, kernel_size=3, strides=(2,2), padding='same', name=conv_name)(x)
    if batch_norm:
        c1 = BatchNormalization(name=bn_name)(c1)
    c1 = activation_function(c1, activation_cfg, name_prefix=activation2_name)

    # print("c1", c1.shape)
    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k5_3'
        bn_name = name_prefix + 'bn_3'
        activation_name = name_prefix + 'act_3'

    # Channel 2
    c2 = Conv2D(filters=filters, kernel_size=5, strides=(2,2), padding='same', name=conv_name)(x)
    if batch_norm:
        c2 = BatchNormalization(name=bn_name)(c2)
    c2 = activation_function(c2, activation_cfg, name_prefix=activation_name)

    # print("c2", c2.shape)

    x = concatenate([c0, c1, c2])

    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k5_4'
        bn_name = name_prefix + 'bn_4'
        activation_name = name_prefix + 'act_4'

    x = Conv2D(filters, kernel_size=1, padding='same', name=conv_name)(x)
    x = activation_function(x, activation_cfg, name_prefix=activation_name)
    return x


def conv_block(m, num_filters, batch_norm=False, activation_cfg=None, name_prefix=None):
    """
    A block of convolutional layers consisting of 1 Conv2D layer, (optional) BatchNormalization, downsampling, Activation
    Args:
        m: (Tensor) the input tensor
        num_filters: (int) the number of filters for this layer
        downsampling: (str) the downsampling strategy to use, either 'pool' or 'stride'
        batch_norm: (bool) whether to use batch normalization

    Returns:
        m: (Tensor) the output tensor

    """
    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k3_1'
        bn_name = name_prefix + 'bn_1'
        activation_name = name_prefix + 'act_1'

    m = Conv2D(num_filters, kernel_size=(3, 3), padding='same', name=conv_name)(m)
    if batch_norm:
        m = BatchNormalization(name=bn_name)(m)
    m = activation_function(m, activation_cfg, name_prefix=activation_name)
    return m


def conv_block_downsample(m, num_filters, downsampling, batch_norm=False, activation_cfg=None, name_prefix=None):
    """
    A block of convolutional layers consisting of 1 Conv2D layer, (optional) BatchNormalization, downsampling, Activation
    Args:
        m: (Tensor) the input tensor
        num_filters: (int) the number of filters for this layer
        downsampling: (str) the downsampling strategy to use, either 'pool' or 'stride'
        batch_norm: (bool) whether to use batch normalization

    Returns:
        m: (Tensor) the output tensor

    """
    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k3_1'
        bn_name = name_prefix + 'bn_1'
        activation_name = name_prefix + 'act_1'

    if downsampling == "pool":
        m = Conv2D(num_filters, kernel_size=(3, 3), padding='same', name=conv_name)(m)
        if batch_norm:
            m = BatchNormalization(name=bn_name)(m)
        # m = Activation('relu')(m)
        m = activation_function(m, activation_cfg, name_prefix=activation_name)
        m = MaxPooling2D((2, 2), padding='same')(m)
    elif downsampling == "stride":
        m = Conv2D(num_filters, kernel_size=(3, 3), padding='same', strides=(2, 2), name=conv_name)(m)
        if batch_norm:
            m = BatchNormalization(name=bn_name)(m)
        # m = Activation('relu')(m)
        m = activation_function(m, activation_cfg, name_prefix=activation_name)

    else:
        raise ValueError("Downsampling has to be either stride or pool")
    return m


def deconv_block_upsample(m, num_filters, batch_norm=False, activation_cfg=None, name_prefix=None):
    """
    A block of convolutional layers consisting of Upsampling, Conv2D layer, (optional) BatchNormalization, Activation
    Args:
        m: (Tensor) the input tensor
        num_filters: (int) the number of filters for this layer
        batch_norm: (bool) whether to use batch normalization

    Returns:
        m: (Tensor) the output tensor

    """
    if name_prefix is None:
        conv_name = None
        bn_name = None
        activation_name = None
    else:
        conv_name = name_prefix + 'conv2d_k3_1'
        bn_name = name_prefix + 'bn_1'
        activation_name = name_prefix + 'act_1'

    m = UpSampling2D()(m)
    m = Conv2D(num_filters, kernel_size=(3, 3), padding='same', name=conv_name)(m)
    if batch_norm:
        m = BatchNormalization(name=bn_name)(m)
    # m = Activation('relu')(m)
    m = activation_function(m, activation_cfg, name_prefix=activation_name)
    return m
