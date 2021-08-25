import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Dense(nn.Module):
    def __init__(self, in_units, out_units, activation, dropout=0.1, batch_norm=False):
        super(Dense, self).__init__()
        self.dense = nn.Linear(in_units, out_units)
        self.activation = activation
        if dropout and dropout > 0.0:
            self.dropout = nn.Dropout()
        else:
            self.dropout = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d()
        else:
            self.batch_norm = None
    def forward(self, x):
        x = self.dense(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return self.activation(x)

class Resnet_Block(nn.Module):
    """
    Base Resnet Block
    """
    def __init__(self, in_features, out_features, activation, batch_norm=False):
        super(Resnet_Block, self).__init__()
        self.activation = activation
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_features)
        self.conv3x3 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
    def forward(self, x):
        z = self.conv3x3(x)
        if self.batch_norm:
            z = self.batch_norm(z)
        z = self.activation(z)
        z = self.conv3x3(z)
        return x + z


class Inception_Up(nn.Module):
    """
    A convolutional block which performs upsampling by 2.
    """
    def __init__(self, in_features, out_features, activation, batch_norm=False):
        """

        Args:
            in_features: (int) the number of channels going into the block
            out_features: (int) the number of channels leaving the block
            activation: (nn.Module) the activation function e.g. nn.Relu()
            kernel_size: (int or tuple) the kernel size to use
            batch_norm: (bool) Whether to use batch normalisation.
        """
        super(Inception_Up, self).__init__()

        self.inception_block = InceptionBlock(in_features=in_features, out_features=out_features, activation=activation, batch_norm=batch_norm)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode='bilinear')
        out = self.inception_block(out)
        return out

class Inception_Down(nn.Module):
    """
    Inception that downsamples by a factor of 2
    """
    def __init__(self, in_features, out_features, activation, batch_norm=False, pooling='max'):
        super(Inception_Down, self).__init__()
        self.activation = activation
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_features)
            self.batch_norm1 = nn.BatchNorm2d(out_features)
            self.batch_norm3 = nn.BatchNorm2d(out_features)
            self.batch_norm5 = nn.BatchNorm2d(out_features)
            self.batch_norm7 = nn.BatchNorm2d(out_features)
        else:
            self.batch_norm = None

        conv_out_features = int(out_features/4)
        self.conv1x1 = nn.Conv2d(in_features, conv_out_features, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_features, conv_out_features, kernel_size=3, padding=1, stride=2)
        self.conv5x5 = nn.Conv2d(in_features, conv_out_features, kernel_size=5, padding=2, stride=2)
        self.conv7x7 = nn.Conv2d(in_features, conv_out_features, kernel_size=7, padding=3, stride=2)
        if pooling == 'max':
            self.pool = nn.MaxPool2d(2)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(2)
        else:
            raise ValueError("Pooling strategy not implemented. Options are {max, pool}")

    def forward(self, x):
        # 1x1 conv followed by pool
        c1 = self.conv1x1(x)
        if self.batch_norm:
            c1 = self.batch_norm1(c1)
        c1 = self.activation(c1)
        c1 = self.pool(c1)

        # 3x3 conv with striding
        c3 = self.conv3x3(x)
        if self.batch_norm:
            c3 = self.batch_norm3(c3)
        c3 = self.activation(c3)

        # 5z5 conv with striding
        c5 = self.conv5x5(x)
        if self.batch_norm:
            c5 = self.batch_norm5(c5)
        c5 = self.activation(c5)

        # 7x7 conv
        c7 = self.conv7x7(x)
        if self.batch_norm:
            c7 = self.batch_norm7(c7)
        c7 = self.activation(c7)

        out = torch.cat((c1,c3,c5,c7),dim=1)
        return out


class InceptionBlock(nn.Module):
    """
    Simple Inception Block
    """
    def __init__(self, in_features, out_features, activation, batch_norm=False, combine_with_1x1=False):
        """
        Initialises the inception block
        Args:
            in_features: (int) the number of channels going into the block
            out_features: (int) the number of channels leaving the block
            activation: (nn.Module) the activation function e.g. nn.Relu()
            batch_norm: (bool) whether to use batch norm
            combine_with_1x1: (bool) whether to combine the features with a 1x1 conv
        """
        super(InceptionBlock, self).__init__()

        self.activation = activation

        if not combine_with_1x1:
            if out_features % 4 != 0:
                raise ValueError("If not combining with 1x1, out_features must be divisible by 4")
            conv_out_features = int(out_features/4)
        else:
            conv_out_features = out_features

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(conv_out_features)
            self.batch_norm1 = nn.BatchNorm2d(conv_out_features)
            self.batch_norm3 = nn.BatchNorm2d(conv_out_features)
            self.batch_norm5 = nn.BatchNorm2d(conv_out_features)
            self.batch_norm7 = nn.BatchNorm2d(conv_out_features)
        else:
            self.batch_norm = None

        self.conv1x1 = nn.Conv2d(in_features, conv_out_features, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_features, conv_out_features, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_features, conv_out_features, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_features, conv_out_features, kernel_size=7, padding=3)

        self.combine_with_1x1 = combine_with_1x1
        if combine_with_1x1:
            self.combiner = nn.Conv2d(out_features*4, out_features, kernel_size=1)
            self.batch_norm_comb = nn.BatchNorm2d(out_features)

    def forward(self, x):
        # 1x1 conv path
        c1 = self.conv1x1(x)
        if self.batch_norm:
            c1 = self.batch_norm1(c1)
        c1 = self.activation(c1)

        # 3x3 conv path
        c3 = self.conv3x3(x)
        if self.batch_norm:
            c3 = self.batch_norm3(c3)
        c3 = self.activation(c3)

        # 5x5 conv path
        c5 = self.conv5x5(x)
        if self.batch_norm:
            c5 = self.batch_norm5(c5)
        c5 = self.activation(c5)

        # 7x7 conv path
        c7 = self.conv7x7(x)
        if self.batch_norm:
            c7 = self.batch_norm7(c7)
        c7 = self.activation(c7)

        out = torch.cat((c1,c3,c5,c7),dim=1)
        if self.combine_with_1x1:
            out = self.combiner(out)
            if self.batch_norm:
                out = self.batch_norm_comb(out)
            out = self.activation(out)
        return out


def create_vgg_down_block(filters, in_features, activation, kernel_size=3, strategy='stride', batch_norm=False):
    """
    Creates a list of convolutional blocks similar to VGG network. The last layer is used to downsample by the given strategy. Only by 2 is supported.

    Args:
        filters: (list) The number of convolutional filters in each layer. Length of list denotes the number of layers.
        in_features: (int) The number of input channels/features
        activation: (nn.Module) The activation to use for each layer
        kernel_size: (int) The kernel size for each convolutional filter
        strategy: (str) the downsampling strategy to use. Options are {stride,avg,max}
        batch_norm: (nn.Module) Whether to use batch norm. Applied after each layer.
    Returns:
        (list): A list of nn.Modules containing the convolutional blocks
        (out_features): The output features in the last layer for convenience.
    """
    layers = []
    if isinstance(filters, int):
        filters = [filters]

    if not isinstance(filters, list):
        raise ValueError("Filters has to be a list")

    in_feat = in_features
    for n in range(len(filters))[:-1]:
        out_feat = filters[n]
        conv = Conv2d_same(in_features=in_feat, out_features=out_feat, activation=activation, kernel_size=kernel_size, batch_norm=batch_norm)
        layers.append(conv)
        in_feat = out_feat

    layers.append(Conv2d_Down(in_features=in_feat, out_features=filters[-1], activation=activation, kernel_size=kernel_size, strategy=strategy, batch_norm=batch_norm))
    out_feat = filters[-1]
    return layers, out_feat

def create_vgg_up_block(filters, in_features, activation, kernel_size=3, batch_norm=False):
    """
    Creates a list of convolutional blocks similar to VGG network. The first layer in the block upsamples.

    Args:
        filters: (list) The number of convolutional filters in each layer. Length of list denotes the number of layers.
        in_features: (int) The number of input channels/features
        activation: (nn.Module) The activation to use for each layer
        kernel_size: (int) The kernel size for each convolutional filter
        batch_norm: (nn.Module) Whether to use batch norm. Applied after each layer.
    Returns:
        (list): A list of nn.Modules containing the convolutional blocks
        (out_features): The output features in the last layer for convenience.
    """
    layers = []
    if isinstance(filters, int):
        filters = [filters]

    if not isinstance(filters, list):
        raise ValueError("Filters has to be a list")

    in_feat = in_features

    layers.append(Conv2d_Up(in_features=in_feat, out_features=filters[0], activation=activation, kernel_size=kernel_size, batch_norm=batch_norm))
    in_feat = filters[0]

    for n in range(len(filters))[1:]:
        out_feat = filters[n]
        conv = Conv2d_same(in_features=in_feat, out_features=out_feat, activation=activation, kernel_size=kernel_size, batch_norm=batch_norm)
        layers.append(conv)
        in_feat = out_feat

    out_feat = filters[-1]
    return layers, out_feat


class Conv2d_Down(nn.Module):
    """
    A convolutional block which performs downsampling by 2.
    """
    def __init__(self, in_features, out_features, activation, kernel_size=3, strategy='stride', batch_norm=False):
        """

        Args:
            in_features: (int) the number of channels going into the block
            out_features: (int) the number of channels leaving the block
            activation: (nn.Module) the activation function e.g. nn.Relu()
            kernel_size: (int or tuple) the kernel size to use
            strategy: (str) The downsampling strategy to use, one of {stride,avg,max}
            batch_norm: (bool) Whether to use batch normalisation.
        """
        super(Conv2d_Down, self).__init__()
        self.strategy = strategy
        self.activation = activation
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_features)
        else:
            self.batch_norm = None

        if strategy == 'stride':
            self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2))
        elif strategy == 'avg':
            self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2))
            self.pool = nn.AvgPool2d(2)
        elif strategy == 'max':
            self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2))
            self.pool = nn.MaxPool2d(2)
        else:
            raise ValueError("Downsampling strategy needs to be one of {stride, avg, max}")

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        out = self.activation(out)
        if self.strategy == 'avg' or self.strategy == 'max':
            out = self.pool(out)
        return out

class Conv2d_same(nn.Module):
    """
    A convolutional block which performs upsampling by 2.
    """
    def __init__(self, in_features, out_features, activation, kernel_size=3, batch_norm=False):
        """

        Args:
            in_features: (int) the number of channels going into the block
            out_features: (int) the number of channels leaving the block
            activation: (nn.Module) the activation function e.g. nn.Relu()
            kernel_size: (int or tuple) the kernel size to use
            batch_norm: (bool) Whether to use batch normalisation.
        """
        super(Conv2d_same, self).__init__()
        self.activation = activation
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_features)
        else:
            self.batch_norm = None

        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,padding=int((kernel_size-1)/2))

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        out = self.activation(out)
        return out

class Conv2d_Up(nn.Module):
    """
    A convolutional block which performs upsampling by 2.
    """
    def __init__(self, in_features, out_features, activation, kernel_size=3, batch_norm=False):
        """

        Args:
            in_features: (int) the number of channels going into the block
            out_features: (int) the number of channels leaving the block
            activation: (nn.Module) the activation function e.g. nn.Relu()
            kernel_size: (int or tuple) the kernel size to use
            batch_norm: (bool) Whether to use batch normalisation.
        """
        super(Conv2d_Up, self).__init__()
        self.activation = activation
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_features)
        else:
            self.batch_norm = None

        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,padding=int((kernel_size-1)/2))

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode='bilinear')
        out = self.conv(out)
        if self.batch_norm:
            out = self.batch_norm(out)
        out = self.activation(out)
        return out
