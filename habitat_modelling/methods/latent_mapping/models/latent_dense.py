
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from habitat_modelling.ml.torch.models.blocks import Dense
import json

class DenseDrop(nn.Module):
    def __init__(self, in_units, out_units, activation, dropout=None, dropout_enable=True, batch_norm=False):
        super(DenseDrop, self).__init__()
        self.dense = nn.Linear(in_units, out_units)
        self.activation = activation

        self.dropout_enable = dropout_enable
        self.dropout = dropout

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d()
        else:
            self.batch_norm = None
    def forward(self, x):
        x = self.dense(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.dropout and self.dropout_enable:
            x = F.dropout(x, p=self.dropout, training=self.dropout_enable)
        return self.activation(x)

    def enable_dropout(self, dropout_enable):
        self.dropout_enable = dropout_enable

class DropLatentModel2(nn.Module):
    def __init__(self, layer_neurons, input_dim, num_classes, dropout=None, batch_norm=False):
        """Creates the Dense Latent Model

        Args:
            layer_neurons (list): Number of neurons in each layer. Length indicates number of layers. Hardcoded as 2 below.
            input_dim (int): The number of input dimensions.
            num_classes (int): The number of classes
            dropout (float): The dropout  layer
            batch_norm (bool): Whether to use batch normalisation

        Returns:
            type: Description of returned object.

        """
        super(DropLatentModel2, self).__init__()
        # ------------------
        # Store Params
        # ------------------
        self.layer_neurons = layer_neurons
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU()
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.dropout_enable = True

        # --------------------
        # Create Dense Layers
        # --------------------

        in_units = self.input_dim
        self.dense0 = DenseDrop(in_units, layer_neurons[0], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[0]
        self.dense1 = DenseDrop(in_units, layer_neurons[1], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[1]
        self.out = nn.Linear(in_units, num_classes)

        self.enable_dropout(self.dropout_enable)

    def enable_dropout(self, dropout_enable):
        self.dropout_enable = dropout_enable
        self.dense0.enable_dropout(self.dropout_enable)
        self.dense1.enable_dropout(self.dropout_enable)

    def forward(self, x):
        x = self.dense1(self.dense0(x))
        x = self.out(x)
        # x = F.dropout(x, p=self.dropout, training=self.dropout_enable)  # TODO dropout here???
        return F.log_softmax(x, dim=1)

    def predict(self, x, num_samples=10):
        with torch.no_grad():
            y_pred = np.asarray([self.forward(x).detach().cpu().numpy().squeeze() for _ in range(num_samples)])
        return y_pred


    def dump_params(self, path):
        params = {}
        params['input_dim'] = self.input_dim
        params['layer_neurons'] = self.layer_neurons
        params['num_classes'] = self.num_classes
        params['activation'] = 'leakyrelu'
        params['dropout'] = self.dropout
        params['batch_norm'] = self.batch_norm
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)

class DropLatentModel3(nn.Module):
    def __init__(self, layer_neurons, input_dim, num_classes, dropout=None, batch_norm=False):
        """Creates the Dense Latent Model

        Args:
            layer_neurons (list): Number of neurons in each layer. Length indicates number of layers. Hardcoded as 2 below.
            input_dim (int): The number of input dimensions.
            num_classes (int): The number of classes
            dropout (float): The dropout  layer
            batch_norm (bool): Whether to use batch normalisation

        Returns:
            type: Description of returned object.

        """
        super(DropLatentModel3, self).__init__()
        # ------------------
        # Store Params
        # ------------------
        self.layer_neurons = layer_neurons
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU()
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.dropout_enable = True

        # --------------------
        # Create Dense Layers
        # --------------------

        in_units = self.input_dim
        self.dense0 = DenseDrop(in_units, layer_neurons[0], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[0]
        self.dense1 = DenseDrop(in_units, layer_neurons[1], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[1]
        self.dense2 = DenseDrop(in_units, layer_neurons[2], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[2]
        self.out = nn.Linear(in_units, num_classes)

        # self.enable_dropout(self.dropout_enable)

    def enable_dropout(self, dropout_enable):
        self.dropout_enable = dropout_enable
        self.dense0.enable_dropout(self.dropout_enable)
        self.dense1.enable_dropout(self.dropout_enable)
        self.dense2.enable_dropout(self.dropout_enable)

    def forward(self, x):
        x = self.dense2(self.dense1(self.dense0(x)))
        x = self.out(x)
        # x = F.dropout(x, p=self.dropout, training=self.dropout_enable)  # TODO dropout here???
        return F.log_softmax(x, dim=1)

    def predict(self, x, num_samples=10):
        with torch.no_grad():
            y_pred = np.asarray([self.forward(x).detach().cpu().numpy().squeeze() for _ in range(num_samples)])
        return y_pred


    def dump_params(self, path):
        params = {}
        params['input_dim'] = self.input_dim
        params['layer_neurons'] = self.layer_neurons
        params['num_classes'] = self.num_classes
        params['activation'] = 'leakyrelu'
        params['dropout'] = self.dropout
        params['batch_norm'] = self.batch_norm
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)

class DenseLatentModel(nn.Module):
    def __init__(self, layer_neurons, input_dim, num_classes, dropout=None, batch_norm=False):
        """Creates the Dense Latent Model

        Args:
            layer_neurons (list): Number of neurons in each layer. Length indicates number of layers.
            input_dim (int): The number of input dimensions.
            num_classes (int): The number of classes
            dropout (float): The dropout  layer
            batch_norm (bool): Whether to use batch normalisation

        Returns:
            type: Description of returned object.

        """
        super(DenseLatentModel, self).__init__()
        # ------------------
        # Store Params
        # ------------------
        self.layer_neurons = layer_neurons
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU()
        self.dropout = dropout
        self.batch_norm = batch_norm


        # --------------------
        # Create Dense Layers
        # --------------------

        dense_layers = []
        in_units = self.input_dim
        for n, units in enumerate(layer_neurons):
            dense_layers.append(Dense(in_units, units, self.activation, dropout=dropout, batch_norm=batch_norm))
            in_units = units

        dense_layers.append(nn.Linear(in_units, self.num_classes))
        # dense_layers.append(nn.Softmax())

        self.dense_model = nn.Sequential(*dense_layers)


    def forward(self, x):
        return F.log_softmax(self.dense_model(x), dim=1)

    def predict(self, x, num_samples=10):
        with torch.no_grad():
            y_pred = np.asarray([self.forward(x).detach().cpu().numpy().squeeze() for _ in range(num_samples)])
        return y_pred


    def dump_params(self, path):
        params = {}
        params['input_dim'] = self.input_dim
        params['layer_neurons'] = self.layer_neurons
        params['num_classes'] = self.num_classes
        params['activation'] = 'leakyrelu'
        params['dropout'] = self.dropout
        params['batch_norm'] = self.batch_norm
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)


class DropLatentNeighbours2(nn.Module):
    def __init__(self, layer_neurons, input_dim, num_classes, dropout=None, batch_norm=False, output_activation='softmax'):
        """Creates the Dense Latent Model

        Args:
            layer_neurons (list): Number of neurons in each layer. Length indicates number of layers. Hardcoded as 2 below.
            input_dim (int): The number of input dimensions.
            num_classes (int): The number of classes
            dropout (float): The dropout  layer
            batch_norm (bool): Whether to use batch normalisation

        Returns:
            type: Description of returned object.

        """
        super(DropLatentNeighbours2, self).__init__()
        # ------------------
        # Store Params
        # ------------------
        self.layer_neurons = layer_neurons
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU()
        self.dropout = dropout
        self.batch_norm = batch_norm


        if output_activation != 'softmax' and output_activation != 'log_softmax':
            raise ValueError("")
        self.output_activation = output_activation

        self.dropout_enable = True

        # --------------------
        # Create Dense Layers
        # --------------------

        in_units = self.input_dim
        self.dense0 = DenseDrop(in_units, layer_neurons[0], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[0]
        self.dense1 = DenseDrop(in_units, layer_neurons[1], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[1]
        self.out = nn.Linear(in_units, num_classes)

    def enable_dropout(self, dropout_enable):
        self.dropout_enable = dropout_enable
        self.dense0.enable_dropout(self.dropout_enable)
        self.dense1.enable_dropout(self.dropout_enable)

    def forward(self, x):
        x = self.dense1(self.dense0(x))
        x = self.out(x)
        # x = F.dropout(x, p=self.dropout, training=self.dropout_enable)  # TODO dropout here???
        # return F.log_softmax(x, dim=1)
        if self.output_activation == 'softmax':
            output = F.softmax(x, dim=1)
        elif self.output_activation == 'log_softmax':
            output = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Only softmax and log_softmax outputs are available")

        return output

    def predict(self, x, num_samples=10):
        with torch.no_grad():
            y_pred = np.asarray([self.forward(x).detach().cpu().numpy().squeeze() for _ in range(num_samples)])
        return y_pred


    def dump_params(self, path):
        params = {}
        params['input_dim'] = self.input_dim
        params['layer_neurons'] = self.layer_neurons
        params['num_classes'] = self.num_classes
        params['dropout'] = self.dropout
        params['batch_norm'] = self.batch_norm
        params['output_activation'] = self.output_activation
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)


class DropLatentNeighbours3(nn.Module):
    def __init__(self, layer_neurons, input_dim, num_classes, dropout=None, batch_norm=False, output_activation='softmax'):
        """Creates the Dense Latent Model

        Args:
            layer_neurons (list): Number of neurons in each layer. Length indicates number of layers. Hardcoded as 2 below.
            input_dim (int): The number of input dimensions.
            num_classes (int): The number of classes
            dropout (float): The dropout  layer
            batch_norm (bool): Whether to use batch normalisation

        Returns:
            type: Description of returned object.

        """
        super(DropLatentNeighbours3, self).__init__()
        # ------------------
        # Store Params
        # ------------------
        self.layer_neurons = layer_neurons
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU()
        self.dropout = dropout
        self.batch_norm = batch_norm


        if output_activation != 'softmax' and output_activation != 'log_softmax':
            raise ValueError("")
        self.output_activation = output_activation

        self.dropout_enable = True

        # --------------------
        # Create Dense Layers
        # --------------------

        in_units = self.input_dim
        self.dense0 = DenseDrop(in_units, layer_neurons[0], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[0]
        self.dense1 = DenseDrop(in_units, layer_neurons[1], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[1]
        self.dense2 = DenseDrop(in_units, layer_neurons[2], self.activation, dropout=self.dropout, dropout_enable=True, batch_norm=False)
        in_units = layer_neurons[2]
        self.out = nn.Linear(in_units, num_classes)

    def enable_dropout(self, dropout_enable):
        self.dropout_enable = dropout_enable
        self.dense0.enable_dropout(self.dropout_enable)
        self.dense1.enable_dropout(self.dropout_enable)
        self.dense2.enable_dropout(self.dropout_enable)

    def forward(self, x):
        x = self.dense2(self.dense1(self.dense0(x)))
        x = self.out(x)
        # x = F.dropout(x, p=self.dropout, training=self.dropout_enable)  # TODO dropout here???
        # return F.log_softmax(x, dim=1)
        if self.output_activation == 'softmax':
            output = F.softmax(x, dim=1)
        elif self.output_activation == 'log_softmax':
            output = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Only softmax and log_softmax outputs are available")

        return output

    def predict(self, x, num_samples=10):
        with torch.no_grad():
            y_pred = np.asarray([self.forward(x).detach().cpu().numpy().squeeze() for _ in range(num_samples)])
        return y_pred


    def dump_params(self, path):
        params = {}
        params['input_dim'] = self.input_dim
        params['layer_neurons'] = self.layer_neurons
        params['num_classes'] = self.num_classes
        params['dropout'] = self.dropout
        params['batch_norm'] = self.batch_norm
        params['output_activation'] = self.output_activation
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)