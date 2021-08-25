import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_modelling.datasets.torch.coco_habitat import CocoImgBathy
from habitat_modelling.ml.torch.models.blocks import create_vgg_down_block, create_vgg_up_block, Conv2d_same
from torch.autograd import Variable
import json
import os

class PassThrough(nn.Module):
    def forward(self, x):
        return x



def reparameterization(mu, logvar, latent_dim, Tensor):
    """
    Reparameterization trick from Kingma. Performs sampling of a normal distribution given mu (mean) and log variance (logvar).

    Args:
        mu: the predicted mean of the distribution
        logvar: the log variance of the distribution
        latent_dim: the number of latent dimensions
        Tensor: The tensor type to assign the output to. Often torch.cuda.FloatTensor or torch.FloatTensor

    Returns:

    """
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z


class ImageEncoder(nn.Module):
    def __init__(self, image_shape,
                 image_latent_dim,
                 image_conv_filters,
                 activation='dummy',
                 block_type='plain',
                 batch_norm=False,
                 variational=False,
                 categorical=False,
                 num_classes=None,
                 tensor=None):
        """Creates the image encoder

        Args:
            image_shape (tuple): Image Shape, given by (channels, width, height) - pytorch convention
            image_latent_dim (int): Size of the latent dimension as output
            image_conv_filters (list): Number of filters in each layer. Length indicates number of layers
            activation (nn.Module): The activation layer used.
            block_type (str): The block type to use
            batch_norm (bool): Whether to use batch normalisation
            variational (bool): Whether to use z sampling - used for variational or adversarial autoencoders
            categorical (bool): Whether to output two latent spaces: z, c
            tensor (Tensor): The pytorch function to convert output to specified tensor type (e.g. cuda or not). Used for VAEs or AAEs.

        Returns:
            type: Description of returned object.

        """
        super(ImageEncoder, self).__init__()
        # -----------------
        # Check parameters
        # -----------------
        if image_shape[1] % 2**len(image_conv_filters) != 0 or image_shape[2] % 2**len(image_conv_filters) != 0:
            raise ValueError("")
        # -----------------
        # Store Parameters
        # -----------------
        self.image_shape = image_shape
        self.image_latent_dim = image_latent_dim
        self.image_conv_filters = image_conv_filters
        self.activation = nn.LeakyReLU()
        self.batch_norm = batch_norm
        self.variational = variational
        self.categorical = categorical
        self.num_classes = num_classes
        self.tensor = tensor
        self.conv_layers = []
        # -----------------
        # Create layers
        # -----------------
        in_feat = image_shape[0]
        curr_shape = image_shape
        for n, feats in enumerate(image_conv_filters):
            block_layers, in_feat = create_vgg_down_block(feats, in_features=in_feat, activation=self.activation, kernel_size=3, strategy='stride', batch_norm=self.batch_norm)
            self.conv_layers.extend(block_layers)
            curr_shape = (in_feat, int(curr_shape[1] / 2), int(curr_shape[2] / 2))

        self.conv_model = nn.Sequential(*self.conv_layers)

        self.latent = nn.Linear(in_features=np.prod(curr_shape), out_features=self.image_latent_dim)
        self.last_shape = curr_shape

        if self.variational:
            self.mu = nn.Linear(in_features=np.prod(curr_shape), out_features=self.image_latent_dim)
            self.logvar = nn.Linear(in_features=np.prod(curr_shape), out_features=self.image_latent_dim)

        if self.categorical:
            self.style = nn.Sequential(nn.Linear(in_features=np.prod(curr_shape), out_features=self.image_latent_dim), nn.Tanh())
            self.category = nn.Sequential(nn.Linear(in_features=np.prod(curr_shape), out_features=self.num_classes), nn.Softmax())

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(-1, np.prod(self.last_shape))  # Flatten
        if self.variational:
            mu = self.mu(x)
            logvar = self.logvar(x)
            z = reparameterization(mu, logvar, self.image_latent_dim, self.tensor)
            return z
        elif self.categorical:
            z = self.style(x)
            c = self.category(x)
            return z,c
        else:
            z = self.latent(x)
            return z

    def dump_params_to_file(self, path):
        params = {
            "image_shape": self.image_shape,
            "image_latent_dim": self.image_latent_dim,
            "image_conv_filters": self.image_conv_filters,
            "batch_norm": self.batch_norm,
        }
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)


class ImageGenerator(nn.Module):
    def __init__(self, image_shape, image_latent_dim, image_conv_filters, activation='dummy', block_type='plain', batch_norm=False):
        """
        Creates the Image Generator

        Args:
            image_shape: (tuple) the image shape in format (channels, width, height).
            bathy_latent_dim: (int) number of dimensions in the bathymetry latent space
            image_conv_filters: (list) number of filters in each convolutional layer.
            activation: (nn.Module) The activation function to use.
            block_type: (str) The block type to use
            batch_norm: (bool) Whether to use batch normalisation.
        """
        super(ImageGenerator, self).__init__()
        # -----------------
        # Store parameters
        # -----------------
        self.image_shape = image_shape
        self.image_latent_dim = image_latent_dim
        self.image_conv_filters = image_conv_filters[::-1]
        # self.activation = activation
        self.activation = nn.LeakyReLU()
        self.batch_norm = batch_norm
        self.conv_layers = []

        if isinstance(self.image_conv_filters[0], list):
            in_feat = self.image_conv_filters[0][0]
        else:
            in_feat = self.image_conv_filters[0]

        self.starting_conv_shape = (in_feat, int(self.image_shape[1]/2**len(self.image_conv_filters)), int(self.image_shape[2]/2**len(self.image_conv_filters)))
        curr_shape = self.starting_conv_shape

        self.linear_join = nn.Linear(in_features=self.image_latent_dim,
                                     out_features=np.prod(self.starting_conv_shape))


        # in_feat = self.starting_conv_shape[0]

        # -----------------
        # Create layers
        # -----------------
        for n, feats in enumerate(self.image_conv_filters):
            block_layers, in_feat = create_vgg_up_block(feats, in_features=in_feat, activation=self.activation, kernel_size=3, batch_norm=self.batch_norm)
            self.conv_layers.extend(block_layers)
            curr_shape = (in_feat, int(curr_shape[1] * 2), int(curr_shape[2] * 2))

        last_layer = Conv2d_same(in_features=curr_shape[0], out_features=image_shape[0], kernel_size=1, activation=nn.Sigmoid())
        self.conv_layers.append(last_layer)

        self.conv_model = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        x = self.linear_join(x)
        x = x.view(-1, *self.starting_conv_shape)
        x = self.conv_model(x)
        return x

    def dump_params_to_file(self, path):
        params = {
            "image_shape": self.image_shape,
            "image_latent_dim": self.image_latent_dim,
            "image_conv_filters": self.image_conv_filters,
            "batch_norm": self.batch_norm,
        }
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)


class ImageAutoencoder:
    """
    Multi-modal semi supervised autoencoders.
    """
    def __init__(self,
                 image_shape,
                 image_latent_dim,
                 image_conv_filters,
                 image_block_type,
                 image_activation_cfg,
                 batch_norm=False,
                 image_encoder_weights=None,
                 image_generator_weights=None,
                 gpus=1
                 ):
        """
        Initialises each component model
        Args:
            image_shape: (tuple) the shape of the image. E.g. (256,256,3)
            bathy_shape: (tuple) the shape of the bathymetric map, E.g. (21,21,1)
            image_latent_dim: (int) number of latent dimensions for the image. E.g. 200
            bathy_latent_dim: (int) number of latent dimensions for the bathymetry. E.g. 100
            image_filters: (list) number of convolutional filters in each layer. E.g. [32,32,32,32]
            bathy_conv_filters: (list) number of convolutional filters in each layer. E.g. [32,32,32]
            bathy_dense_neurons: (list) number of neurons in each dense layer. E.g. [128,128]
            image_block_type: (str) the type of block to use, one of {plain, inception, inception_downsample, resnet_inception}
            bathy_block_type: (str) the type of block to use, e.g. {plain, inception, resnet_inception}
            image_activation_cfg: (dict) A dictionary specifying the activation function to use.
            bathy_activation_cfg: (dict) A dictionary specifying the activation function to use.
            batch_norm: (bool) Whether to use batch normalisation
            loss: (str) The loss function to use. One of {mse,binary_crossentropy}
            lr: (float) The learning rate to use
            image_encoder_weights: (str) path to the image encoder weights .h5 file
            image_generator_weights: (str) path to the image generator weights .h5 file
            bathy_encoder_weights: (str) path to the bathy encoder weights .h5 file
            bathy_generator_weights: (str) path to the bathy generator weights .h5 file
        """
        # Store the parameters
        self.image_shape = image_shape
        self.image_latent_dim = image_latent_dim
        self.batch_norm = batch_norm
        self.image_activation_cfg = image_activation_cfg

        self.image_conv_filters = image_conv_filters

        self.image_block_type = image_block_type

        # Set variables for weights paths
        self.image_encoder_weights = image_encoder_weights
        self.image_generator_weights = image_generator_weights

        # Image activation
        self.image_activation = nn.LeakyReLU()  # NOT USED

        # Initialise the models
        self.image_encoder = None
        self.image_generator = None
        self.zi_discriminator = None


    def create_image_autoencoder(self):
        self.image_encoder = ImageEncoder(image_shape=self.image_shape,
                                          image_latent_dim=self.image_latent_dim,
                                          image_conv_filters=self.image_conv_filters,
                                          activation=self.image_activation,
                                          block_type=self.image_block_type,
                                          batch_norm=self.batch_norm,
                                          variational=False,
                                          categorical=False,
                                          num_classes=None,
                                          tensor=None)

        self.image_generator = ImageGenerator(image_shape=self.image_shape,
                                              image_latent_dim=self.image_latent_dim,
                                              image_conv_filters=self.image_conv_filters,
                                              activation=self.image_activation,
                                              block_type=self.image_block_type,
                                              batch_norm=self.batch_norm)

    def dump_params_to_file(self, path):
        self.image_encoder.dump_params_to_file(os.path.join(os.path.dirname(path), 'encoder_params.json'))
        self.image_generator.dump_params_to_file(os.path.join(os.path.dirname(path), 'generator_params.json'))

        params = {
                "image_shape": self.image_shape,
                "image_latent_dim": self.image_latent_dim,
                "image_conv_filters": self.image_conv_filters,
                "image_block_type": self.image_block_type,
                "image_activation_cfg": self.image_activation_cfg,
                "batch_norm": self.batch_norm,
                "image_encoder_weights": self.image_encoder_weights,
                "image_generator_weights": self.image_generator_weights,
                "gpus": 1
        }

        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)





class BathyEncoder(nn.Module):
    def __init__(self, bathy_shape,
                 bathy_latent_dim,
                 bathy_conv_filters,
                 bathy_dense_layers,
                 activation,
                 block_type='plain',
                 batch_norm=False,
                 variational=False,
                 categorical=False,
                 num_classes=None,
                 tensor=None):
        """Creates the bathy encoder

        Args:
            bathy_shape (tuple): Bathy Shape, given by (channels, width, height) - pytorch convention
            bathy_latent_dim (int): Size of the latent dimension as output
            bathy_conv_filters (list): Number of filters in each layer. Length indicates number of layers
            bathy_dense_neurons (list): Number of units in each layer. Length indicates number of layers
            activation (nn.Module): The activation layer used.
            block_type (str): The block type to use
            batch_norm (bool): Whether to use batch normalisation
            variational (bool): Whether to use z sampling - used for variational or adversarial autoencoders
            categorical (bool): Whether to output two latent spaces: z, c
            tensor (Tensor): The pytorch function to convert output to specified tensor type (e.g. cuda or not). Used for VAEs or AAEs.

        Returns:
            type: Description of returned object.

        """
        super(BathyEncoder, self).__init__()
        # -----------------
        # Store Parameters
        # -----------------
        self.bathy_shape = bathy_shape
        self.bathy_latent_dim = bathy_latent_dim
        self.bathy_conv_filters = bathy_conv_filters
        self.bathy_dense_layers = bathy_dense_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.variational = variational
        self.categorical = categorical
        self.num_classes = num_classes
        self.block_type = block_type
        self.tensor = tensor
        self.conv_layers = []
        # -----------------
        # Create layers
        # -----------------
        in_feat = bathy_shape[0]
        curr_shape = bathy_shape
        for n, feats in enumerate(bathy_conv_filters):
            conv = Conv2d_same(in_feat, feats, activation=self.activation, kernel_size=3, batch_norm=self.batch_norm)
            in_feat = feats
            self.conv_layers.append(conv)
            curr_shape = (in_feat, self.bathy_shape[1], self.bathy_shape[2])

        self.after_conv_shape = curr_shape

        self.conv_model = nn.Sequential(*self.conv_layers)

        # -----------------
        # Join Layer
        # -----------------
        self.linear_join = nn.Linear(in_features=np.prod(curr_shape), out_features=np.prod(self.bathy_shape))
            
        # ------------------
        # Dense Layers
        # ------------------
        dense_layers = []
        in_units = np.prod(curr_shape)
        for n, units in enumerate(bathy_dense_layers):
            dense_layers.append(nn.Linear(in_features=in_units, out_features=units))
            dense_layers.append(self.activation)
            in_units = units

        if len(dense_layers) > 0:
            self.dense_model = nn.Sequential(*dense_layers)
        else:
            self.dense_model = None

        # -----------------
        # Output Layers
        # -----------------
        self.latent = nn.Linear(in_units, out_features=self.bathy_latent_dim)

        if self.variational:
            self.mu = nn.Linear(in_features=in_units, out_features=self.bathy_latent_dim)
            self.logvar = nn.Linear(in_features=in_units, out_features=self.bathy_latent_dim)

        if self.categorical:
            self.style = nn.Sequential(nn.Linear(in_features=in_units, out_features=self.bathy_latent_dim), nn.Tanh())
            self.category = nn.Sequential(nn.Linear(in_features=in_units, out_features=self.num_classes), nn.Softmax())

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(-1, np.prod(self.after_conv_shape))  # Flatten
        # x = self.linear_join(x)
        if self.dense_model:
            x = self.dense_model(x)

        if self.variational:
            mu = self.mu(x)
            logvar = self.logvar(x)
            z = reparameterization(mu, logvar, self.bathy_latent_dim, self.tensor)
            return z, mu, logvar
        elif self.categorical:
            z = self.style(x)
            c = self.category(x)
            return z,c
        else:
            z = self.latent(x)
            return z

    def dump_params_to_file(self, path):
        params = {
            "bathy_shape": self.bathy_shape,
            "bathy_latent_dim": self.bathy_latent_dim,
            "bathy_conv_filters": self.bathy_conv_filters,
            "bathy_dense_layers": self.bathy_dense_layers,
            "activation": "leakyrelu",
            "block_type": self.block_type,
            "batch_norm": self.batch_norm,
            "variational": self.variational,
            "categorical": self.categorical,
            "num_classes": self.num_classes,
        }
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)

class BathyGenerator(nn.Module):
    def __init__(self, bathy_shape,
                 bathy_latent_dim,
                 bathy_conv_filters,
                 bathy_dense_layers,
                 activation,
                 block_type='plain',
                 batch_norm=False,
                 variational=False,
                 categorical=False,
                 num_classes=None,
                 tensor=None):
        """Creates the bathy encoder

        Args:
            bathy_shape (tuple): Bathy Shape, given by (channels, width, height) - pytorch convention
            bathy_latent_dim (int): Size of the latent dimension as output
            bathy_conv_filters (list): Number of filters in each layer. Length indicates number of layers
            bathy_dense_neurons (list): Number of units in each layer. Length indicates number of layers
            activation (nn.Module): The activation layer used.
            block_type (str): The block type to use
            batch_norm (bool): Whether to use batch normalisation
            variational (bool): Whether to use z sampling - used for variational or adversarial autoencoders
            categorical (bool): Whether to output two latent spaces: z, c
            tensor (Tensor): The pytorch function to convert output to specified tensor type (e.g. cuda or not). Used for VAEs or AAEs.

        Returns:
            type: Description of returned object.

        """
        super(BathyGenerator, self).__init__()
        # -----------------
        # Store Parameters
        # -----------------
        self.bathy_shape = bathy_shape
        self.bathy_latent_dim = bathy_latent_dim
        self.bathy_conv_filters = bathy_conv_filters
        self.bathy_dense_layers = bathy_dense_layers
        self.block_type = block_type
        self.activation = activation
        self.batch_norm = batch_norm
        self.variational = variational
        self.categorical = categorical
        self.num_classes = num_classes
        self.tensor = tensor
        self.conv_layers = []

        # ------------------
        # Dense Layers
        # ------------------
        dense_layers = []
        in_units = np.prod(self.bathy_latent_dim)
        for n, units in enumerate(bathy_dense_layers[::-1]):
            dense_layers.append(nn.Linear(in_features=in_units, out_features=units))
            dense_layers.append(self.activation)
            in_units = units
        if len(dense_layers) > 0:
            self.dense_model = nn.Sequential(*dense_layers)
        else:
            self.dense_model = None

        # -----------------
        # Join Layer
        # -----------------
        self.linear_join = nn.Linear(in_units, out_features=np.prod(self.bathy_shape))

        self.starting_conv_shape = self.bathy_shape

        # -----------------
        # Conv layers
        # -----------------
        in_feat = bathy_shape[0]
        curr_shape = self.bathy_shape
        for n, feats in enumerate(bathy_conv_filters):
            conv = Conv2d_same(in_feat, feats, activation=self.activation, kernel_size=3, batch_norm=self.batch_norm)
            in_feat = feats
            self.conv_layers.append(conv)

        last_layer = nn.Conv2d(in_feat, bathy_shape[0], kernel_size=1, padding=0)
        self.conv_layers.append(last_layer)

        self.conv_model = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        x = self.dense_model(x)
        x = self.linear_join(x)
        x = x.view(-1, *self.starting_conv_shape)
        x = self.conv_model(x)
        return x

    def dump_params_to_file(self, path):
        params = {
            "bathy_shape": self.bathy_shape,
            "bathy_latent_dim": self.bathy_latent_dim,
            "bathy_conv_filters": self.bathy_conv_filters,
            "bathy_dense_layers": self.bathy_dense_layers,
            "activation": "leakyrelu",
            "block_type": self.block_type,
            "batch_norm": self.batch_norm,
        }
        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)



class BathyAutoencoder:
    """
    Multi-modal semi supervised autoencoders.
    """
    def __init__(self,
                 bathy_shape,
                 bathy_latent_dim,
                 bathy_conv_filters,
                 bathy_neurons,
                 bathy_block_type,
                 bathy_activation_cfg,
                 batch_norm=False,
                 bathy_encoder_weights=None,
                 bathy_generator_weights=None,
                 gpus=1
                 ):
        """
        Initialises each component model
        Args:
            image_shape: (tuple) the shape of the image. E.g. (256,256,3)
            bathy_latent_dim: (int) number of latent dimensions for the bathymetry. E.g. 100
            bathy_conv_filters: (list) number of convolutional filters in each layer. E.g. [32,32,32]
            bathy_neurons: (list) number of neurons in each dense layer. E.g. [128,128]
            bathy_block_type: (str) the type of block to use, e.g. {plain, inception, resnet_inception}
            bathy_activation_cfg: (dict) A dictionary specifying the activation function to use.
            batch_norm: (bool) Whether to use batch normalisation
            loss: (str) The loss function to use. One of {mse,binary_crossentropy}
            lr: (float) The learning rate to use
            bathy_encoder_weights: (str) path to the bathy encoder weights .h5 file
            bathy_generator_weights: (str) path to the bathy generator weights .h5 file
        """
        # Store the parameters
        self.bathy_shape = bathy_shape
        self.bathy_latent_dim = bathy_latent_dim
        self.batch_norm = batch_norm
        self.bathy_activation_cfg = bathy_activation_cfg

        self.bathy_conv_filters = bathy_conv_filters

        self.bathy_block_type = bathy_block_type

        self.bathy_neurons = bathy_neurons

        # Set variables for weights paths
        self.bathy_encoder_weights = bathy_encoder_weights
        self.bathy_generator_weights = bathy_generator_weights

        # Image activation
        self.bathy_activation = nn.LeakyReLU()


        # Initialise the models
        self.bathy_encoder = None
        self.bathy_generator = None
        self.zb_discriminator = None


    def create_bathy_autoencoder(self):
        self.bathy_encoder = BathyEncoder(bathy_shape=self.bathy_shape,
                                          bathy_latent_dim=self.bathy_latent_dim,
                                          bathy_conv_filters=self.bathy_conv_filters,
                                          bathy_dense_layers=self.bathy_neurons,
                                          activation=self.bathy_activation,
                                          block_type=self.bathy_block_type,
                                          batch_norm=self.batch_norm,
                                          variational=False,
                                          categorical=False,
                                          num_classes=None,
                                          tensor=None)

        self.bathy_generator = BathyGenerator(bathy_shape=self.bathy_shape,
                                              bathy_latent_dim=self.bathy_latent_dim,
                                              bathy_dense_layers=self.bathy_neurons,
                                              bathy_conv_filters=self.bathy_conv_filters,
                                              activation=self.bathy_activation,
                                              block_type=self.bathy_block_type,
                                              batch_norm=self.batch_norm)

    def dump_params_to_file(self, path):
        self.bathy_encoder.dump_params_to_file(os.path.join(os.path.dirname(path), 'encoder_params.json'))
        self.bathy_generator.dump_params_to_file(os.path.join(os.path.dirname(path), 'generator_params.json'))

        params = {
                "bathy_shape": self.bathy_shape,
                "bathy_latent_dim": self.bathy_latent_dim,
                "bathy_conv_filters": self.bathy_conv_filters,
                "bathy_neurons": self.bathy_neurons,
                "bathy_block_type": self.bathy_block_type,
                "bathy_activation_cfg": self.bathy_activation_cfg,
                "batch_norm": self.batch_norm,
                "bathy_encoder_weights": self.bathy_encoder_weights,
                "bathy_generator_weights": self.bathy_generator_weights,
                "gpus": 1
        }

        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)


class BathyVariationalAutoencoder:
    """
    Multi-modal semi supervised autoencoders.
    """
    def __init__(self,
                 bathy_shape,
                 bathy_latent_dim,
                 bathy_conv_filters,
                 bathy_neurons,
                 bathy_block_type,
                 bathy_activation_cfg,
                 batch_norm=False,
                 bathy_encoder_weights=None,
                 bathy_generator_weights=None,
                 tensor=None,
                 gpus=1
                 ):
        """
        Initialises each component model
        Args:
            image_shape: (tuple) the shape of the image. E.g. (256,256,3)
            bathy_latent_dim: (int) number of latent dimensions for the bathymetry. E.g. 100
            bathy_conv_filters: (list) number of convolutional filters in each layer. E.g. [32,32,32]
            bathy_neurons: (list) number of neurons in each dense layer. E.g. [128,128]
            bathy_block_type: (str) the type of block to use, e.g. {plain, inception, resnet_inception}
            bathy_activation_cfg: (dict) A dictionary specifying the activation function to use.
            batch_norm: (bool) Whether to use batch normalisation
            loss: (str) The loss function to use. One of {mse,binary_crossentropy}
            lr: (float) The learning rate to use
            bathy_encoder_weights: (str) path to the bathy encoder weights .h5 file
            bathy_generator_weights: (str) path to the bathy generator weights .h5 file
        """
        # Store the parameters
        self.bathy_shape = bathy_shape
        self.bathy_latent_dim = bathy_latent_dim
        self.batch_norm = batch_norm
        self.bathy_activation_cfg = bathy_activation_cfg
        self.tensor = tensor
        self.bathy_conv_filters = bathy_conv_filters

        self.bathy_block_type = bathy_block_type

        self.bathy_neurons = bathy_neurons

        # Set variables for weights paths
        self.bathy_encoder_weights = bathy_encoder_weights
        self.bathy_generator_weights = bathy_generator_weights

        # Image activation
        self.bathy_activation = nn.LeakyReLU()


        # Initialise the models
        self.bathy_encoder = None
        self.bathy_generator = None
        self.zb_discriminator = None


    def create_bathy_autoencoder(self):
        self.bathy_encoder = BathyEncoder(bathy_shape=self.bathy_shape,
                                          bathy_latent_dim=self.bathy_latent_dim,
                                          bathy_conv_filters=self.bathy_conv_filters,
                                          bathy_dense_layers=self.bathy_neurons,
                                          activation=self.bathy_activation,
                                          block_type=self.bathy_block_type,
                                          batch_norm=self.batch_norm,
                                          variational=True,
                                          categorical=False,
                                          num_classes=None,
                                          tensor=self.tensor)

        self.bathy_generator = BathyGenerator(bathy_shape=self.bathy_shape,
                                              bathy_latent_dim=self.bathy_latent_dim,
                                              bathy_dense_layers=self.bathy_neurons,
                                              bathy_conv_filters=self.bathy_conv_filters,
                                              activation=self.bathy_activation,
                                              block_type=self.bathy_block_type,
                                              batch_norm=self.batch_norm)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, kl_alpha=1.0):
        # reconstruction_loss = F.binary_cross_entropy(recon_x, x.view(-1, torch.prod(torch.tensor(x.shape)).item()), reduction='sum')
        reconstruction_loss = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #print(reconstruction_loss, KLD)
        return reconstruction_loss+KLD*kl_alpha

    def dump_params_to_file(self, path):
        self.bathy_encoder.dump_params_to_file(os.path.join(os.path.dirname(path), 'encoder_params.json'))
        self.bathy_generator.dump_params_to_file(os.path.join(os.path.dirname(path), 'generator_params.json'))

        params = {
                "bathy_shape": self.bathy_shape,
                "bathy_latent_dim": self.bathy_latent_dim,
                "bathy_conv_filters": self.bathy_conv_filters,
                "bathy_neurons": self.bathy_neurons,
                "bathy_block_type": self.bathy_block_type,
                "bathy_activation_cfg": self.bathy_activation_cfg,
                "batch_norm": self.batch_norm,
                "bathy_encoder_weights": self.bathy_encoder_weights,
                "bathy_generator_weights": self.bathy_generator_weights,
                "gpus": 1
        }

        json.dump(params, open(path, 'w'), indent=4, sort_keys=True)
