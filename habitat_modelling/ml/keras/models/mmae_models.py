from __future__ import print_function, division
import numpy as np
import os
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, GlobalAveragePooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.layers
from keras import optimizers
from keras import regularizers
from keras.losses import mse, binary_crossentropy
import csv
import matplotlib.pyplot as plt
import argparse
import neptune
import json
import sys
from PIL import Image

from keras import backend as K
from habitat_modelling.datasets.keras.coco_habitat import CocoImgBathyGenerator
from habitat_modelling.ml.keras.models.model_utils import activation_function
from habitat_modelling.ml.keras.transforms.image_transforms import ImageAugmenter, ImageToFloat, ResizeImage, RandomCrop
from habitat_modelling.ml.keras.transforms.bathy_transforms import FourierFeatures, ModelFeatures
from habitat_modelling.ml.keras.models.blocks import resnet_inception_block, inception_downsample, inception_block, conv_block_downsample, deconv_block_upsample, dense_block, conv_block

class ImageVAE:
    """
    Image Variational Autoencoder

    """
    def __init__(self, image_shape, filters, input_latent_dim, output_latent_dim,
                 block_type='plain', activation_cfg=None, batch_norm=False):
        """
        Initialises the autoencoder with the default parameters
        Args:
            image_shape: (tuple) The image shape to be used. Height,width must be divisible by 2**len(block_filters).
            filters: (list) the number of filters in each block. Length of the list is equal to amount of downsampling.
            input_latent_dim: (int) the number of units in the latent dimension that the decoder takes (equal to output_latent_dim_image + output_latent_dim_bathy + 1.
            output_latent_dim: (int) the number of units in the latent dimension that the encoder outputs.
            block_type: (str) the type of blocks to use. One of {plain,inception_downsample,inception,inception_resenet}
            activation_cfg: (dict) Configure the activations used. If None, uses ReLU.
            batch_norm: (bool) Whether to use batch norm
        """
        self.decoder = None
        self.encoder = None
        self.aa = None
        self.inputs = None
        self.reconstruction_loss = None

        self.z_mean = None
        self.z_log_var = None

        self.img_input = None

        self.isolo = None

        # Params

        self.image_shape = image_shape

        self.activation_cfg = activation_cfg
        self.filters = filters
        self.batch_norm = batch_norm
        self.input_latent_dim = input_latent_dim
        self.output_latent_dim = output_latent_dim
        self.block = block_type

    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def encoder_model(self):
        """
        Creates the encoder model

        Returns: (Model) The encoder model

        """
        self.img_input = Input(tuple(self.image_shape), name="img_input")

        model_prefix = 'image/encoder/'

        m = self.img_input

        for i,ft in enumerate(self.filters):
            if self.block == 'plain':
                m = conv_block_downsample(m, ft, downsampling='stride', batch_norm=self.batch_norm, activation_cfg=self.activation_cfg, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block == 'inception':
                m = inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
                m = MaxPooling2D(name=model_prefix + 'max_pool_%d')(m)
            elif self.block == 'inception_downsample':
                m = inception_downsample(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
            elif self.block == 'resnet_inception':
                m = resnet_inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'resnet_block_%d/'%i)
                m = MaxPooling2D(name=model_prefix + 'max_pool_%d')(m)
            else:
                raise ValueError("Block type (%s) not implemented"%self.block)


        m = Flatten()(m)

        self.z_mean = Dense(self.output_latent_dim, name=model_prefix+'z_mean')(m)
        self.z_log_var = Dense(self.output_latent_dim, name=model_prefix+'z_log_var')(m)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(ImageVAE.sampling, output_shape=(self.output_latent_dim,), name=model_prefix+'z')([self.z_mean, self.z_log_var])

        encoder = Model(self.img_input, [self.z_mean, self.z_log_var, z], name='image_encoder')

        return encoder

    def decoder_model(self):
        """
        Creates the decoder

        Returns:
            (Model): The decoder
        """

        last_conv_shape = (self.image_shape[0] // (2 ** len(self.filters)), self.image_shape[1] // (2 ** len(self.filters)), 3)

        model_prefix = 'image/encoder/'

        latent_input = Input((self.input_latent_dim,), name=model_prefix+'latent_input')

        m = latent_input

        m = Dense(np.prod(last_conv_shape))(m)
        m = activation_function(m, self.activation_cfg)
        m = Reshape(last_conv_shape)(m)

        for i,ft in enumerate(self.filters[::-1]):
            if self.block == 'plain':
                m = deconv_block_upsample(m, ft, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block == 'inception':
                m = UpSampling2D(name=model_prefix+'upsample_%d'%i)(m)
                m = inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
            elif self.block == 'inception_downsample':
                m = UpSampling2D(name=model_prefix+'upsample_%d'%i)(m)
                m = inception_downsample(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
            elif self.block == 'resnet_inception':
                m = UpSampling2D(name=model_prefix+'upsample_%d'%i)(m)
                m = resnet_inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'resnet_block_%d/'%i)
            else:
                raise ValueError("Block type (%s) not implemented" % self.block)

        decoded = Conv2D(3, kernel_size=(1, 1), padding='same', activation='tanh', name=model_prefix + 'decoded_output')(m)

        decoder = Model(latent_input, decoded, name='image_decoder')
        return decoder

    def create_model(self):
        """
        Creates the variational autoencoder model
        Returns:
            None
        """
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        # self.vae = Model(self.img_input, self.decoder(self.encoder(self.img_input)[2]), name='image_vae')

    def compile_solo(self, lr):
        """
        Compiles the model for solo training, so that it can preinitialise dual network. Only call this if training solo.
        Returns:

        """
        if self.encoder is None or self.decoder is None:
            self.create_model()

        self.isolo = Model(self.img_input, self.decoder(self.encoder(self.img_input)[2]))

        optimizer = Adam(lr=lr)

        # ---------------------------------
        # Compile Bathy to Bathy model (b2b)
        # ---------------------------------
        loss_alg = 'mse'
        if loss_alg == 'mse' or loss_alg == 'mean_squared_error':
            reconstruction_loss = mse(K.flatten(self.img_input), K.flatten(self.isolo.outputs[0]))
        elif loss_alg == 'binary_crossentropy':
            reconstruction_loss = binary_crossentropy(
                K.flatten(self.img_input), K.flatten(self.isolo.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        # self.reconstruction_loss *= np.prod(self.image_shape) ** 2  # This makes loss way too large
        reconstruction_loss *= np.prod(self.image_shape)
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.isolo.add_loss(vae_loss)

        self.isolo.compile(optimizer, loss=loss_alg)



class BathyVAE:
    """

    """
    def __init__(self, bathy_shape, conv_layers, dense_layers, input_latent_dim, output_latent_dim,
                 block_type='plain', activation_cfg=None, batch_norm=False, dropout=0.1):
        """
        Initialises the autoencoder with the default parameters
        Args:
            image_shape: (tuple) The image shape to be used. Height,width must be divisible by 2**len(block_filters).
            conv_layers: (list) the number of filters in each conv. Length of the list is equal to number of conv layers
            dense_layers: (list) the number of neurons in each layer
            input_latent_dim: (int) the number of units in the latent dimension that the decoder takes (equal to output_latent_dim_image + output_latent_dim_bathy + 1.
            output_latent_dim: (int) the number of units in the latent dimension that the encoder outputs.
            block_type: (str) the type of blocks to use. One of {plain,inception_downsample,inception,inception_resenet}
            activation_cfg: (dict) Configure the activations used. If None, uses ReLU.
            batch_norm: (bool) Whether to use batch norm
            dropout: (float) dropout for the dense layers. If batch norm is on, this should be kept low (e.g. max of 0.1)
        """
        self.decoder = None
        self.encoder = None
        self.ae = None
        self.inputs = None
        self.reconstruction_loss = None

        self.z_mean = None
        self.z_log_var = None

        self.bathy_input = None

        self.bsolo = None  # Only used for training solo

        self.input_latent_dim = input_latent_dim
        self.output_latent_dim = output_latent_dim

        # Params
        self.bathy_shape = bathy_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

        self.activation_cfg = activation_cfg
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.block = block_type

    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def encoder_model(self):
        """
        Creates the encoder model

        Returns: (Model) The encoder model

        """
        self.bathy_input = Input(tuple(self.bathy_shape), name="bathy_input")

        m = self.bathy_input

        model_prefix = 'bathy/encoder/'

        for i, ft in enumerate(self.conv_layers):
            if self.block == 'plain':
                m = conv_block_downsample(m, ft, downsampling='stride', batch_norm=self.batch_norm, activation_cfg=self.activation_cfg, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block == 'inception':
                m = inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block == 'inception_downsample':
                m = inception_downsample(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block == 'resnet_inception':
                m = resnet_inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            else:
                raise ValueError("Block type (%s) not implemented" % self.block)

        m = Flatten()(m)  # Flatten after the conv layers

        for i,neurons in enumerate(self.dense_layers):
            m = dense_block(m, neurons, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, dropout=self.dropout, name_prefix=model_prefix + 'dense_block_%d/'%i)


        self.z_mean = Dense(self.output_latent_dim, name=model_prefix + 'z_mean')(m)
        self.z_log_var = Dense(self.output_latent_dim, name=model_prefix + 'z_log_var')(m)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(ImageVAE.sampling, output_shape=(self.output_latent_dim,), name=model_prefix + 'z')([self.z_mean, self.z_log_var])


        encoder = Model(self.bathy_input, [self.z_mean, self.z_log_var, z], name='bathy_encoder')

        return encoder


    def decoder_model(self):
        """
        Creates the decoder

        Returns:
            (Model): The decoder
        """
        model_prefix = 'bathy/decoder/'

        latent_input = Input((self.input_latent_dim,), name=model_prefix + 'latent_input')

        m = latent_input

        for i,neurons in enumerate(self.dense_layers[::-1]):
            m = dense_block(m, neurons, activation_cfg=self.activation_cfg, batch_norm=False, dropout=self.dropout, name_prefix=model_prefix + 'dense_block_%d/'%i)

        m = dense_block(m, np.prod(self.bathy_shape), activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, dropout=self.dropout, name_prefix=model_prefix + 'dense_block_%d/'%(i+1))

        m = Reshape(target_shape=self.bathy_shape, name=model_prefix + 'reshape_1')(m)

        for i,ft in enumerate(self.conv_layers[::-1]):
            if self.block == 'plain':
                m = conv_block(m, ft, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block == 'inception':
                m = inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block == 'resnet_inception':
                m = resnet_inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            else:
                raise ValueError("Block type (%s) not implemented" % self.block)
        # Output
        decoded = Conv2D(1, kernel_size=(1, 1), padding='same', activation='tanh', name=model_prefix+'decoded_output')(m)

        decoder = Model(latent_input, decoded, name='bathy_decoder')
        return decoder

    def create_model(self):
        """
        Creates the autoencoder model
        Returns:
            None
        """

        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()

    def compile_solo(self, lr):
        """
        Compiles the model for solo training, so that it can preinitialise dual network. Only call this if training solo.
        Returns:

        """
        if self.encoder is None or self.decoder is None:
            self.create_model()

        self.bsolo = Model(self.bathy_input, self.decoder(self.encoder(self.bathy_input)[2]))

        optimizer = Adam(lr=lr)

        # ---------------------------------
        # Compile Bathy to Bathy model (b2b)
        # ---------------------------------
        loss_alg = 'mse'
        if loss_alg == 'mse' or loss_alg == 'mean_squared_error':
            reconstruction_loss = mse(K.flatten(self.bathy_input), K.flatten(self.bsolo.outputs[0]))
        elif loss_alg == 'binary_crossentropy':
            reconstruction_loss = binary_crossentropy(
                K.flatten(self.bathy_input), K.flatten(self.bsolo.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        # self.reconstruction_loss *= np.prod(self.image_shape) ** 2  # This makes loss way too large
        reconstruction_loss *= np.prod(self.bathy_shape)
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.bsolo.add_loss(vae_loss)

        self.bsolo.compile(optimizer, loss=loss_alg)



class MultimodalVAE:
    def __init__(self, image_shape, bathy_shape,
                 image_latent_dim, bathy_latent_dim,
                 image_filters,
                 bathy_conv_filters, bathy_dense_neurons,
                 image_block_type, bathy_block_type,
                 image_activation_cfg, bathy_activation_cfg,
                 batch_norm=False, loss='mse', lr=1e-4
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
        """
        # Store the parameters
        self.image_shape = image_shape
        self.bathy_shape = bathy_shape
        self.image_latent_dim = image_latent_dim
        self.bathy_latent_dim = bathy_latent_dim
        self.depth_latent_dim = 1
        self.loss = loss
        self.lr = lr

        self.total_latent_dim = self.image_latent_dim + self.bathy_latent_dim + self.depth_latent_dim

        self.ivae = ImageVAE(image_shape=image_shape,
                             filters=image_filters,
                             input_latent_dim=self.total_latent_dim,
                             output_latent_dim=self.image_latent_dim,
                             block_type=image_block_type,
                             activation_cfg=image_activation_cfg,
                             batch_norm=batch_norm)

        self.bvae = BathyVAE(bathy_shape=bathy_shape,
                             conv_layers=bathy_conv_filters,
                             dense_layers=bathy_dense_neurons,
                             input_latent_dim=self.total_latent_dim,
                             output_latent_dim=self.bathy_latent_dim,
                             block_type=bathy_block_type,
                             activation_cfg=bathy_activation_cfg,
                             batch_norm=batch_norm)

        self.ivae.create_model()
        self.bvae.create_model()


        # ---------------------------
        # Create the inputs
        # ---------------------------
        # img_input = Input(image_shape)
        # bathy_input = Input(bathy_shape)
        depth_input = Input((1,))

        # ---------------------------
        # Create the noise inputs
        # ---------------------------
        img_noise = Input((self.image_latent_dim,))
        bathy_noise = Input((self.bathy_latent_dim,))

        # ---------------------------
        # Create the encoded ouputs
        # ---------------------------
        zi = self.ivae.encoder(self.ivae.img_input)[2]
        zb = self.bvae.encoder(self.bvae.bathy_input)[2]

        # ---------------------------------------
        # Create latent space concatenation model
        # ---------------------------------------
        cat_model = self.concatenate_model()

        # ---------------------------------
        # Create Image to Image model (i2i)
        # ---------------------------------
        i2i_latent = cat_model([zi, bathy_noise, depth_input])
        self.i2i = Model([self.ivae.img_input, bathy_noise, depth_input], self.ivae.decoder(i2i_latent))

        # ---------------------------------
        # Create Bathy to Bathy model (b2b)
        # ---------------------------------
        b2b_latent = cat_model([img_noise, zb, depth_input])
        self.b2b = Model([img_noise, self.bvae.bathy_input, depth_input], self.bvae.decoder(b2b_latent))

        # ------------------------------------------
        # Create Image + Bathy to Image model (ib2i)
        # ------------------------------------------
        ib2i_latent = cat_model([zi, zb, depth_input])
        self.ib2i = Model([self.ivae.img_input, self.bvae.bathy_input, depth_input], self.ivae.decoder(ib2i_latent))

        # ------------------------------------------
        # Create Image + Bathy to Bathy model (ib2b)
        # ------------------------------------------
        ib2b_latent = cat_model([zi, zb, depth_input])
        self.ib2b = Model([self.ivae.img_input, self.bvae.bathy_input, depth_input], self.bvae.decoder(ib2b_latent))

        # TODO replace concatenation with in decoder concatenation ??? do this

        # ---------------------------------
        # Create the optimizer
        # ---------------------------------
        adam = optimizers.Adam(lr=self.lr)

        # ---------------------------------
        # Compile Image to Image model (i2i)
        # ---------------------------------
        if self.loss == 'mse' or self.loss == 'mean_squared_error':
            reconstruction_loss = mse(K.flatten(self.ivae.img_input), K.flatten(self.i2i.outputs[0]))
        elif self.loss == 'binary_crossentropy':
            reconstruction_loss = binary_crossentropy(
                K.flatten(self.ivae.img_input), K.flatten(self.i2i.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        # self.reconstruction_loss *= np.prod(self.image_shape) ** 2  # This makes loss way too large
        reconstruction_loss *= np.prod(self.image_shape)
        kl_loss = 1 + self.ivae.z_log_var - K.square(self.ivae.z_mean) - K.exp(self.ivae.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.i2i.add_loss(vae_loss)

        self.i2i.compile(adam, loss=self.loss)

        # ---------------------------------
        # Compile Bathy to Bathy model (b2b)
        # ---------------------------------
        if self.loss == 'mse' or self.loss == 'mean_squared_error':
            reconstruction_loss = mse(K.flatten(self.bvae.bathy_input), K.flatten(self.b2b.outputs[0]))
        elif self.loss == 'binary_crossentropy':
            reconstruction_loss = binary_crossentropy(
                K.flatten(self.bvae.bathy_input), K.flatten(self.b2b.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        # self.reconstruction_loss *= np.prod(self.image_shape) ** 2  # This makes loss way too large
        reconstruction_loss *= np.prod(self.bathy_shape)
        kl_loss = 1 + self.bvae.z_log_var - K.square(self.bvae.z_mean) - K.exp(self.bvae.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.b2b.add_loss(vae_loss)

        self.b2b.compile(adam, loss=self.loss)

        # ---------------------------------
        # Create Image + Bathy to Image model (ib2i)
        # ---------------------------------
        if self.loss == 'mse' or self.loss == 'mean_squared_error':
            reconstruction_loss = mse(K.flatten(self.ivae.img_input), K.flatten(self.ib2i.outputs[0]))
        elif self.loss == 'binary_crossentropy':
            reconstruction_loss = binary_crossentropy(
                K.flatten(self.ivae.img_input), K.flatten(self.ib2i.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        # self.reconstruction_loss *= np.prod(self.image_shape) ** 2  # This makes loss way too large
        reconstruction_loss *= np.prod(self.image_shape)
        kl_loss = 1 + self.ivae.z_log_var - K.square(self.ivae.z_mean) - K.exp(self.ivae.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.ib2i.add_loss(vae_loss)

        self.ib2i.compile(adam, loss=self.loss)

        # ---------------------------------
        # Compile Image + Bathy to Bathy model (ib2b)
        # ---------------------------------
        if self.loss == 'mse' or self.loss == 'mean_squared_error':
            reconstruction_loss = mse(K.flatten(self.bvae.bathy_input), K.flatten(self.ib2b.outputs[0]))
        elif self.loss == 'binary_crossentropy':
            reconstruction_loss = binary_crossentropy(
                K.flatten(self.bvae.bathy_input), K.flatten(self.ib2b.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        # self.reconstruction_loss *= np.prod(self.image_shape) ** 2  # This makes loss way too large
        reconstruction_loss *= np.prod(self.bathy_shape)
        kl_loss = 1 + self.bvae.z_log_var - K.square(self.bvae.z_mean) - K.exp(self.bvae.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.ib2b.add_loss(vae_loss)

        self.ib2b.compile(adam, loss=self.loss)





    def concatenate_model(self):
        img_input = Input((self.image_latent_dim,))
        bathy_input = Input((self.bathy_latent_dim,))
        depth_input = Input((1,))

        x = concatenate([img_input, bathy_input, depth_input])
        return Model([img_input, bathy_input, depth_input], x)


