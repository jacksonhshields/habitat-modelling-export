import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.losses import mse, binary_crossentropy

from keras.models import Model
from keras import optimizers

from keras.layers import Input, Dense, Lambda, Activation, GlobalAveragePooling2D, Flatten, MaxPooling2D, UpSampling2D, Reshape, BatchNormalization, GlobalMaxPooling2D, Conv2D, Conv2DTranspose, Deconv2D

from .model_utils import activation_function

class ConvVariationalAutoEncoder:
    """
    ConvVariationalAutoEncoder: A convolutional autoencoder for use with image inputs.

    To Use:
    cvae = ConvVariationalAutoEncoder(image_shape=(256,256,3),
                                     latent_dim=100,
                                     downsampling=args.downsampling,
                                     block_filters=[64,64,64,32,32],
                                     batch_norm=args.batch_norm,
                                     activation_cfg=activation_cfg)

    cvae.create_model()
    cvae.compile()
    cvae.vae.train_on_batch(X,X)

    """
    def __init__(self, image_shape, block_filters, latent_dim, downsampling,
                 lr=1e-4, loss='mean_squared_error',activation_cfg=None, batch_norm=False):
        """
        Initialises the autoencoder with the default parameters
        Args:
            image_shape: (tuple) The image shape to be used. Height,width must be divisible by 2**len(block_filters).
            block_filters: (list) the number of filters in each block. Length of the list is equal to amount of downsampling.
            latent_dim: (int) the number of units in the latent dimension
            downsampling: (str) The downsampling strategy to use. Either 'pool' or 'stride'.
            lr: (float) The learning rate for the model
            loss: (str) The loss function used, options are {mean_squared_error}
            activation_cfg: (dict) Configure the activations used. If None, uses ReLU.
            batch_norm: (bool) Whether to use batch norm
        """
        self.decoder = None
        self.encoder = None
        self.aa = None
        self.img_input = None
        self.reconstruction_loss = None

        self.z_mean = None
        self.z_log_var = None

        # Params

        self.image_shape = image_shape

        self.activation_cfg = activation_cfg
        self.block_filters = block_filters
        self.downsampling = downsampling
        self.batch_norm = batch_norm
        self.latent_dim = latent_dim
        self.lr = lr
        self.loss = loss

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

        m = self.img_input

        # ----------
        # Encoding
        # ----------

        def conv_block(m, num_filters, downsampling, batch_norm=False):
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
            if downsampling == "pool":
                m = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(m)
                if batch_norm:
                    m = BatchNormalization()(m)
                # m = Activation('relu')(m)
                m = activation_function(m, self.activation_cfg)
                m = MaxPooling2D((2, 2), padding='same')(m)
            elif downsampling == "stride":
                m = Conv2D(num_filters, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(m)
                if batch_norm:
                    m = BatchNormalization()(m)
                # m = Activation('relu')(m)
                m = activation_function(m, self.activation_cfg)

            else:
                raise ValueError("Downsampling has to be either stride or pool")
            return m

        for filt in self.block_filters:
            m = conv_block(m, num_filters=filt, downsampling=self.downsampling, batch_norm=self.batch_norm)

        m = Conv2D(3, kernel_size=(1, 1), padding='same')(m)
        if self.batch_norm:
            m = BatchNormalization()(m)
        # m = Activation('relu')(m)
        m = activation_function(m, self.activation_cfg)

        m = Flatten(name='latent_space')(m)

        self.z_mean = Dense(self.latent_dim, name='z_mean')(m)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(m)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(ConvVariationalAutoEncoder.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        encoder = Model(self.img_input, [self.z_mean, self.z_log_var, z], name='encoder')

        return encoder

    def decoder_model(self):
        """
        Creates the decoder

        Returns:
            (Model): The decoder
        """

        last_conv_shape = (self.image_shape[0] // (2 ** len(self.block_filters)), self.image_shape[1] // (2 ** len(self.block_filters)), 3)

        latent_input = Input((self.latent_dim,))

        m = Dense(np.prod(last_conv_shape))(latent_input)
        if self.batch_norm:
            m = BatchNormalization()(m)
        m = activation_function(m, self.activation_cfg)

        m = Reshape(last_conv_shape)(m)
        m = Conv2D(3, kernel_size=(1, 1), padding='same')(m)
        if self.batch_norm:
            m = BatchNormalization()(m)
        # m = Activation('relu')(m)
        m = activation_function(m, self.activation_cfg)

        def deconv_block(m, num_filters, batch_norm=False):
            """
            A block of convolutional layers consisting of Upsampling, Conv2D layer, (optional) BatchNormalization, Activation
            Args:
                m: (Tensor) the input tensor
                num_filters: (int) the number of filters for this layer
                batch_norm: (bool) whether to use batch normalization

            Returns:
                m: (Tensor) the output tensor

            """
            m = UpSampling2D()(m)
            m = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(m)
            if batch_norm:
                m = BatchNormalization()(m)
            # m = Activation('relu')(m)
            m = activation_function(m, self.activation_cfg)
            return m

        for filt in self.block_filters[::-1]:
            m = deconv_block(m, filt, batch_norm=self.batch_norm)

        # Output
        decoded = Conv2D(3, kernel_size=(1, 1), padding='same', activation='tanh')(m)

        decoder = Model(latent_input, decoded, name='decoder')
        return decoder

    def create_model(self):
        """
        Creates the autoencoder model
        Returns:
            None
        """
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        self.vae = Model(self.img_input, self.decoder(self.encoder(self.img_input)[2]), name='autoencoder')

    def compile(self):
        """
        Compiles the model
        Returns:

        """
        if self.loss == 'mse' or self.loss == 'mean_squared_error':
            self.reconstruction_loss = mse(K.flatten(self.img_input), K.flatten(self.vae.outputs[0]))
        elif self.loss == 'binary_crossentropy':
            self.reconstruction_loss = binary_crossentropy(
                K.flatten(self.img_input), K.flatten(self.vae.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        # self.reconstruction_loss *= np.prod(self.image_shape) ** 2  # This makes loss way too large
        self.reconstruction_loss *= np.prod(self.image_shape)
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(self.reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        adam = optimizers.Adam(lr=self.lr)
        self.vae.compile(adam, loss=self.loss)