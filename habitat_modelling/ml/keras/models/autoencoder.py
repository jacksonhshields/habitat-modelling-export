import numpy as np

from keras import optimizers

from keras.models import Model

from keras.layers import Input, Dense, Flatten, MaxPooling2D, UpSampling2D, Reshape, BatchNormalization, Conv2D

from .model_utils import activation_function

from habitat_modelling.ml.keras.models.blocks import resnet_inception_block, inception_downsample, inception_block, conv_block_downsample, deconv_block_upsample, dense_block, conv_block



class ConvAutoEncoder:
    """
    ConvAutoEncoder: A convolutional autoencoder for use with image inputs.

    To Use:
    cae = ConvAutoEncoder(image_shape=(256,256,3),
                         latent_dim=100,
                         downsampling=args.downsampling,
                         block_filters=[64,64,64,32,32],
                         batch_norm=args.batch_norm,
                         activation_cfg=activation_cfg)

    cae.create_model()
    cae.compile()
    cae.aa.train_on_batch(X,X)

    """
    def __init__(self,
                 image_shape,
                 conv_filters,
                 latent_dim,
                 downsampling='stride',
                 block_type='plain',
                 lr=1e-4,
                 loss='mean_squared_error',
                 activation_cfg=None,
                 batch_norm=False,
                 encoder_weights=None,
                 generator_weights=None):
        """
        Initialises the autoencoder with the default parameters
        Args:
            image_shape: (tuple) The image shape to be used. Height,width must be divisible by 2**len(block_filters).
            block_filters: (list) the number of filters in each block. Length of the list is equal to amount of downsampling.
            latent_dim: (int) the number of units in the latent dimension
            downsampling: (str) The downsampling strategy to use. Either 'pool' or 'stride'.
            block_type: (str) The type of network block to use, options are {plain, inception, inception_downsample, resnet_inception}.
            lr: (float) The learning rate for the model
            loss: (str) The loss function used, options are {mean_squared_error}
            activation_cfg: (dict) Configure the activations used. If None, uses ReLU.
            batch_norm: (bool) Whether to use batch norm
            encoder_weights: (str) Path to the weights to load into the encoder. Loads by name so doesn't have to match.
            generator_weights: (str) Path to the weights to load into the generator. Loads by name so doesn't have to match.
        """
        self.image_input = None
        self.encoder = None
        self.decoder = None
        self.ae = None

        # Params
        self.image_shape = image_shape
        self.block_type = block_type

        self.image_input = None
        self.encoder = None
        self.decoder = None
        self.ae = None

        self.activation_cfg = activation_cfg
        self.conv_filters = conv_filters
        self.downsampling = downsampling
        self.batch_norm = batch_norm
        self.latent_dim = latent_dim
        self.lr = lr
        self.loss = loss

        self.encoder_weights = encoder_weights
        self.generator_weights = generator_weights

    def create_encoder(self, weights_path=None):
        """Creates the encoder

        Args:
            weights_path (str): Path to the weights of the model (loaded via name).

        Returns:
            Model: The image encoder model

        """
        # img_input = Input(tuple(self.image_shape), name="img_input")

        model_prefix = 'encoder/'

        # m = img_input
        m = self.image_input

        for i,ft in enumerate(self.conv_filters):
            if self.block_type == 'plain':
                m = conv_block_downsample(m, ft, downsampling=self.downsampling, batch_norm=self.batch_norm, activation_cfg=self.activation_cfg, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block_type == 'inception':
                m = inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
                m = MaxPooling2D(name=model_prefix + 'max_pool_%d' %i)(m)
            elif self.block_type == 'inception_downsample':
                m = inception_downsample(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
            elif self.block_type == 'resnet_inception':
                m = resnet_inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'resnet_block_%d/'%i)
                m = MaxPooling2D(name=model_prefix + 'max_pool_%d'%i)(m)
            else:
                raise ValueError("Block type (%s) not implemented"%self.block_type)

        m = Flatten()(m)

        z = Dense(self.latent_dim, name=model_prefix+'z')(m)

        encoder = Model(self.image_input, z, name='encoder')

        if weights_path:
            encoder.load_weights(weights_path, by_name=True, skip_mismatch=True)

        return encoder

    def create_generator(self, weights_path=None, solo=False):
        """
        Creates the generator, (also the decoder). Calling it the generator for consistency with GAN models.

        Args:
            weights_path (str): Path to the weights of the model (loaded via name).
            solo (str): Path to the weights of the model (loaded via name).

        Returns:
            (Model): The generator
        """

        last_conv_shape = (self.image_shape[0] // (2 ** len(self.conv_filters)), self.image_shape[1] // (2 ** len(self.conv_filters)), 3)

        model_prefix = 'image/generator/'

        zi = Input((self.latent_dim,), name=model_prefix+'zi')
        m = zi
        m = Dense(np.prod(last_conv_shape))(m)
        m = activation_function(m, self.activation_cfg)
        m = Reshape(last_conv_shape)(m)

        for i,ft in enumerate(self.conv_filters[::-1]):
            if self.block_type == 'plain':
                m = deconv_block_upsample(m, ft, batch_norm=self.batch_norm, name_prefix=model_prefix + 'conv_block_%d/'%i)
            elif self.block_type == 'inception':
                m = UpSampling2D(name=model_prefix+'upsample_%d'%i)(m)
                m = inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
            elif self.block_type == 'inception_downsample':
                m = UpSampling2D(name=model_prefix+'upsample_%d'%i)(m)
                m = inception_downsample(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'inception_block_%d/'%i)
            elif self.block_type == 'resnet_inception':
                m = UpSampling2D(name=model_prefix+'upsample_%d'%i)(m)
                m = resnet_inception_block(m, ft, activation_cfg=self.activation_cfg, batch_norm=self.batch_norm, name_prefix=model_prefix + 'resnet_block_%d/'%i)
            else:
                raise ValueError("Block type (%s) not implemented" % self.block_type)

        generated = Conv2D(self.image_shape[2], kernel_size=(1, 1), padding='same', activation='tanh', name=model_prefix + 'generated_output')(m)

        generator = Model(zi, generated, name='generator')

        if weights_path:
            generator.load_weights(weights_path, by_name=True, skip_mismatch=True)

        return generator


    def build_autoencoder(self):
        """
        Creates the autoencoder model
        Returns:
            None
        """
        self.image_input = Input(self.image_shape)

        self.encoder = self.create_encoder(self.encoder_weights)
        self.decoder = self.create_generator(self.generator_weights)
        self.ae = Model(self.image_input, self.decoder(self.encoder(self.image_input)), name='autoencoder')

        adam = optimizers.Adam(lr=self.lr)
        self.ae.compile(adam, loss=self.loss)

