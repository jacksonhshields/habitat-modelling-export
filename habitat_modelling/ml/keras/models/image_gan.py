class GAN():
    """
    Image GAN

    Usage:
    gan = GAN(image_shape=image_shape,
              latent_dim=latent_dim,
              generator_filters=[64,64,32,32],
              discriminator_filters=[64,64],
              downsampling=args.downsampling,
              lr=args.lr)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(args.epochs):
        for batch, (img_batch, dmap_batch, depth_batch) in enumerate(train_generator):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Generate a batch of new images
            gen_imgs = gan.generator.predict(noise)

            # Train the discriminator
            d_loss_real = gan.discriminator.train_on_batch(img_batch, valid)
            d_loss_fake = gan.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = gan.combined.train_on_batch(noise, valid)
    """
    def __init__(self, image_shape, latent_dim, generator_filters, discriminator_filters, lr, activation_cfg=None, batch_norm=False, downsampling="pool"):
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_norm = batch_norm
        self.activation_cfg = activation_cfg

        self.generator_filters = generator_filters
        self.discriminator_filters = discriminator_filters

        self.downsampling = downsampling

        optimizer = Adam(self.lr, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
        """
        Creates the decoder

        Returns:
            (Model): The decoder
        """

        last_conv_shape = (self.image_shape[0] // (2 ** len(self.generator_filters)),
                           self.image_shape[1] // (2 ** len(self.generator_filters)), 3)

        noise = Input(shape=(self.latent_dim,))

        m = Dense(np.prod(last_conv_shape))(noise)
        if self.batch_norm:
            m = BatchNormalization()(m)
        m = activation_function(m, self.activation_cfg)

        m = Reshape(last_conv_shape)(m)
        m = Conv2D(3, kernel_size=(1, 1), padding='same')(m)
        if self.batch_norm:
            m = BatchNormalization()(m)
        m = activation_function(m, self.activation_cfg)

        def deconv_block(tensor, num_filters, batch_norm=False):
            """
            A block of convolutional layers consisting of Upsampling, Conv2D layer, (optional) BatchNormalization, Activation
            Args:
                tensor: (Tensor) the input tensor
                num_filters: (int) the number of filters for this layer
                batch_norm: (bool) whether to use batch normalization

            Returns:
                m: (Tensor) the output tensor

            """
            tensor = UpSampling2D()(tensor)
            tensor = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(tensor)
            if batch_norm:
                tensor = BatchNormalization()(tensor)
            tensor = activation_function(tensor, self.activation_cfg)
            return tensor

        for filt in self.generator_filters:
            m = deconv_block(m, filt, batch_norm=self.batch_norm)

        # Output
        generated = Conv2D(3, kernel_size=(1, 1), padding='same', activation='tanh')(m)

        generator = Model(noise, generated, name='generator')
        generator.summary()
        return generator

    def build_discriminator(self):
        """
        Creates the disciminator model

        Returns: (Model) The encoder model

        """
        img_input = Input(tuple(self.image_shape), name="img_input")

        m = img_input

        # ----------
        # Encoding
        # ----------

        def conv_block(tensor, num_filters, downsampling, batch_norm=False):
            """
            A block of convolutional layers consisting of 1 Conv2D layer, (optional) BatchNormalization, downsampling, Activation
            Args:
                tensor: (Tensor) the input tensor
                num_filters: (int) the number of filters for this layer
                downsampling: (str) the downsampling strategy to use, either 'pool' or 'stride'
                batch_norm: (bool) whether to use batch normalization

            Returns:
                m: (Tensor) the output tensor

            """
            if downsampling == "pool":
                tensor = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(tensor)
                if batch_norm:
                    tensor = BatchNormalization()(tensor)
                tensor = activation_function(tensor, self.activation_cfg)
                tensor = MaxPooling2D((2, 2), padding='same')(tensor)
            elif downsampling == "stride":
                tensor = Conv2D(num_filters, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(tensor)
                if batch_norm:
                    tensor = BatchNormalization()(tensor)
                tensor = activation_function(tensor, self.activation_cfg)

            else:
                raise ValueError("Downsampling has to be either stride or pool")
            return tensor

        for filt in self.discriminator_filters:
            m = conv_block(m, num_filters=filt, downsampling=self.downsampling, batch_norm=self.batch_norm)

        m = GlobalAveragePooling2D()(m)
        # m = Flatten()(m)

        m = Dense(1, name='discriminator_output')(m)

        m = Activation('sigmoid')(m)

        decision = m

        discriminator = Model(img_input, decision, name='discriminator')

        discriminator.summary()

        return discriminator