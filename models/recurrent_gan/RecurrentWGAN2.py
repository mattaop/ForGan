import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.recurrent_gan.RecurrentGAN import RecurrentGAN


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, time_series, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([time_series, interpolated], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        time_series, real_forecast = data
        # Get the batch size
        batch_size = tf.shape(real_forecast)[0]
        labels = tf.concat(
            [tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0
        )
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector

            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_forecast = self.generator([time_series, random_latent_vectors], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([time_series, fake_forecast], training=True)
                # Get the logits for real images
                real_logits = self.discriminator([time_series, real_forecast], training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, time_series, real_forecast, fake_forecast)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        print('gen', tf.shape(random_latent_vectors))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([time_series, random_latent_vectors], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([time_series, generated_images], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        print('loss')
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class WGAN2(RecurrentGAN):
    def __init__(self, cfg):
        RecurrentGAN.__init__(self, cfg)
        self.plot_rate = cfg['plot_rate']
        self.print_coverage = cfg['print_coverage']
        self.plot_folder = 'RecurrentGAN'
        self.window_size = cfg['window_size']
        self.forecasting_horizon = cfg['forecast_horizon']
        self.recurrent_forecasting = cfg['recurrent_forecasting']
        if self.recurrent_forecasting:
            self.output_size = 1
        else:
            self.output_size = self.forecasting_horizon
        self.new_training_loop = cfg['new_training_loop']
        self.noise_vector_size = cfg['noise_vector_size']
        self.discriminator_epochs = cfg['discriminator_epochs']
        self.mixed_batches = cfg['mixed_batches']
        self.mc_forward_passes = cfg['mc_forward_passes']

        self.layers = cfg['layers']
        self.num_layers = 1
        self.optimizer = keras.optimizers.Adam(cfg['learning_rate'], 0.5)
        self.loss_function = 'binary_crossentropy'
        self.weight_init = keras.initializers.RandomNormal(mean=0., stddev=0.02)
        self.model = None

    def build_model(self):
        print('=== Config===', '\nModel name:', self.model_name, '\nNoise vector size:', self.noise_vector_size,
              '\nDiscriminator epochs:', self.discriminator_epochs, '\nGenerator nodes', self.generator_nodes,
              '\nDiscriminator nodes:', self.discriminator_nodes, '\nOptimizer:', self.optimizer,
              '\nLearning rate:', self.learning_rate)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # self.discriminator.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        #self.generator.compile(loss=self.loss_function, optimizer=self.optimizer)

        wgan = WGAN(
            discriminator=self.discriminator,
            generator=self.generator,
            latent_dim=self.noise_vector_size,
            discriminator_extra_steps=3,
        )

        # Compile the wgan model
        wgan.compile(
            d_optimizer=self.optimizer,
            g_optimizer=self.optimizer,
            g_loss_fn=self.generator_loss_1,
            d_loss_fn=self.discriminator_loss_1,
        )
        self.model = wgan

    def build_generator(self):
        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = layers.Input(shape=noise_shape)
        historic_inp = layers.Input(shape=historic_shape)
        hist = historic_inp
        if self.layers == 'lstm':
            hist = layers.LSTM(self.generator_nodes, return_sequences=False)(hist)
        else:
            hist = layers.SimpleRNN(self.generator_nodes, return_sequences=False)(hist)
        # noise = layers.Flatten()(noise_inp)
        x = tf.concat([hist, noise_inp], axis=1)
        if self.dropout:
            x = layers.Dropout(0.2)(x, training=True)
        # if self.batch_norm:
        #    x = BatchNormalization()(x)
        x = layers.Dense(self.generator_nodes + self.noise_vector_size)(x)
        x = layers.ReLU()(x)
        if self.dropout:
            x = layers.Dropout(0.4)(x, training=True)
        # x = Dense(16)(x)
        # x = ReLU()(x)
        prediction = layers.Dense(self.output_size)(x)

        model = keras.Model(inputs=[historic_inp, noise_inp], outputs=prediction)
        # model.summary()
        return model

    def build_discriminator(self):

        historic_shape = (self.window_size, 1)
        future_shape = (self.output_size, 1)

        historic_inp = layers.Input(shape=historic_shape)
        future_inp = layers.Input(shape=future_shape)

        x = layers.concatenate([historic_inp, future_inp], axis=1)

        if self.layers == 'lstm':
            for i in range(self.num_layers - 1):
                x = layers.LSTM(self.discriminator_nodes, return_sequences=True)(x)
            x = layers.LSTM(self.discriminator_nodes, return_sequences=False)(x)
        else:
            for i in range(self.num_layers - 1):
                x = layers.SimpleRNN(self.discriminator_nodes, return_sequences=True)(x)
            x = layers.SimpleRNN(self.discriminator_nodes, return_sequences=False)(x)

        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        if self.dropout:
            x = layers.Dropout(0.2)(x)

        # x = LeakyReLU(alpha=0.2)(x)
        x = layers.Dense(self.discriminator_nodes)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        if self.dropout:
            x = layers.Dropout(0.4)(x)
        # x = Dropout(0.2)(x)
        # x = Dense(32)(x)
        # x = LeakyReLU(alpha=0.1)(x)
        # x = Dropout(0.2)(x)
        validity = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=[historic_inp, future_inp], outputs=validity)
        # model.summary()

        return model

    def discriminator_loss_1(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions to be used for generator
    def generator_loss_1(self, fake_img):
        return -tf.reduce_mean(fake_img)

    def fit(self, x, y, x_val=None, y_val=None, epochs=1, batch_size=32, verbose=1):

        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

        return None

