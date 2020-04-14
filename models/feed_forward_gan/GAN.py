import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import normalize

from data.generate_noise import generate_noise


class GAN:
    def __init__(self):
        self.plot_rate = 100
        self.plot_folder = 'feed_forward_GAN'
        self.noise_vector_size = 10  # Try larger vector
        self.noise_type = 'normal'  # uniform

        self.optimizer = Adam(lr=0.0005, beta_1=0.5)
        self.loss_function = 'binary_crossentropy'

        # Layer that add a dimension as the last axis
        self.expand_dims = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))

        # Build and compile the discriminator
        self.discriminator = None
        # Build and compile the generator
        self.generator = None
        # The generator takes noise as input and generated imgs
        self.combined = None

    def build_gan(self):
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.loss_function, optimizer=self.optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.noise_vector_size,))

        forecast = self.generator(z)

        # For the combined model we will only train the generator
        frozen_discriminator = Model(inputs=self.discriminator.inputs, outputs=self.discriminator.outputs)
        frozen_discriminator.trainable = False
        # self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = frozen_discriminator(forecast)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(inputs=z, outputs=valid)
        self.combined.summary()
        self.combined.compile(loss=self.loss_function, optimizer=self.optimizer)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        noise_inp = Input(shape=noise_shape)

        x = Dense(16)(noise_inp)
        x = ReLU()(x)
        # x = BatchNormalization()(x)
        # x = Dense(16)(x)
        # x = ReLU()(x)
        # x = BatchNormalization()(x)

        prediction = Dense(1)(x)

        model = Model(inputs=noise_inp, outputs=prediction)
        model.summary()
        return model

    def build_discriminator(self):
        future_shape = (1,)

        future_inp = Input(shape=future_shape)

        x = Dense(64)(future_inp)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = Dropout(0.4)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=future_inp, outputs=validity)
        model.summary()

        return model

    def _generate_noise(self, batch_size):
        if self.noise_type.lower() == 'uniform':
            return np.random.uniform(-1, 1, (batch_size, self.noise_vector_size))
        return np.random.normal(0, 1, (batch_size, self.noise_vector_size))

    def _get_labels(self, batch_size, real=True):
        if real:
            return np.ones((batch_size, 1))
        else:
            return np.zeros((batch_size, 1))

    def train(self, epochs, batch_size=128, data_samples=5000, discriminator_epochs=1):
        # Set up directories
        paths = ['ims',
                 'ims/' + self.plot_folder
                 ]
        for i in paths:
            if not os.path.exists(i):
                os.makedirs(i)

        # Load the data
        data = generate_noise(data_samples)

        half_batch = int(batch_size / 2)
        forecast_mse = np.zeros(epochs)
        kl_divergence = np.zeros(epochs)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for d_epochs in range(discriminator_epochs):
                # Select a random half batch of images
                idx = np.random.randint(0, data.shape[0], half_batch)
                real_samples = data[idx]

                noise = self._generate_noise(half_batch)  # Normalisere til 1

                # Generate a half batch of new images
                gen_forecasts = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(real_samples,
                                                                self._get_labels(batch_size=half_batch, real=True))
                d_loss_fake = self.discriminator.train_on_batch(gen_forecasts,
                                                                self._get_labels(batch_size=half_batch, real=False))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = self._generate_noise(batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, self._get_labels(batch_size=batch_size, real=True))

            # Measure forecast MSE of generator
            forecast_mse[epoch] = mean_squared_error(real_samples, gen_forecasts)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                  (epoch, d_loss[0], 100*(d_loss[1]), g_loss, forecast_mse[epoch]))

            #kl_divergence[epoch] = self.kl_divergence(real_samples, gen_forecasts)
            #print("KL-divergence: ", kl_divergence[epoch])

            if epoch % self.plot_rate == 0:
                self.plot_distributions(real_samples, gen_forecasts,
                                        f'ims/' + self.plot_folder + f'/plot.{epoch:03d}.png')

        plt.figure()
        plt.plot(np.linspace(1, epochs, epochs), forecast_mse, label='Training loss generator')
        plt.legend()
        plt.show()

    def plot_distributions(self, real_samples, fake_samples, filename=None):
        sns.kdeplot(fake_samples.flatten(), color='red', alpha=0.6, label='GAN', shade=True)
        sns.kdeplot(real_samples.flatten(), color='blue', alpha=0.6, label='Real', shade=True)
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def monte_carlo_prediction(self, mc_forward_passes=500):
        data = generate_noise(mc_forward_passes)

        # predict on batch
        noise = self._generate_noise(mc_forward_passes)
        prediction = self.generator.predict(noise)

        print('KL-divergence:', self.kl_divergence(data, prediction))
        self.plot_distributions(real_samples=data, fake_samples=prediction)

    # calculate the kl divergence
    def kl_divergence(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions
                    Parameters
                    ----------
                    p, q : array-like, dtype=float, shape=n
                    Discrete probability distributions.
                    """
        p = normalize(p.reshape(-1, 1)).flatten()
        q = normalize(q.reshape(-1, 1)).flatten()
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))


if __name__ == '__main__':
    gan = GAN()
    gan.build_gan()
    gan.train(epochs=500, batch_size=512, discriminator_epochs=1)
    gan.monte_carlo_prediction(mc_forward_passes=5000)
