import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import mean_absolute_error,  sparse_categorical_crossentropy, binary_crossentropy
from keras.losses import mean_squared_error as mse
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import normalize
from keras import backend as K
from config.load_config import load_config_file
from data.generate_noise import generate_noise
from utility.compute_statistics import compute_coverage
from utility.diversity_sensitive_loss import DiversitySensitiveLoss
from models.feed_forward_gan.GAN import GAN


class DSGAN(GAN):
    def __init__(self, cfg):
        GAN.__init__(self, cfg)
        self.plot_folder = 'feed_forward_DSGAN'
        self.noise_vector_size = 50  # Try larger vector
        self.noise_type = 'normal'  # uniform

        self.alpha = 0.01
        self.beta = 0.01
        self.tau = 5
        self.diversity_loss = DiversitySensitiveLoss(self.alpha, self.beta, self.tau, self.discriminator_loss)
        self.generator_loss = self.diversity_loss.dummy_loss
        self.combined_loss = self.diversity_loss.loss_function

        # Layer that add a dimension as the last axis
        self.expand_dims = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))

        # Build and compile the discriminator
        self.discriminator = None
        # Build and compile the generator
        self.generator = None
        # The generator takes noise as input and generated imgs
        self.combined = None

    def build_model(self):
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.discriminator_loss, optimizer=self.optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.generator_loss, optimizer=self.optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.noise_vector_size,))
        z2 = Input(shape=(self.noise_vector_size,))

        forecast = self.generator(z)
        forecast2 = self.generator(z2)

        # For the combined model we will only train the generator
        frozen_discriminator = Model(inputs=self.discriminator.inputs, outputs=self.discriminator.outputs)
        frozen_discriminator.trainable = False
        # self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = frozen_discriminator(forecast)
        y_true = Input(shape=(1,))

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        loss = Lambda(self.combined_loss, output_shape=(1,))([y_true, valid, z, z2, forecast, forecast2])
        self.combined = Model(inputs=[z, z2, y_true], outputs=loss)
        self.combined.summary()
        self.combined.compile(loss=self.generator_loss, optimizer=self.optimizer)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        noise_inp = Input(shape=noise_shape)

        x = Dense(16)(noise_inp)
        x = ReLU()(x)
        x = BatchNormalization()(x)
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
        x = Dropout(0.1)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.4)(x)
        validity = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))(x)

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

    def train(self, epochs, batch_size=128, data_samples=5000):
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

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for d_epochs in range(self.discriminator_epochs):
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

            noise1 = self._generate_noise(batch_size)
            noise2 = self._generate_noise(batch_size)
            #self.diversity_loss.loss_function([tf.cast(self._get_labels(batch_size=batch_size, real=True), tf.float32),
            #                                   tf.cast(self.discriminator.predict(self.generator.predict(noise1)), tf.float32),
            #                                   noise1, noise2, self.generator.predict(noise1),
            #                                   self.generator.predict(noise2)])

            # Train the generator
            g_loss = self.combined.train_on_batch([noise1, noise2, self._get_labels(batch_size=batch_size, real=True)],
                                                  self._get_labels(batch_size=batch_size, real=True))

            # Measure forecast MSE of generator
            try:
                forecast_mse[epoch] = mean_squared_error(real_samples, gen_forecasts)
            except ValueError:
                print(gen_forecasts)
                forecast_mse[epoch] = 0

            #kl_divergence[epoch] = self.kl_divergence(real_samples, gen_forecasts)
            #print("KL-divergence: ", kl_divergence[epoch])

            if epoch % self.plot_rate == 0:
                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                      (epoch, d_loss[0], 100 * (d_loss[1]), g_loss, forecast_mse[epoch]))
                self.plot_distributions(real_samples, gen_forecasts,
                                        f'ims/' + self.plot_folder + f'/plot.{epoch:03d}.png')
        """
        plt.figure()
        plt.plot(np.linspace(1, epochs, epochs), forecast_mse, label='Training loss generator')
        plt.legend()
        plt.show()
        """
        """
        with tf.Session() as sess:
            plt.figure()
            plt.plot(np.linspace(1, epochs, epochs), sess.run(self.diversity_loss.distribution_loss[1:]),
                     label='distribution_loss')
            plt.plot(np.linspace(1, epochs, epochs), sess.run(self.diversity_loss.diversity_loss[1:]),
                     label='diversity_loss')
            plt.plot(np.linspace(1, epochs, epochs), sess.run(self.diversity_loss.distance_loss[1:]),
                     label='distance_loss')
            plt.legend()
            plt.show()
        """

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1):
        half_batch = int(batch_size / 2)
        forecast_mse = np.zeros(epochs)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for d_epochs in range(self.discriminator_epochs):
                # Select a random half batch of images
                idx = np.random.randint(0, x.shape[0], half_batch)
                real_samples = x[idx]

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

            noise1 = self._generate_noise(batch_size)
            noise2 = self._generate_noise(batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise1, noise2], self._get_labels(batch_size=batch_size, real=True))

            # Measure forecast MSE of generator
            forecast_mse[epoch] = mean_squared_error(real_samples, gen_forecasts)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                  (epoch, d_loss[0], 100 * (d_loss[1]), g_loss, forecast_mse[epoch]))

            # kl_divergence[epoch] = self.kl_divergence(real_samples, gen_forecasts)
            # print("KL-divergence: ", kl_divergence[epoch])

            if epoch % self.plot_rate == 0:
                self.plot_distributions(real_samples, gen_forecasts,
                                        f'ims/' + self.plot_folder + f'/plot.{epoch:03d}.png')

        plt.figure()
        plt.plot(np.linspace(1, epochs, epochs), forecast_mse, label='Training loss generator')
        plt.legend()
        plt.show()

    def plot_distributions(self, real_samples, fake_samples, filename=None):
        # print('Plot directory: ', filename)
        sns.kdeplot(fake_samples.flatten(), color='red', alpha=0.6, label='GAN', shade=True)
        sns.kdeplot(real_samples.flatten(), color='blue', alpha=0.6, label='Real', shade=True)
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def monte_carlo_prediction(self, data, mc_forward_passes=500):

        # predict on batch
        noise = self._generate_noise(mc_forward_passes)
        prediction = self.generator.predict(noise)
        # self.plot_distributions(real_samples=data, fake_samples=prediction)
        return prediction

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
    data = generate_noise(5000)
    config = load_config_file('C:\\Users\\mathi\\PycharmProjects\\gan\\config\\config.yml')
    coverage_80_PI_1, coverage_95_PI_1 = [], []
    coverage_80_PI_2, coverage_95_PI_2 = [], []
    kl_div, uncertainty_list = [], []
    for i in range(5):
        gan = DSGAN(config['gan'])
        gan.build_model()
        gan.train(epochs=2000, batch_size=1024)
        predictions = gan.monte_carlo_prediction(generate_noise(5000), mc_forward_passes=5000)
        prediction_mean = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        print(gan.compute_kl_divergence(predictions, generate_noise(5000)))
        kl_div.append(gan.compute_kl_divergence(predictions, generate_noise(5000)))
        uncertainty_list.append(uncertainty)
        coverage_80_PI_1.append(compute_coverage(upper_limits=np.vstack([np.quantile(predictions, q=0.9, axis=0)]*10000),
                                                 lower_limits=np.vstack([np.quantile(predictions, q=0.1, axis=0)]*10000),
                                                 actual_values=generate_noise(10000)))
        coverage_95_PI_1.append(compute_coverage(upper_limits=np.vstack([np.quantile(predictions, q=0.975, axis=0)]*10000),
                                                 lower_limits=np.vstack([np.quantile(predictions, q=0.025, axis=0)]*10000),
                                                 actual_values=generate_noise(10000)))
        coverage_80_PI_2.append(compute_coverage(upper_limits=np.vstack([prediction_mean+1.28*uncertainty]*10000),
                                                 lower_limits=np.vstack([prediction_mean-1.28*uncertainty]*10000),
                                                 actual_values=generate_noise(10000)))
        coverage_95_PI_2.append(compute_coverage(upper_limits=np.vstack([prediction_mean+1.96*uncertainty]*10000),
                                                 lower_limits=np.vstack([prediction_mean-1.96*uncertainty]*10000),
                                                 actual_values=generate_noise(10000)))
    print('80% PI Coverage:', np.mean(coverage_80_PI_1), ', std:', np.std(coverage_80_PI_1))
    print('95% PI Coverage:', np.mean(coverage_95_PI_1), ', std:', np.std(coverage_95_PI_1))

    print('80% PI Coverage:', np.mean(coverage_80_PI_2), ', std:', np.std(coverage_80_PI_2))
    print('95% PI Coverage:', np.mean(coverage_95_PI_2), ', std:', np.std(coverage_95_PI_2))
    print('KL-divergence mean:', np.mean(kl_div), ', std:', np.std(kl_div))
    print('Uncertainty mean:', np.mean(uncertainty_list), ', std:', np.std(uncertainty_list))