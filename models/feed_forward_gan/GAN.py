import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
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


class GAN:
    def __init__(self, cfg):
        self.plot_rate = cfg['plot_rate']
        self.plot_folder = 'feed_forward_GAN'
        self.noise_vector_size = cfg['noise_vector_size']  # Try larger vector
        self.noise_type = 'normal'  # uniform
        self.discriminator_epochs = cfg['discriminator_epochs']

        self.optimizer = Adam(lr=cfg['learning_rate'], beta_1=0.5)
        self.discriminator_loss = binary_crossentropy
        self.generator_loss = binary_crossentropy

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
        # self.combined.summary()
        self.combined.compile(loss=self.generator_loss, optimizer=self.optimizer)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        noise_inp = Input(shape=noise_shape)

        x = Dense(16)(noise_inp)
        x = ReLU()(x)
        # x = Dropout(0.1)(x)
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
        # x = Dropout(0.1)(x)
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

            noise = self._generate_noise(batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise,
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

    def compute_probs(self, data, n=10):
        h, e = np.histogram(data, n)
        p = h / data.shape[0]
        return e, p

    def support_intersection(self, p, q):
        return list(filter(lambda x: (x[0] != 0) & (x[1] != 0), list(zip(p, q))))

    def get_probs(self, list_of_tuples):
        p = np.array([p[0] for p in list_of_tuples])
        q = np.array([p[1] for p in list_of_tuples])
        return p, q

    def kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))

    def js_divergence(self, p, q):
        m = (1. / 2.) * (p + q)
        return (1. / 2.) * self.kl_divergence(p, m) + (1. / 2.) * self.kl_divergence(q, m)

    def compute_kl_divergence(self, train_sample, test_sample, n_bins=100):
        """
        Computes the KL Divergence using the support intersection between two different samples
        """
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)
        list_of_tuples = self.support_intersection(p, q)
        p, q = self.get_probs(list_of_tuples)
        return self.kl_divergence(p, q)

    def compute_js_divergence(self, train_sample, test_sample, n_bins=100):
        """
        Computes the JS Divergence using the support intersection between two different samples
        """
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)

        list_of_tuples = self.support_intersection(p, q)
        p, q = self.get_probs(list_of_tuples)

        return self.js_divergence(p, q)


if __name__ == '__main__':
    config = load_config_file('C:\\Users\\mathi\\PycharmProjects\\gan\\config\\config.yml')
    coverage_80_PI_1, coverage_95_PI_1 = [], []
    coverage_80_PI_2, coverage_95_PI_2 = [], []
    kl_div, js_div, uncertainty_list = [], [], []
    for i in range(10):
        gan = GAN(config['gan'])
        gan.build_model()
        gan.train(epochs=2000, batch_size=32)
        predictions = gan.monte_carlo_prediction(generate_noise(5000), mc_forward_passes=5000)
        prediction_mean = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        kl_div.append(gan.compute_kl_divergence(predictions, generate_noise(5000)))
        js_div.append(gan.compute_js_divergence(predictions, generate_noise(5000)))
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
    print('JS-divergence mean:', np.mean(js_div), ', std:', np.std(js_div))
    print('Uncertainty mean:', np.mean(uncertainty_list), ', std:', np.std(uncertainty_list))

