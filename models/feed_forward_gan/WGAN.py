import numpy as np
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop
# from keras_radam import RAdam
from keras import backend
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from models.feed_forward_gan.GAN import GAN
from utility.ClipConstraint import ClipConstraint
from config.load_config import load_config_file
from data.generate_noise import generate_noise
from utility.compute_statistics import compute_coverage


class WGAN(GAN):
    def __init__(self, cfg):
        GAN.__init__(self, cfg)
        self.plot_rate = 100
        self.plot_folder = 'feed_forward_WGAN'
        self.noise_vector_size = 10  # Try larger vector

        self.optimizer = RMSprop(lr=cfg['learning_rate'])
        self.loss_function = self.wasserstein_loss
        self.discriminator_loss = self.wasserstein_loss
        self.generator_loss = self.wasserstein_loss

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def build_discriminator(self):
        future_shape = (1,)
        future_inp = Input(shape=future_shape)

        const = ClipConstraint(0.1)

        x = Dense(64, kernel_constraint=const)(future_inp)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(64, kernel_constraint=const)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = Dropout(0.4)(x)
        validity = Dense(1)(x)

        model = Model(inputs=future_inp, outputs=validity)
        model.summary()

        return model

    def _get_labels(self, batch_size, real=True):
        if real:
            return np.ones((batch_size, 1))
        else:
            return -np.ones((batch_size, 1))


if __name__ == '__main__':
    data = generate_noise(5000)
    config = load_config_file('C:\\Users\\mathi\\PycharmProjects\\gan\\config\\config.yml')
    coverage_80_PI_1, coverage_95_PI_1 = [], []
    coverage_80_PI_2, coverage_95_PI_2 = [], []
    kl_div, uncertainty_list = [], []

    for i in range(10):
        gan = WGAN(config['gan'])
        gan.build_model()
        gan.train(epochs=2000, batch_size=1024)
        predictions = gan.monte_carlo_prediction(generate_noise(5000), mc_forward_passes=5000)
        prediction_mean = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        kl_div.append(gan.compute_kl_divergence(predictions, generate_noise(5000)))
        uncertainty_list.append(uncertainty)
        coverage_80_PI_1.append(
            compute_coverage(upper_limits=np.vstack([np.quantile(predictions, q=0.9, axis=0)] * 10000),
                             lower_limits=np.vstack([np.quantile(predictions, q=0.1, axis=0)] * 10000),
                             actual_values=generate_noise(10000)))
        coverage_95_PI_1.append(
            compute_coverage(upper_limits=np.vstack([np.quantile(predictions, q=0.975, axis=0)] * 10000),
                             lower_limits=np.vstack([np.quantile(predictions, q=0.025, axis=0)] * 10000),
                             actual_values=generate_noise(10000)))
        coverage_80_PI_2.append(compute_coverage(upper_limits=np.vstack([prediction_mean + 1.28 * uncertainty] * 10000),
                                                 lower_limits=np.vstack([prediction_mean - 1.28 * uncertainty] * 10000),
                                                 actual_values=generate_noise(10000)))
        coverage_95_PI_2.append(compute_coverage(upper_limits=np.vstack([prediction_mean + 1.96 * uncertainty] * 10000),
                                                 lower_limits=np.vstack([prediction_mean - 1.96 * uncertainty] * 10000),
                                                 actual_values=generate_noise(10000)))
    print('80% PI Coverage:', np.mean(coverage_80_PI_1), ', std:', np.std(coverage_80_PI_1))
    print('95% PI Coverage:', np.mean(coverage_95_PI_1), ', std:', np.std(coverage_95_PI_1))

    print('80% PI Coverage:', np.mean(coverage_80_PI_2), ', std:', np.std(coverage_80_PI_2))
    print('95% PI Coverage:', np.mean(coverage_95_PI_2), ', std:', np.std(coverage_95_PI_2))
    print('KL-divergence mean:', np.mean(kl_div), ', std:', np.std(kl_div))
    print('Uncertainty mean:', np.mean(uncertainty_list), ', std:', np.std(uncertainty_list))
