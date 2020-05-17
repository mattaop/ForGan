import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm

from config.load_config import load_config_file
from data.generate_noise import generate_noise
from utility.compute_statistics import compute_coverage
from models.recurrent_gan.RecurrentGAN import RecurrentGAN


class RecurrentConvGAN(RecurrentGAN):
    def __init__(self, cfg):
        RecurrentGAN.__init__(self, cfg)
        self.plot_folder = 'RecurrentConvGAN'

    def build_discriminator(self):
        historic_shape = (self.window_size, 1)
        future_shape = (self.output_size, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        x = Concatenate(axis=1)([historic_inp, future_inp])

        x = Conv1D(16, kernel_size=4)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        # x = Conv1D(16, kernel_size=4)(x)
        # x = LeakyReLU(alpha=0.1)(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        # x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(32)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.4)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model


if __name__ == '__main__':
    config = load_config_file('C:\\Users\\mathi\\PycharmProjects\\gan\\config\\config.yml')
    coverage_80_PI_1, coverage_95_PI_1 = [], []
    coverage_80_PI_2, coverage_95_PI_2 = [], []
    kl_div, js_div, uncertainty_list = [], [], []
    for i in range(1):
        gan = RecurrentConvGAN(config['gan'])
        gan.build_model()
        gan.train(epochs=200, batch_size=32)
        predictions = gan.monte_carlo_forecast(generate_noise(5000))[0, 0]
        prediction_mean = predictions.mean(axis=0)
        print(predictions.shape)
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

