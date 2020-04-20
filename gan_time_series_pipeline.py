import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import random as rn
import tensorflow as tf
print(tf.__version__)
seed = 1
rn.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from keras import backend as k
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count={'CPU': 1})
sess = tf.Session(graph=tf.get_default_graph(), config=config)
k.set_session(sess)

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.load_config import load_config_file
from models.get_model import get_GAN
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from utility.compute_coverage import print_coverage, compute_coverage


def configure_model(model_name):
    gan = get_GAN(model_name)
    gan.build_gan()

    paths = ['ims',
             'ims/' + gan.plot_folder
             ]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    return gan


def load_data(cfg, window_size):
    if cfg['data_source'].lower() == 'sine':
        data = generate_sine_data(num_points=500)
    else:
        return None
    train = data[:-int(len(data)*cfg['test_split'])]
    test = data[-int(len(data)*cfg['test_split']+window_size):]
    train, test = scale_data(train, test)
    return train, test


def scale_data(train, test):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test


def train_gan(gan, data, epochs, batch_size=128, discriminator_epochs=1):
    # Load the data
    x_train, y_train = split_sequence(data, gan.window_size, gan.forecasting_horizon)

    half_batch = int(batch_size / 2)
    forecast_mse = np.zeros(epochs)
    G_loss = np.zeros(epochs)
    D_loss = np.zeros(epochs)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for d_epochs in range(discriminator_epochs):
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            historic_time_series = x_train[idx]
            future_time_series = y_train[idx]

            noise = gan._generate_noise(half_batch)  # Normalisere til 1

            # Generate a half batch of new images
            gen_forecasts = gan.generator.predict([historic_time_series, noise])

            # Train the discriminator
            d_loss_real = gan.discriminator.train_on_batch([historic_time_series, future_time_series],
                                                           gan._get_labels(batch_size=half_batch, real=True))
            d_loss_fake = gan.discriminator.train_on_batch([historic_time_series,
                                                             tf.keras.backend.expand_dims(gen_forecasts, axis=-1)],
                                                           gan._get_labels(batch_size=half_batch, real=False))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = gan._generate_noise(batch_size)

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid_y = np.array([1] * batch_size)

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        historic_time_series = x_train[idx]
        # Train the generator
        g_loss = gan.combined.train_on_batch([historic_time_series, noise], valid_y)

        # Measure forecast MSE of generator
        forecast_mse[epoch] = mean_squared_error(future_time_series[:, :, 0], gen_forecasts)
        # kl_divergence[epoch] = sum(self.kl_divergence(future_time_series[:, i, 0], gen_forecasts[:, i])
        #                           for i in range(self.forecasting_horizon))/self.forecasting_horizon
        G_loss[epoch] = g_loss
        D_loss[epoch] = d_loss[0]
        # Plot the progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
              (epoch, d_loss[0], 100 * d_loss[1], g_loss, forecast_mse[epoch]))
        # print("KL-divergence: ", kl_divergence[epoch])

        if epoch % gan.plot_rate == 0:
            gan.plot_distributions(future_time_series[:, :, 0], gen_forecasts,
                                   f'ims/' + gan.plot_folder + f'/epoch{epoch:03d}.png')

    plt.figure()
    plt.plot(np.linspace(1, epochs, epochs), forecast_mse, label='Forecast error generator')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(np.linspace(1, epochs, epochs), G_loss, label='Generator loss')
    plt.plot(np.linspace(1, epochs, epochs), D_loss, label='Discriminator loss')
    plt.legend()
    plt.show()

    return gan


def test_model(gan, data, mc_forward_passes=500):
    forecast = gan.monte_carlo_forecast(data, int(len(data)-gan.window_size), mc_forward_passes)  # steps x horizon x mc_forward_passes
    forecast_mean = forecast.mean(axis=-1)
    forecast_std = forecast.std(axis=-1)

    x_pred = np.linspace(gan.window_size+1, len(data), len(data)-gan.window_size)
    plt.figure()
    plt.plot(np.linspace(1, len(data), len(data)), data, label='Data')
    plt.plot(x_pred, forecast_mean[:, 0], label='Predictions')
    plt.fill_between(x_pred, forecast_mean[:, 0]-1.28*forecast_std[:, 0], forecast_mean[:, 0]+1.28*forecast_std[:, 0],
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
    plt.fill_between(x_pred, forecast_mean[:, 0]-1.96*forecast_std[:, 0], forecast_mean[:, 0]+1.96*forecast_std[:, 0],
                     alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
    plt.legend()
    plt.show()
    print('Forecast error:', mean_squared_error(data[gan.window_size:], forecast_mean[:, 0]))
    print('Mean forecast standard deviation:', forecast_std.mean(axis=0))
    print('80%-prediction interval coverage:', compute_coverage(actual_values=data[gan.window_size:],
                                                                upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                                                lower_limits=np.quantile(forecast, q=0.1, axis=-1)))
    print('95%-prediction interval coverage:', compute_coverage(actual_values=data[gan.window_size:],
                                                                upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                                                lower_limits=np.quantile(forecast, q=0.025, axis=-1)))
    print_coverage(mean=forecast_mean[:, 0], uncertainty=forecast_std[:, 0], actual_values=data[gan.window_size:])


def pipeline():
    cfg = load_config_file('config\\config.yml')
    gan = configure_model(model_name=cfg['gan']['model_name'])
    train, test = load_data(cfg=cfg['data'], window_size=gan.window_size)
    trained_gan = train_gan(gan=gan, data=train, epochs=800, batch_size=256, discriminator_epochs=2)
    test_model(gan=trained_gan, data=test, mc_forward_passes=500)


if __name__ == '__main__':
    pipeline()
