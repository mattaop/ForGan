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
seed = 4
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
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.load_config import load_config_file
from models.get_model import get_gan
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from data.load_data import load_oslo_temperature, load_australia_temperature
from utility.compute_statistics import *


def configure_model(cfg):
    gan = get_gan(cfg)
    gan.build_model()

    paths = ['ims',
             'ims/' + gan.plot_folder
             ]
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)
    return gan


def load_data(cfg, window_size):
    if cfg['data_source'].lower() == 'sine':
        data = generate_sine_data(num_points=1000)
    elif cfg['data_source'].lower() == 'oslo':
        data = load_oslo_temperature()
    elif cfg['data_source'].lower() == 'australia':
        data = load_australia_temperature()
    else:
        return None
    print('Data shape', data.shape)
    train = data[:-int(len(data) * cfg['test_split'])]
    test = data[-int(len(data) * cfg['test_split'] + window_size):]
    train, test = scale_data(train, test)
    return train, test


def scale_data(train, test):
    scaler = MinMaxScaler(feature_range=(10 ** (-10), 1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test


def plot_results(y, label, y2=None, y2_label=None, title='', y_label='Value'):
    x = np.linspace(1, len(y), len(y))
    plt.figure()
    plt.title(title)
    plt.plot(x, y, label=label)
    if y2 is not None:
        plt.plot(x, y2, label=y2_label)
    plt.ylabel(y_label)
    plt.xlabel('Forecast horizon')
    plt.legend()
    plt.show()


def train_gan(gan, data, epochs, batch_size=128, verbose=1):
    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(data, gan.window_size, gan.output_size)

    history = gan.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return gan


def test_model(model, data, plot=True):
    forecast = model.monte_carlo_forecast(data, steps=int(len(data) - model.window_size), plot=plot)  # steps x horizon x mc_forward_passes
    forecast_mse = sliding_window_mse(forecast, data[model.window_size:], model.forecasting_horizon)

    forecast_smape = sliding_window_smape(forecast, data[model.window_size:], model.forecasting_horizon)

    coverage_80 = sliding_window_coverage(actual_values=data[model.window_size:],
                                          upper_limits=model.pred_int_80[:, :, 1],
                                          lower_limits=model.pred_int_80[:, :, 0],
                                          forecast_horizon=model.forecasting_horizon)
    coverage_95 = sliding_window_coverage(actual_values=data[model.window_size:],
                                          upper_limits=model.pred_int_95[:, :, 1],
                                          lower_limits=model.pred_int_95[:, :, 0],
                                          forecast_horizon=model.forecasting_horizon)

    width_80 = model.pred_int_80[:, :, 1] - model.pred_int_80[:, :, 0]
    width_95 = model.pred_int_95[:, :, 1] - model.pred_int_95[:, :, 0]

    return forecast_mse, forecast_smape, coverage_80, coverage_95, width_80, width_95


def pipeline():
    cfg = load_config_file('config\\config.yml')
    forecast_mse_list, forecast_smape_list = [], []
    width_80_list, width_95_list = [], []
    coverage_80_list, coverage_95_list = [], []
    for i in range(1):
        gan = configure_model(cfg=cfg['gan'])
        train, test = load_data(cfg=cfg['data'], window_size=gan.window_size)
        trained_model = train_gan(gan=gan, data=train, epochs=cfg['gan']['epochs'],
                                  batch_size=cfg['gan']['batch_size'], verbose=1)
        forecast_mse, forecast_smape, coverage_80, coverage_95, width_80,  width_95 = \
            test_model(model=trained_model, data=test, plot=False)
        forecast_mse_list.append(forecast_mse), forecast_smape_list.append(forecast_smape)
        coverage_80_list.append(coverage_80), coverage_95_list.append(coverage_95)
        width_80_list.append(width_80), width_95_list.append(width_95)

    print('========================================================'
          '\n================ Point Forecast Metrics ================'
          '\n========================================================')
    print('Mean forecast MSE:', np.mean(np.mean(forecast_mse_list, axis=0)))
    print('Forecast MSE:', np.mean(np.array(forecast_mse_list), axis=0))
    print('Mean forecast SMAPE:', np.mean(np.mean(forecast_smape_list, axis=0)))
    print('Forecast SMAPE:', np.mean(np.array(forecast_smape_list), axis=0))

    print('========================================================'
          '\n================== Model Uncertainty ==================='
          '\n========================================================')
    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(coverage_80_list, axis=0)),
          ', width:', np.mean(width_80_list),
          '\n Forecast horizon:', np.mean(np.array(coverage_80_list), axis=0))
    print('95%-prediction interval coverage - Mean:',  np.mean(np.mean(coverage_95_list, axis=0)),
          ', width:', np.mean(width_95_list),
          '\n Forecast horizon:', np.mean(np.array(coverage_95_list), axis=0))

    if cfg['gan']['model'].lower() in ['arima', 'es']:
        file_name = ("test_results/data_" + cfg['data']['data_source'].lower() + "_model_" + cfg['gan']['model'].lower())
    else:
        file_name = ("test_results/data_" + cfg['data']['data_source'] .lower() + "_model_" + cfg['gan']['model'].lower() +
                     "_epochs_%d_D_epochs_%d_batch_size_%d_noise_vec_%d_learning_rate_%f.txt" %
                     (cfg['gan']['epochs'], cfg['gan']['discriminator_epochs'], cfg['gan']['batch_size'],
                      cfg['gan']['noise_vector_size'], cfg['gan']['learning_rate']))
    mse = np.mean(np.array(forecast_mse_list), axis=0)
    smap = np.mean(np.array(forecast_smape_list), axis=0)
    c_80 = np.mean(np.array(coverage_80_list), axis=0)
    c_95 = np.mean(np.array(coverage_95_list), axis=0)
    with open(file_name, "w") as f:
        f.write("mse,smape,coverage_80,coverage_95\n")
        for (mse, smap, c_80, c_95) in zip(mse, smap, c_80, c_95):
            f.write("{0},{1},{2},{3}\n".format(mse, smap, c_80, c_95))


if __name__ == '__main__':
    pipeline()
