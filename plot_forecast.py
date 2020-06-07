import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import pandas as pd
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

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from keras.models import load_model

from config.load_config import load_config_file
from models.get_model import get_gan
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from data.load_data import load_oslo_temperature, load_australia_temperature
from utility.compute_statistics import *
from time_series_pipeline import configure_model, load_data, plot_results
from time_series_pipeline_with_validation import test_model, compute_validation_error
from time_series_pipeline_baseline import train_model


def read_files(file_path, file_name):
    df = pd.read_csv(file_path.lower() + file_name, header=0)
    return df


def plot_figures(data, test, window_size, columnName, data_set='oslo'):
    x_pred = np.linspace(window_size + 1, len(test), len(data[0]['forecast']))
    plt.figure()
    plt.plot(np.linspace(1, len(test), len(test)), test)
    plt.plot(x_pred, data[0]['forecast'], label='ARIMA')
    plt.plot(x_pred, data[1]['forecast'], label='ETS')
    plt.plot(x_pred, data[2]['forecast'], label='MC dropout')
    plt.plot(x_pred, data[3]['forecast'], label='ForGAN')
    plt.title(columnName[1] + ' ' + columnName[2] + ' Time Series Forecasting')
    plt.legend()
    plt.savefig('plots/' + data_set + '/' + columnName[1] + '_' + columnName[2] + '_forecasting_horizon.png')
    #plt.show()
    plt.close()

    for i in range(len(data)):
        plt.figure()
        plt.plot(np.linspace(1, len(test), len(test)), test),
        plt.plot(x_pred, data[i]['forecast'], label=model_names[i])
        plt.fill_between(x_pred, data[i]['pred_int_80_low'], data[i]['pred_int_80_high'],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(x_pred,
                         data[i]['pred_int_95_low'],
                         data[i]['pred_int_95_high'],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.title(columnName[1] + ' ' + columnName[2] + ' Time Series Forecasting')
        plt.legend(loc=2)
        plt.savefig('plots/' + data_set + '/' + columnName[1] + '_' + columnName[2] + '_' + model_names[i] +
                    '_forecasting_horizon.png')
        #plt.show()
        plt.close()


def pipeline(model_paths):
    cfg = load_config_file(model_paths[3]+'config.yml')
    train, test, scaler = load_data(cfg=cfg, window_size=cfg['window_size'])

    arima_data = read_files(model_paths[0], 'test_results.txt_forecast_horizon.csv')
    es_data = read_files(model_paths[1], 'test_results.txt_forecast_horizon.csv')
    mc_data = read_files(model_paths[2], 'test_results.txt_forecast_horizon.csv')
    gan_data = read_files(model_paths[3], 'test_results_generator_5000.h5.txt_forecast_horizon.csv')
    data = [arima_data, es_data, mc_data, gan_data]
    plot_figures(data, scaler.inverse_transform(test[:cfg['window_size'] + len(data[0]['forecast'])]),
                 cfg['window_size'], None, data_set='oslo')


def avocado_pipeline(model_paths):
    cfg = load_config_file(model_paths[3]+'config.yml')
    train_df, test_df, scalers = load_data(cfg=cfg, window_size=cfg['window_size'])
    i = 0
    for columnName, columnData in test_df.iteritems():
        test = columnData.values.reshape(-1, 1)
        scaler = scalers[i]
        arima_data = read_files(model_paths[0], columnName[1] + "_" + columnName[2] + '_forecast_horizon.csv')
        es_data = read_files(model_paths[1],  columnName[1] + "_" + columnName[2] + '_forecast_horizon.csv')
        mc_data = read_files(model_paths[2],  columnName[1] + "_" + columnName[2] + '__forecast_horizon.csv')
        gan_data = read_files(model_paths[3], 'generator_30000.h5_' + columnName[1] + "_" + columnName[2] + '_test_results.txt_forecast_horizon.csv')
        data = [arima_data, es_data, mc_data, gan_data]
        plot_figures(data, scaler.inverse_transform(test[:cfg['window_size']+len(data[0]['forecast'])]),
                     cfg['window_size'], columnName, data_set='avocado')
        i += 1


if __name__ == '__main__':
    model_path = ['results/avocado/arima/',
                  'results/avocado/es/',
                  'results/avocado/rnn/minmax/rnn_epochs_40_D_epochs_5_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000500/',
                  'results/avocado/recurrentgan/minmax/rnn_epochs_30000_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000100/']
    model_names = ['ARIMA', 'ETS', 'MC Dropout', 'ForGAN']
    avocado_pipeline(model_path)
