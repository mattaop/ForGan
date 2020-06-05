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


def train_model(model, data, cfg):
    # Split data in training and validation set
    train, val = data[:-int(len(data)*cfg['val_split'])], data[-int(model.window_size+len(data)*cfg['val_split']):]

    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(train, model.window_size, model.output_size)
    x_val, y_val = split_sequence(val, model.window_size, model.output_size)

    validation_mse = compute_validation_error(model, x_val, y_val)

    return model, validation_mse, val


def pipeline(model_path, model_name):
    cfg = load_config_file(model_path+'config.yml')
    model = configure_model(cfg=cfg)
    model.generator = load_model(model_path+model_name)
    train_df, test_df, scalers = load_data(cfg=cfg, window_size=model.window_size)
    print(train_df.columns)
    if cfg['data_source'] == 'avocado':
        for i in range(len(train_df.columns.values)):
            column = train_df.columns.values[i]
            train = train_df[column].values.reshape(-1, 1)
            test = test_df[column].values.reshape(-1, 1)
            scaler = scalers[i]
            trained_model, validation_mse, val = train_model(model=model, data=train, cfg=cfg)
            # trained_model.forecasting_horizon = len(test)-trained_model.window_size
            forecast = trained_model.recurrent_forecast(np.expand_dims(test[:model.window_size], 0))
            print(forecast.shape)
            forecast_mean = np.mean(forecast, axis=-1)
            forecast_var = np.std(forecast, axis=-1)
            x_pred = np.linspace(model.window_size + 1, len(forecast_mean) + model.window_size, len(forecast_mean))

            plt.figure()
            plt.plot(np.linspace(1, trained_model.window_size + trained_model.forecasting_horizon,
                                 trained_model.window_size + trained_model.forecasting_horizon),
                     scaler.inverse_transform(test[:trained_model.window_size + trained_model.forecasting_horizon]),
                     label='Data')
            plt.plot(x_pred,
                     scaler.inverse_transform(forecast_mean.reshape(-1, 1)), label='ForGAN')
            plt.fill_between(x_pred,
                             scaler.inverse_transform(np.quantile(forecast, q=0.1, axis=-1).reshape(-1, 1))[:, 0],
                             scaler.inverse_transform(np.quantile(forecast, q=0.9, axis=-1).reshape(-1, 1))[:, 0],
                             alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
            plt.fill_between(x_pred,
                             scaler.inverse_transform(np.quantile(forecast, q=0.025, axis=-1).reshape(-1, 1))[:, 0],
                             scaler.inverse_transform(np.quantile(forecast, q=0.975, axis=-1).reshape(-1, 1))[:, 0],
                             alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
            plt.title('ForGAN Time Series Forecasting')
            plt.legend()
            plt.savefig('plots/sine/forecasting_forgan.png')
            plt.show()

    else:
        trained_model, validation_mse, val = train_model(model=model, data=train, cfg=cfg)
        # trained_model.forecasting_horizon = len(test)-trained_model.window_size
        forecast = trained_model.recurrent_forecast(np.expand_dims(test[:model.window_size], 0))
        print(forecast.shape)
        forecast_mean = np.mean(forecast, axis=-1)
        forecast_var = np.std(forecast, axis=-1)
        x_pred = np.linspace(model.window_size + 1, len(forecast_mean)+model.window_size, len(forecast_mean))
        plt.figure()
        plt.plot(np.linspace(1, trained_model.window_size+trained_model.forecasting_horizon,
                             trained_model.window_size+trained_model.forecasting_horizon),
                 scaler.inverse_transform(test[:trained_model.window_size+trained_model.forecasting_horizon]), label='Data')
        plt.plot(x_pred,
                 scaler.inverse_transform(forecast_mean.reshape(-1, 1)), label='ForGAN')
        plt.fill_between(x_pred,
                         scaler.inverse_transform(np.quantile(forecast, q=0.1, axis=-1).reshape(-1, 1))[:, 0],
                         scaler.inverse_transform(np.quantile(forecast, q=0.9, axis=-1).reshape(-1, 1))[:, 0],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(x_pred,
                         scaler.inverse_transform(np.quantile(forecast, q=0.025, axis=-1).reshape(-1, 1))[:, 0],
                         scaler.inverse_transform(np.quantile(forecast, q=0.975, axis=-1).reshape(-1, 1))[:, 0],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.title('ForGAN Time Series Forecasting')
        plt.legend()
        plt.savefig('plots/sine/forecasting_forgan.png')
        plt.show()


if __name__ == '__main__':
    model_path = 'results/avocado/recurrentgan/minmax/rnn_epochs_30000_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000100/'
    model_name = 'generator_30000.h5'
    pipeline(model_path, model_name)
