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
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count={'CPU': 1})
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

from config.load_config import load_config_file
from models.get_model import get_gan
from utility.split_data import split_sequence
from data.generate_sine import generate_sine_data
from data.load_data import load_oslo_temperature, load_australia_temperature
from utility.compute_statistics import *
from time_series_pipeline import configure_model, load_data, plot_results


def compute_validation_error(model, x_val, y_val):
    # Compute inherent noise on validation set
    y_predicted = model.forecast(x_val)

    validation_mse = np.zeros(model.output_size)
    for i in range(model.output_size):
        validation_mse[i] = mean_squared_error(y_val[:, i], y_predicted[:, i])
    return validation_mse


def train_model(model, data, epochs, cfg, batch_size=128, verbose=1):
    # Split data in training and validation set
    train, val = data[:-int(len(data)*cfg['val_split'])], data[-int(model.window_size+len(data)*cfg['val_split']):]

    # Split training data into (x_t-l, ..., x_t), (x_t+1) pairs
    x_train, y_train = split_sequence(train, model.window_size, model.output_size)
    x_val, y_val = split_sequence(val, model.window_size, model.output_size)
    history = model.fit(x_train, y_train, x_val=x_val, y_val=y_val, epochs=epochs, batch_size=batch_size, verbose=verbose)

    validation_mse = compute_validation_error(model, x_val, y_val)

    return model, validation_mse, val


def test_model(model, data, validation_mse, cfg, naive_error, scaler, plot=True, file_name="/test_results.txt",
               min_max=None,  disable_pbar=False):
    forecast = model.monte_carlo_forecast(data, steps=int(len(data)-model.window_size), plot=plot,
                                          disable_pbar=disable_pbar)  # steps x horizon x mc_forward_passes
    forecast_mean = forecast.mean(axis=-1)
    forecast_std = forecast.std(axis=-1)
    forecast_var = forecast.var(axis=-1)

    total_uncertainty = np.sqrt(forecast_var + validation_mse)
    if plot:
        x_pred = np.linspace(model.window_size+1, len(data), len(data)-model.window_size)
        plt.figure()
        plt.plot(np.linspace(1, len(data), len(data)), data, label='Data')
        plt.plot(x_pred, forecast_mean[:, 0], label='Predictions')
        plt.fill_between(x_pred, forecast_mean[:, 0]-1.28*forecast_std[:, 0], forecast_mean[:, 0]+1.28*forecast_std[:, 0],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(x_pred, forecast_mean[:, 0]-1.96*forecast_std[:, 0], forecast_mean[:, 0]+1.96*forecast_std[:, 0],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.legend()
        plt.show()
    forecast_mse = sliding_window_mse(scaler.inverse_transform(forecast_mean),
                                      scaler.inverse_transform(data[model.window_size:]),
                                      model.forecasting_horizon)
    forecast_smape = sliding_window_smape(scaler.inverse_transform(forecast_mean),
                                          scaler.inverse_transform(data[model.window_size:]),
                                          model.forecasting_horizon)
    forecast_mase = sliding_window_mase(scaler.inverse_transform(forecast_mean),
                                        scaler.inverse_transform(data[model.window_size:]),
                                        model.forecasting_horizon, naive_error)
    msis_80 = mean_scaled_interval_score(y=scaler.inverse_transform(data[model.window_size:]),
                                         u=scaler.inverse_transform(np.quantile(forecast, q=0.9, axis=-1)),
                                         l=scaler.inverse_transform(np.quantile(forecast, q=0.1, axis=-1)),
                                         alpha=0.2,
                                         h=model.forecasting_horizon, naive_error=naive_error)
    msis_95 = mean_scaled_interval_score(y=scaler.inverse_transform(data[model.window_size:]),
                                         u=scaler.inverse_transform(np.quantile(forecast, q=0.975, axis=-1)),
                                         l=scaler.inverse_transform(np.quantile(forecast, q=0.025, axis=-1)),
                                         alpha=0.05,
                                         h=model.forecasting_horizon, naive_error=naive_error)

    coverage_80_1 = sliding_window_coverage(actual_values=data[model.window_size:],
                                            upper_limits=np.quantile(forecast, q=0.9, axis=-1),
                                            lower_limits=np.quantile(forecast, q=0.1, axis=-1),
                                            forecast_horizon=model.forecasting_horizon)
    coverage_95_1 = sliding_window_coverage(actual_values=data[model.window_size:],
                                            upper_limits=np.quantile(forecast, q=0.975, axis=-1),
                                            lower_limits=np.quantile(forecast, q=0.025, axis=-1),
                                            forecast_horizon=model.forecasting_horizon)
    coverage_80_2 = sliding_window_coverage(actual_values=data[model.window_size:],
                                            upper_limits=forecast_mean + 1.28*total_uncertainty,
                                            lower_limits=forecast_mean - 1.28*total_uncertainty,
                                            forecast_horizon=model.forecasting_horizon)
    coverage_95_2 = sliding_window_coverage(actual_values=data[model.window_size:],
                                            upper_limits=forecast_mean + 1.96 * total_uncertainty,
                                            lower_limits=forecast_mean - 1.96 * total_uncertainty,
                                            forecast_horizon=model.forecasting_horizon)
    width_80_1 = np.mean(scaler.inverse_transform(np.quantile(forecast, q=0.9, axis=-1))
                         - scaler.inverse_transform(np.quantile(forecast, q=0.1, axis=-1)), axis=0)
    width_95_1 = np.mean(scaler.inverse_transform(np.quantile(forecast, q=0.975, axis=-1))
                         - scaler.inverse_transform(np.quantile(forecast, q=0.025, axis=-1)), axis=0)
    width_80_2 = 2*1.28 * np.mean(total_uncertainty, axis=0)
    width_95_2 = 2*1.96 * np.mean(total_uncertainty, axis=0)

    file_path = cfg['results_path'] + file_name

    if cfg['model_name'].lower() == 'rnn':
        forecast_std = total_uncertainty.mean(axis=0)
        coverage_80 = coverage_80_2
        coverage_95 = coverage_95_2
        width_80 = width_80_2
        width_95 = width_95_2
        if min_max:
            forecast_std *= min_max
            width_80 *= min_max
            width_95 *= min_max
    else:
        forecast_std = forecast_std.mean(axis=0)
        coverage_80 = coverage_80_1
        coverage_95 = coverage_95_1
        width_80 = width_80_1
        width_95 = width_95_1
        if min_max:
            forecast_std *= min_max

    with open(file_path, "a") as f:
        f.write("mse,smape,mase,std,coverage_80,coverage_95,width_80,width_95,msis_80,msis_95\n")
        for (mse, smape, mase, std, c_80, c_95, w_80, w_95, m_80, m_95) in zip(forecast_mse, forecast_smape,
                                                                               forecast_mase, forecast_std, coverage_80,
                                                                               coverage_95, width_80, width_95, msis_80,
                                                                               msis_95):
            f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n".format(mse, smape, mase, std, c_80, c_95, w_80, w_95,
                                                                       m_80, m_95))
    try:
        file_name = file_name.replace('validation_results.txt', '')
    except:
        file_name = file_name.replace('test_results.txt', '')
    if cfg['model_name'].lower() == 'rnn':
        with open(cfg['results_path'] + file_name + "_forecast_one_step.csv", "w") as f:
            f.write("forecast,pred_int_80_low,pred_int_80_high,pred_int_95_low,pred_int_95_high\n")
            for (forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low, pred_int_95_up) in \
                    zip(scaler.inverse_transform(forecast_mean)[:, 0],
                        scaler.inverse_transform(forecast_mean - 1.28*total_uncertainty)[:, 0],
                        scaler.inverse_transform(forecast_mean + 1.28*total_uncertainty)[:, 0],
                        scaler.inverse_transform(forecast_mean - 1.96*total_uncertainty)[:, 0],
                        scaler.inverse_transform(forecast_mean + 1.96*total_uncertainty)[:, 0]):
                f.write(
                    "{0},{1},{2},{3},{4}\n".format(forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low,
                                                   pred_int_95_up))
        with open(cfg['results_path'] + file_name + "_forecast_horizon.csv", "w") as f:
            f.write("forecast,pred_int_80_low,pred_int_80_high,pred_int_95_low,pred_int_95_high\n")
            for (forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low, pred_int_95_up) in \
                    zip(scaler.inverse_transform(forecast_mean)[0, :],
                        scaler.inverse_transform(forecast_mean - 1.28*total_uncertainty)[0, :],
                        scaler.inverse_transform(forecast_mean + 1.28*total_uncertainty)[0, :],
                        scaler.inverse_transform(forecast_mean - 1.96*total_uncertainty)[0, :],
                        scaler.inverse_transform(forecast_mean + 1.96*total_uncertainty)[0, :]):
                f.write(
                    "{0},{1},{2},{3},{4}\n".format(forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low,
                                                   pred_int_95_up))
    else:
        with open(cfg['results_path'] + file_name + "_forecast_one_step.csv", "w") as f:
            f.write("forecast,pred_int_80_low,pred_int_80_high,pred_int_95_low,pred_int_95_high\n")
            for (forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low, pred_int_95_up) in \
                    zip(scaler.inverse_transform(forecast_mean)[:, 0],
                        scaler.inverse_transform(np.quantile(forecast, q=0.1, axis=-1))[:, 0],
                        scaler.inverse_transform(np.quantile(forecast, q=0.9, axis=-1))[:, 0],
                        scaler.inverse_transform(np.quantile(forecast, q=0.025, axis=-1))[:, 0],
                        scaler.inverse_transform(np.quantile(forecast, q=0.975, axis=-1))[:, 0]):
                f.write("{0},{1},{2},{3},{4}\n".format(forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low,
                                                       pred_int_95_up))
        with open(cfg['results_path'] + file_name + "_forecast_horizon.csv", "w") as f:
            f.write("forecast,pred_int_80_low,pred_int_80_high,pred_int_95_low,pred_int_95_high\n")
            for (forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low, pred_int_95_up) in \
                    zip(scaler.inverse_transform(forecast_mean)[0, :],
                        scaler.inverse_transform(np.quantile(forecast, q=0.1, axis=-1))[0, :],
                        scaler.inverse_transform(np.quantile(forecast, q=0.9, axis=-1))[0, :],
                        scaler.inverse_transform(np.quantile(forecast, q=0.025, axis=-1))[0, :],
                        scaler.inverse_transform(np.quantile(forecast, q=0.975, axis=-1))[0, :]):
                f.write("{0},{1},{2},{3},{4}\n".format(forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low,
                                                       pred_int_95_up))
    return forecast_mse, forecast_smape, forecast_mase, forecast_std, coverage_80, coverage_95, width_80, width_95, msis_80, msis_95


def time_series_validation_pipeline(cfg):
    forecast_mse_list, forecast_smape_list, forecast_mase_list = [], [], []
    validation_mse_list, forecast_std_list = [], []
    w_80_list, w_95_list = [], []
    c_80_list, c_95_list = [], []
    msis_80_list, msis_95_list = [], []
    for i in range(1):
        model = configure_model(cfg=cfg)
        train, test, scaler = load_data(cfg=cfg, window_size=model.window_size)
        naive_error = compute_naive_error(scaler.inverse_transform(train), seasonality=cfg['seasonality'],
                                          forecast_horizon=model.forecasting_horizon)
        start_time = time.time()
        trained_model, validation_mse, val = train_model(model=model, data=train, epochs=cfg['epochs'], cfg=cfg,
                                                         batch_size=cfg['batch_size'], verbose=0)
        training_time = time.time() - start_time
        test_model(model=trained_model, data=val, validation_mse=validation_mse, cfg=cfg, naive_error=naive_error,
                   scaler=scaler, plot=False, file_name="/validation_results.txt",
                   min_max=(np.max(scaler.inverse_transform(train))-np.min(scaler.inverse_transform(train)))
                            / (np.max(train)-np.min(train)),
                   disable_pbar=False)
        mse, smape, mase, std, c_80, c_95, w_80, w_95, msis_80, msis_95 = \
            test_model(model=trained_model, data=test,
                       validation_mse=validation_mse, cfg=cfg, naive_error=naive_error, scaler=scaler,
                       plot=False, file_name="/test_results.txt",
                       min_max=(np.max(scaler.inverse_transform(train)) - np.min(scaler.inverse_transform(train)))
                                / (np.max(train) - np.min(train)),
                       disable_pbar=False)
        msis_80_list.append(msis_80), msis_95_list.append(msis_95)
        forecast_mse_list.append(mse), forecast_smape_list.append(smape), forecast_mase_list.append(mase)
        validation_mse_list.append(validation_mse)
        forecast_std_list.append(std)
        c_80_list.append(c_80), c_95_list.append(c_95)
        w_80_list.append(w_80), w_95_list.append(w_95)

    print('========================================================'
          '\n================ Point Forecast Metrics ================'
          '\n========================================================')
    # print('Mean validation MSE:', np.mean(np.mean(validation_mse_list, axis=0)))
    print('Mean forecast MSE:', np.mean(np.mean(forecast_mse_list, axis=0)))
    # print('Forecast MSE:', np.mean(np.array(forecast_mse_list), axis=0))
    print('Mean forecast SMAPE:', np.mean(np.mean(forecast_smape_list, axis=0)))
    # print('Forecast SMAPE:', np.mean(np.array(forecast_smape_list), axis=0))
    print('Mean forecast MASE:', np.mean(np.mean(forecast_mase_list, axis=0)))
    print('Training time:', training_time)

    print('========================================================'
          '\n================== Model Uncertainty ==================='
          '\n========================================================')
    print('Estimated Standard deviation:', np.mean(np.mean(forecast_std_list, axis=0)))
    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(c_80_list, axis=0)),
          ', width:', np.mean(np.mean(np.array(w_80_list), axis=0)),
          # '\n Forecast horizon:', np.mean(np.array(c_80_list), axis=0)
          )
    print('80%-prediction interval MSIS:', np.mean(np.mean(msis_80_list, axis=0)))

    print('95%-prediction interval coverage - Mean:',  np.mean(np.mean(c_95_list, axis=0)),
          ', width:', np.mean(np.mean(np.array(w_95_list), axis=0)),
          # '\n Forecast horizon:', np.mean(np.array(c_95_list), axis=0)
          )
    print('95%-prediction interval MSIS:', np.mean(np.mean(msis_95_list, axis=0)))


if __name__ == '__main__':
    cfg = load_config_file('config\\config.yml')
    time_series_validation_pipeline(cfg)
