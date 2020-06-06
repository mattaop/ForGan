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


from config.load_config import load_config_file
from time_series_pipeline import configure_model, load_data, train_model
from utility.compute_statistics import *


def test_model(model, data, naive_error, cfg, scaler, plot=True, file_name="/test_results.txt", min_max=None,
               disable_pbar=False):
    forecast = model.monte_carlo_forecast(data, steps=int(len(data) - model.window_size), plot=plot,
                                          disable_pbar=disable_pbar)  # steps x horizon x mc_forward_passes
    print(forecast.shape)
    forecast_mse = sliding_window_mse(scaler.inverse_transform(forecast),
                                      scaler.inverse_transform(data[model.window_size:]),
                                      model.forecasting_horizon)
    forecast_std = np.sqrt(model.variance)

    forecast_smape = sliding_window_smape(scaler.inverse_transform(forecast),
                                          scaler.inverse_transform(data[model.window_size:]),
                                          model.forecasting_horizon)
    forecast_mase = sliding_window_mase(scaler.inverse_transform(forecast),
                                        scaler.inverse_transform(data[model.window_size:]),
                                        model.forecasting_horizon,
                                        naive_error)

    if plot:
        x_pred = np.linspace(model.window_size+1, len(data), len(data)-model.window_size)
        plt.figure()
        plt.plot(np.linspace(1, len(data), len(data)), data, label='Data')
        plt.plot(x_pred, forecast[:, 0], label='Predictions')
        plt.fill_between(x_pred, model.pred_int_80[:, 0, 0], model.pred_int_80[:, 0, 1],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='80%-PI')
        plt.fill_between(x_pred, model.pred_int_95[:, 0, 1], model.pred_int_95[:, 0, 1],
                         alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848', label='95%-PI')
        plt.legend()
        plt.show()
    msis_80 = mean_scaled_interval_score(y=scaler.inverse_transform(data[model.window_size:]),
                                         u=scaler.inverse_transform(model.pred_int_80[:, :, 1]),
                                         l=scaler.inverse_transform(model.pred_int_80[:, :, 0]), alpha=0.2,
                                         h=model.forecasting_horizon, naive_error=naive_error)
    msis_95 = mean_scaled_interval_score(y=scaler.inverse_transform(data[model.window_size:]),
                                         u=scaler.inverse_transform(model.pred_int_95[:, :, 1]),
                                         l=scaler.inverse_transform(model.pred_int_95[:, :, 0]), alpha=0.05,
                                         h=model.forecasting_horizon, naive_error=naive_error)
    coverage_80 = sliding_window_coverage(actual_values=scaler.inverse_transform(data[model.window_size:]),
                                          upper_limits=scaler.inverse_transform(model.pred_int_80[:, :, 1]),
                                          lower_limits=scaler.inverse_transform(model.pred_int_80[:, :, 0]),
                                          forecast_horizon=model.forecasting_horizon)
    coverage_95 = sliding_window_coverage(actual_values=data[model.window_size:],
                                          upper_limits=model.pred_int_95[:, :, 1],
                                          lower_limits=model.pred_int_95[:, :, 0],
                                          forecast_horizon=model.forecasting_horizon)

    width_80 = np.mean(scaler.inverse_transform(model.pred_int_80[:, :, 1])
                       - scaler.inverse_transform(model.pred_int_80[:, :, 0]), axis=0)
    width_95 = np.mean(scaler.inverse_transform(model.pred_int_95[:, :, 1])
                       - scaler.inverse_transform(model.pred_int_95[:, :, 0]), axis=0)
    if min_max:
        forecast_std = forecast_std * min_max
    file_path = cfg['results_path'] + file_name

    with open(file_path, "a") as f:
        f.write("mse,smape,mase,std,coverage_80,coverage_95,width_80,width_95,msis_80,msis_95\n")
        for (mse, smape, mase, std, c_80, c_95, w_80, w_95, m_80, m_95) in zip(forecast_mse, forecast_smape,
                                                                               forecast_mase, forecast_std, coverage_80,
                                                                               coverage_95, width_80, width_95, msis_80,
                                                                               msis_95):
            f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n".format(mse, smape, mase, std, c_80, c_95, w_80, w_95,
                                                                       m_80, m_95))
    file_name = file_name.replace('_test_results.txt', '')

    with open(cfg['results_path'] + file_name + "_forecast_one_step.csv", "w") as f:
        f.write("forecast,pred_int_80_low,pred_int_80_high,pred_int_95_low,pred_int_95_high\n")
        for (forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low, pred_int_95_up) in \
                zip(scaler.inverse_transform(forecast)[:, 0],
                    scaler.inverse_transform(model.pred_int_80[:, :, 0])[:, 0],
                    scaler.inverse_transform(model.pred_int_80[:, :, 1])[:, 0],
                    scaler.inverse_transform(model.pred_int_95[:, :, 0])[:, 0],
                    scaler.inverse_transform(model.pred_int_95[:, :, 1])[:, 0]):
            f.write("{0},{1},{2},{3},{4}\n".format(forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low,
                                                   pred_int_95_up))
    with open(cfg['results_path'] + file_name + "_forecast_horizon.csv", "w") as f:
        f.write("forecast,pred_int_80_low,pred_int_80_high,pred_int_95_low,pred_int_95_high\n")
        for (forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low, pred_int_95_up) in \
                zip(scaler.inverse_transform(forecast)[0, :],
                    scaler.inverse_transform(model.pred_int_80[:, :, 0])[0, :],
                    scaler.inverse_transform(model.pred_int_80[:, :, 1])[0, :],
                    scaler.inverse_transform(model.pred_int_95[:, :, 0])[0, :],
                    scaler.inverse_transform(model.pred_int_95[:, :, 1])[0, :]):
            f.write("{0},{1},{2},{3},{4}\n".format(forecast_value, pred_int_80_low, pred_int_80_high, pred_int_95_low,
                                                   pred_int_95_up))

    return forecast_mse, forecast_smape, forecast_mase, forecast_std, coverage_80, coverage_95, width_80, width_95, msis_80, msis_95


def time_series_baseline_pipeline(cfg):
    forecast_mse_list, forecast_smape_list, forecast_mase_list = [], [], []
    width_80_list, width_95_list = [], []
    coverage_80_list, coverage_95_list = [], []
    msis_80_list, msis_95_list = [], []
    for i in range(1):
        model = configure_model(cfg=cfg)
        train, test, scaler = load_data(cfg=cfg, window_size=model.window_size)
        naive_error = compute_naive_error(scaler.inverse_transform(train), seasonality=cfg['seasonality'],
                                          forecast_horizon=model.forecasting_horizon)
        start_time = time.time()
        trained_model = train_model(model=model, data=train, epochs=cfg['epochs'],
                                    batch_size=cfg['batch_size'], verbose=1)
        training_time = time.time() - start_time
        mse, smape, mase, std, c_80, c_95, w_80, w_95, msis_80, msis_95 = \
            test_model(model=trained_model, data=test, naive_error=naive_error, scaler=scaler, cfg=cfg, plot=True,
                       min_max=(np.max(scaler.inverse_transform(train))-np.min(scaler.inverse_transform(train)))/
                               (np.max(train)-np.min(train)),
                       disable_pbar=False)
        msis_80_list.append(msis_80), msis_95_list.append(msis_95)
        forecast_mse_list.append(mse), forecast_smape_list.append(smape), forecast_mase_list.append(mase)
        coverage_80_list.append(c_80), coverage_95_list.append(c_95)
        width_80_list.append(w_80), width_95_list.append(w_95)

    print('========================================================'
          '\n================ Point Forecast Metrics ================'
          '\n========================================================')
    print('Mean forecast MSE:', np.mean(np.mean(forecast_mse_list, axis=0)))
    # print('Forecast MSE:', np.mean(np.array(forecast_mse_list), axis=0))
    print('Mean forecast SMAPE:', np.mean(np.mean(forecast_smape_list, axis=0)))
    # print('Forecast SMAPE:', np.mean(np.array(forecast_smape_list), axis=0))
    print('Mean forecast MASE:', np.mean(np.mean(forecast_mase_list, axis=0)))
    print('Training time:', training_time)

    print('========================================================'
          '\n================== Model Uncertainty ==================='
          '\n========================================================')
    print('80%-prediction interval coverage - Mean:', np.mean(np.mean(coverage_80_list, axis=0)),
          ', width:', np.mean(width_80_list),
          # '\n Forecast horizon:', np.mean(np.array(coverage_80_list), axis=0)
          )
    print('80%-prediction interval MSIS:', np.mean(np.mean(msis_80_list, axis=0)))
    print('95%-prediction interval coverage - Mean:',  np.mean(np.mean(coverage_95_list, axis=0)),
          ', width:', np.mean(width_95_list),
          # '\n Forecast horizon:', np.mean(np.array(coverage_95_list), axis=0)
    )
    print('95%-prediction interval MSIS:', np.mean(np.mean(msis_95_list, axis=0)))


if __name__ == '__main__':
    cfg = load_config_file('config\\config.yml')
    time_series_baseline_pipeline(cfg)