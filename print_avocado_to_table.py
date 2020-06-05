import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_avocado
show_plot = True


def read_files(file_name, columnname):
    # print( ' file_name', file_name)
    df = pd.read_csv(file_name.lower() + '/' + columnname[1] + '_' + columnname[2] + '_' + 'test_results.txt', header=0)
    return df


def print_point_forecast_results(df, model_paths, model_names):
    # print(model_paths)
    #results = pd.DataFrame(columns=['es_mse', 'es_80', 'es_95', 'mc dropout_mse', 'mc dropout_80', 'mc dropout_95',
                                   # 'forgan_mse', 'forgan_80', 'forgan_95'])
    results = np.zeros([len(df.columns), len(model_names)*3])
    i = 0
    for columnName, columnData in df.iteritems():
        if columnName[2] == 'conventional':
            print('\\multirow{2}{*}{' + columnName[1] + '} & ' + columnName[2], end='')
        else:
            print(' & ' + columnName[2], end='')
        j = 0
        for model_path, model_name in zip(model_paths, model_names):
            df = read_files(model_path, columnName)
            results[i, 3*j] = np.mean(df['mse']) * 1
            results[i, 3*j+1] = np.mean(df['coverage_80']) * 100
            results[i, 3*j+2] = np.mean(df['coverage_95']) * 100
            print(" & $ %.4f $ & $ %.2f $ & $ %.2f $"
                  % (np.mean(df['mse']) * 1, np.mean(df['coverage_80']) * 100, np.mean(df['coverage_95']) * 100), end='')
            j += 1
        print(" \\\\")
        i += 1
    print(np.mean(results, axis=0), np.std(results, axis=0))


def print_uncertainty_results(df, model_paths, model_names):
    for columnName, columnData in df.iteritems():
        for model_path, model_name in zip(model_paths, model_names):
            df = read_files(model_path, columnName)
            print(
                model_name + " & $%.3f $ & $ %.3f $ & $ %.3f $ & $ %.2f \\%%  $  &  $ %.2f \\%% $ & $ %.3f $ & $ %.3f $\\\\ "
                % (np.mean(df['std']) * 1, np.mean(df['msis_80']) * 1, np.mean(df['msis_95']) * 1,
                   np.mean(df['coverage_80']) * 100,
                   np.mean(df['coverage_95']) * 100, np.mean(df['width_80']) * 1,
                   np.mean(df['width_95']) * 1))


def print_results(model_paths, model_names):
    df = load_avocado()
    print_point_forecast_results(df, model_paths, model_names)
    # print_uncertainty_results(df, model_paths, model_names)


def main():
    compare_models = [#'ARIMA',
        'ES', 'MC Dropout', 'ForGAN']
    compare_model_paths = [#'results/avocado/arima',
                                'results/avocado/es',
                                'results/avocado/rnn/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000100',
                                'results/avocado/recurrentgan/minmax/rnn_epochs_30000_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000100']
    model_paths = compare_model_paths
    model_names = compare_models
    print_results(model_paths=model_paths,
                  model_names=model_names)


if __name__ == '__main__':
    main()
