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
    results = np.zeros([len(model_names), 9, len(df.columns)])
    i = 0
    for columnName, columnData in df.iteritems():
        if columnName[2] == 'conventional':
            print('\\multirow{2}{*}{' + columnName[1] + '} & ' + columnName[2], end='')
        else:
            print(' & ' + columnName[2], end='')
        j = 0
        for model_path, model_name in zip(model_paths, model_names):
            df = read_files(model_path, columnName)
            results[j, 0, i] = np.mean(df['mse']) * 1
            results[j, 1, i] = np.mean(df['smape']) * 1
            results[j, 2, i] = np.mean(df['mase']) * 1
            results[j, 3, i] = np.mean(df['msis_80']) * 1
            results[j, 4, i] = np.mean(df['msis_95']) * 1
            results[j, 5, i] = np.mean(df['coverage_80']) * 100
            results[j, 6, i] = np.mean(df['coverage_95']) * 100
            results[j, 7, i] = np.mean(df['width_80']) * 1
            results[j, 8, i] = np.mean(df['width_95']) * 1

            print(" & $ %.4f $ & $ %.2f $ & $ %.2f $"
                  % (np.mean(df['mse']) * 1, np.mean(df['coverage_80']) * 100, np.mean(df['coverage_95']) * 100), end='')
            j += 1
        print(" \\\\")
        i += 1
    mean = np.mean(results, axis=-1)
    std = np.std(results, axis=-1)
    for i in range(len(mean)):
        """
        print(model_names[i] + " & $ %.3f $ & $ %.3f $ & $ %.2f $ & $ %.2f $ & $ %.3f $ & $ %.3f $ \\\\ "
              % (mean[i, 0], std[i, 0], mean[i, 1],  std[i, 1], mean[i, 2], std[i, 2]))
        """
        print('\\multirow{2}{*}{' + model_names[i] + "} & $ %.3f $ & $ %.2f $ & $ %.3f $ \\\\ "
              % (mean[i, 0],  mean[i, 1], mean[i, 2]))
        print(" & ($ %.3f $) & ($ %.2f $) & ($ %.3f $) \\\\ "
              % (std[i, 0], std[i, 1], std[i, 2]))

    for i in range(len(mean)):
        """
        print(model_names[i] + " & $ %.3f \pm %.3f $ & $ %.3f \pm %.3f $ & $ %.2f \\%% \pm %.2f \\%% $  &  $ %.2f \\%% \pm %.2f \\%% & $ %.3f \pm %.3f $ & $ %.3f \pm %.3f $\\\\"
              % (mean[i, 3], std[i, 3], mean[i, 4],  std[i, 4], mean[i, 5], std[i, 5], mean[i, 6],  std[i, 6], mean[i,  7], std[i, 7], mean[i,  8], std[i, 8]))
        """
        print('\\multirow{2}{*}{' + model_names[i] + "} & $ %.3f $ & $ %.3f $ & $ %.2f \\%%  $  &  $ %.2f \\%% $ & $ %.3f $ & $ %.3f $\\\\ "
              % (mean[i, 3],  mean[i, 4], mean[i, 5], mean[i, 6], mean[i, 7], mean[i, 8]))
        print(" & ($ %.3f $) & ($ %.3f $) & ($ %.2f \\%%  $)  &  ($ %.2f \\%% $) & ($ %.3f $) & ($ %.3f $) \\\\ "
              % (std[i, 3], std[i, 4], std[i, 5], std[i, 6], std[i, 7], std[i, 8]))



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
    compare_models = ['ARIMA', 'ES', 'MC Dropout', 'ForGAN']
    compare_model_paths = ['results/avocado/arima',
                                'results/avocado/es',
                                'results/avocado/rnn/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000100',
                                'results/avocado/recurrentgan/minmax/rnn_epochs_30000_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.000100']
    model_paths = compare_model_paths
    model_names = compare_models
    print_results(model_paths=model_paths,
                  model_names=model_names)


if __name__ == '__main__':
    main()
