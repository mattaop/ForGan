import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

show_plot = True


def read_files(file_name):
    df = pd.read_csv(file_name.lower() + '/validation_results.txt', header=0)
    return df


def print_point_forecast_results(model_paths, labels, model_names):
    for model_paths, label, model_name in zip(model_paths, labels, model_names):
        df = read_files(model_paths)

        """
        print(
            model_name + " & " + label + " & $ %.4f $ & $ %.2f $ & $ %.3f $ & $ %.2f $  &  $ %.2f $ & $ %.3f $ & $ %.3f $\\\\ "
            % (np.mean(df['mse']) * 1, np.mean(df['smape']) * 1, np.mean(df['mase']) * 1,
               np.mean(df['coverage_80']) * 100, np.mean(df['coverage_95']) * 100, np.mean(df['width_80']) * scale,
               np.mean(df['width_95']) * scale))
        
        print(model_name + " & $ %.4f $ & $ %.2f $ & $ %.3f $ & $ %.2f $  &  $ %.2f $ & $ %.3f $ & $ %.3f $\\\\ "
              % (np.mean(df['mse'])*1, np.mean(df['smape'])*1, np.mean(df['mase'])*1,
                 np.mean(df['coverage_80'])*100, np.mean(df['coverage_95'])*100, np.mean(df['width_80'])*1,
                 np.mean(df['width_95'])*1))
        """
        print(model_name + " & $ %.4f $ & $ %.2f $ & $ %.3f $ \\\\ "
              % (np.mean(df['mse']) * 1, np.mean(df['smape']) * 1, np.mean(df['mase']) * 1))


def print_uncertainty_results(model_paths, labels, model_names):
    for model_paths, label, model_name in zip(model_paths, labels, model_names):
        df = read_files(model_paths)
        print(model_name + " & $%.3f $ & $ %.3f $ & $ %.3f $ & $ %.2f \\%%  $  &  $ %.2f \\%% $ & $ %.3f $ & $ %.3f $\\\\ "
              % (np.mean(df['std']) * 1, np.mean(df['msis_80']) * 1, np.mean(df['msis_95']) * 1, np.mean(df['coverage_80']) * 100,
                 np.mean(df['coverage_95']) * 100, np.mean(df['width_80']) * 1,
                 np.mean(df['width_95']) * 1))


def print_results(model_paths, labels, model_names):
    print_point_forecast_results(model_paths, labels, model_names)
    print_uncertainty_results(model_paths, labels, model_names)


def main():
    forgan = ['ForGAN', 'ForGAN', 'ForGAN', 'ForGAN', 'ForGAN', 'ForGAN']

    noise_paths = ['results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_1_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_5_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_10_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_25_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_50_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                   'results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000'
                   ]
    noise_label = ['1', '5', '10', '25', '50', '100']
    d_epochs_label = ["1", "3", "5", "10", "20"]
    d_epochs_paths = ['results/sine/recurrentgan/minmax/rnn_epochs_3000_D_epochs_1_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      'results/sine/recurrentgan/minmax/rnn_epochs_3000_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      'results/sine/recurrentgan/minmax/rnn_epochs_3000_D_epochs_5_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      'results/sine/recurrentgan/minmax/rnn_epochs_3000_D_epochs_10_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                      'results/sine/recurrentgan/minmax/rnn_epochs_3000_D_epochs_20_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000'
]

    compare_models = ['ARIMA', 'ES', 'MC Dropout', 'ForGAN']
    compare_model_paths_oslo = ['results/oslo/arima',
                           'results/oslo/es',
                           'results/oslo/rnn/minmax/rnn_epochs_500_D_epochs_3_batch_size_64_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                                'results/oslo/recurrentgan/minmax/rnn_epochs_5000_D_epochs_10_batch_size_64_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000']

    compare_models = ['ARIMA', 'ES', 'MC Dropout', 'ForGAN']
    compare_model_paths_sine = ['results/sine/arima',
                               'results/sine/es',
                               'results/sine/rnn/minmax/rnn_epochs_2000_D_epochs_3_batch_size_64_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                               'results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000']
    wgan_paths = ['results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_kl_lr_0.001000',
                  'results/sine/recurrentgan/minmax/rnn_epochs_1500_D_epochs_3_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_w_lr_0.001000',
                  'results/sine/recurrentgan/minmax/rnn_epochs_5000_D_epochs_5_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_w_lr_0.000100',
                  'results/sine/recurrentgan/minmax/rnn_epochs_10000_D_epochs_5_batch_size_32_noise_vec_100_gnodes_16_dnodes_64_loss_w_lr_0.000100']
    wgan_models = ['GAN', 'WGAN', 'Optimal WGAN', 'Optimal WGAN']

    model_paths = wgan_paths
    labels = noise_label
    model_names = wgan_models
    print_results(model_paths=model_paths,
                  labels=labels,
                  model_names=model_names,)


if __name__ == '__main__':
    main()
