import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

show_plot = True


def read_files(file_name):
    df = pd.read_csv(file_name + '/test_results.txt', header=0)
    return df


def print_results(model_paths, labels, model_names):
    plt.figure()
    for model_paths, label, model_name in zip(model_paths, labels, model_names):
        df = read_files(model_paths)
        print(model_name + " & " + label + " & $ %.5f $ & $ %.2f $ &  $ %.2f $  &  $ %.2f $ \\\\ "
              % (np.mean(df['mse'])*1, np.mean(df['smape'])*1, np.mean(df['coverage_80'])*100, np.mean(df['coverage_95'])*100))


def main():
    forgan = ['ForGAN', 'ForGAN', 'ForGAN', 'ForGAN', 'ForGAN', 'ForGAN']
    d_epochs_label = ["1", "2", "3", "5", "10", "20"]
    d_epochs_paths = ['results/sine/recurrentgan/Epochs_1500_D_epochs_1_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_2_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_3_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_5_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_10_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_20_batch_size_64_noise_vec_50_lr_0.001000']

    compare_models = ['ARIMA', 'ES', 'MC Dropout', 'ForGAN']
    compare_model_paths = ['results/sine/arima',
                           'results/sine/es',
                           'results/sine/rnn/Epochs_1500_D_epochs_50_batch_size_64_noise_vec_50_lr_0.001000',
                           'results/sine/recurrentgan/Epochs_1000_D_epochs_10_batch_size_64_noise_vec_100_lr_0.001000']

    model_paths = compare_model_paths
    labels = ['', '', '', '']
    model_names = compare_models
    print_results(model_paths=model_paths,
                  labels=labels,
                  model_names=model_names,)


if __name__ == '__main__':
    main()
