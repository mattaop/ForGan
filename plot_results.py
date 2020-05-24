import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

show_plot = True


def read_training_files(file_name, training):
    if training:
        df = pd.read_csv(file_name + '/training_results.txt', header=0)
    else:
        df = pd.read_csv(file_name + '/test_results.txt', header=0)
    return df


def plot_results(save_file, model_paths, file_labels, title, x_label, y_label, value='mse', training=True, plot_rate=1):
    colormap = plt.cm.get_cmap('RdBu', len(model_paths))
    print(colormap)
    plt.figure()
    for model_paths, file_label in zip(model_paths, file_labels):
        df = read_training_files(model_paths, training)[value]
        t = np.linspace(1, len(df) * plot_rate, len(df))
        plt.plot(t, df, label=file_label)
    if value == 'mse' and training:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig('plots/' + save_file)
    if show_plot:
        plt.show()


def plot_training_results(model_paths, file_labels, title, save_file='Train', plot_rate=1):
    plot_results(save_file=save_file + '_train_mse', model_paths=model_paths, file_labels=file_labels, title=title[0],
                 x_label='Epochs', y_label='Mean Squared Error (MSE)', value='mse', training=True, plot_rate=plot_rate)
    plot_results(save_file=save_file + '_train_80', model_paths=model_paths, file_labels=file_labels, title=title[1],
                 x_label='Epochs', y_label='Coverage', value='coverage_80', training=True, plot_rate=plot_rate)
    plot_results(save_file=save_file + '_train_95', model_paths=model_paths, file_labels=file_labels, title=title[2],
                 x_label='Epochs', y_label='Coverage', value='coverage_95', training=True, plot_rate=plot_rate)


def plot_test_results(model_paths, file_labels, title, save_file='Test'):
    plot_results(save_file=save_file + '_test_mse', model_paths=model_paths, file_labels=file_labels, title=title[0],
                 x_label='Forecast horizon', y_label='Mean Squared Error (MSE)', value='mse', training=False,
                 plot_rate=1)
    plot_results(save_file=save_file + '_test_smape', model_paths=model_paths, file_labels=file_labels, title=title[1],
                 x_label='Forecast horizon', y_label='Symmetric Mean Absolute Percentage Error (sMAPE)', value='smape',
                 training=False, plot_rate=1)
    plot_results(save_file=save_file + '_test_80', model_paths=model_paths, file_labels=file_labels, title=title[2],
                 x_label='Forecast horizon', y_label='Coverage', value='coverage_80', training=False,
                 plot_rate=1)
    plot_results(save_file=save_file + '_test_95', model_paths=model_paths, file_labels=file_labels, title=title[3],
                 x_label='Forecast horizon', y_label='Coverage', value='coverage_95', training=False,
                 plot_rate=1)


def main():
    noise_paths = ['results/sine/recurrentgan/Epochs_4000_D_epochs_3_batch_size_64_noise_vec_1_lr_0.001000',
                   'results/sine/recurrentgan/Epochs_4000_D_epochs_3_batch_size_64_noise_vec_5_lr_0.001000',
                   'results/sine/recurrentgan/Epochs_4000_D_epochs_3_batch_size_64_noise_vec_10_lr_0.001000',
                   'results/sine/recurrentgan/Epochs_4000_D_epochs_3_batch_size_64_noise_vec_25_lr_0.001000',
                   'results/sine/recurrentgan/Epochs_4000_D_epochs_3_batch_size_64_noise_vec_50_lr_0.001000',
                   'results/sine/recurrentgan/Epochs_4000_D_epochs_3_batch_size_64_noise_vec_100_lr_0.001000']
    noise_label = ['Noise vector = 1', 'Noise vector = 5', 'Noise vector = 10', 'Noise vector = 25',
                   'Noise vector = 50', 'Noise vector = 100']

    d_epochs_paths = ['results/sine/recurrentgan/Epochs_1500_D_epochs_1_batch_size_64_noise_vec_50_lr_0.001000',
                      # 'results/sine/recurrentgan/Epochs_1500_D_epochs_2_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_3_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_5_batch_size_64_noise_vec_50_lr_0.001000',
                      'results/sine/recurrentgan/Epochs_1500_D_epochs_10_batch_size_64_noise_vec_50_lr_0.001000',
                      # 'results/sine/recurrentgan/Epochs_1500_D_epochs_15_batch_size_64_noise_vec_50_lr_0.001000',
                      # 'results/sine/recurrentgan/Epochs_1500_D_epochs_20_batch_size_64_noise_vec_50_lr_0.001000',
                      # 'results/sine/recurrentgan/Epochs_1500_D_epochs_50_batch_size_64_noise_vec_50_lr_0.001000'
                      ]
    d_epochs_label = ['D$_{epochs}=1$',
                      # 'D$_{epochs}=2$',
                      'D$_{epochs}=3$',
                      'D$_{epochs}=5$',
                      'D$_{epochs}=10$',
                      # 'D$_{epochs}=15$', 'D$_{epochs}=20$', 'D$_{epochs}=50$'
                      ]

    model_paths = d_epochs_paths
    labels = d_epochs_label
    plot_training_results(model_paths=model_paths,
                          file_labels=labels,
                          save_file='sine_d_epochs',
                          title=['Training Mean Squared Error',
                                 'Training 80% Prediction Interval Coverage',
                                 'Training 95% Prediction Interval Coverage'],
                          plot_rate=25)

    model_paths = ['results/sine/arima',
                   'results/sine/es',
                   'results/sine/rnn/Epochs_1500_D_epochs_50_batch_size_64_noise_vec_50_lr_0.001000',
                   'results/sine/recurrentgan/Epochs_1000_D_epochs_10_batch_size_64_noise_vec_100_lr_0.001000']
    labels = ['ARIMA',
              'Exponential Smoothing',
              'MC Dropout',
              'ForGAN']
    plot_test_results(model_paths=model_paths,
                      file_labels=labels,
                      save_file='sine_compare_models',
                      title=['Forecast Mean Squared Error',
                             'Forecast Symmetric Mean Absolute Percentage Error',
                             'Forecast 80% Prediction Interval Coverage',
                             'Forecast 95% Prediction Interval Coverage'])


if __name__ == '__main__':
    main()
