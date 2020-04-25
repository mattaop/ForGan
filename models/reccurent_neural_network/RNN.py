import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import normalize
from tqdm import tqdm

from data.generate_sine import generate_sine_data
from utility.split_data import split_sequence


class RNN:
    def __init__(self):
        self.window_size = 24
        self.forecasting_horizon = 1
        self.optimizer = Adam(lr=0.001)
        self.loss_function = 'mse'

        # Build and compile the discriminator
        self.model = None

    def build_model(self):
        input_shape = (self.window_size, 1)
        inp = Input(shape=input_shape)

        x = SimpleRNN(64, return_sequences=False)(inp)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.4)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.4)(x)
        output = Dense(self.forecasting_horizon)(x)

        model = Model(inputs=inp, outputs=output)
        model.compile(optimizer=self.optimizer, loss=self.loss_function)
        model.summary()

        self.model = model

    def train(self, epochs, batch_size=128, data_samples=5000):
        # Load the data
        data = generate_sine_data(data_samples)
        x_train, y_train = split_sequence(data, self.window_size, self.forecasting_horizon)

        history = self.model.fit(x_train, y_train[:, :, 0], epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

        plt.figure()
        plt.plot(np.linspace(1, epochs, epochs), history.history['loss'], label='Training loss')
        plt.legend()
        plt.show()

    def forecast(self, data):
        return self.model.predict(data)

    def monte_carlo_forecast(self, data, steps=1, mc_forward_passes=500, plot=False):
        time_series = np.expand_dims(data, axis=0)
        forecast = np.zeros([steps, self.forecasting_horizon, mc_forward_passes])
        func = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        for i in tqdm(range(steps)):
            x_input = np.vstack([time_series[:, i:self.window_size + i]]*mc_forward_passes)
            forecast[i] = func([x_input, 0.4])
            #for j in range(mc_forward_passes):
             #   forecast[i, :, j] = func([time_series[:, i:self.window_size + i], 0.4])[0]
        if plot:
            plt.figure()
            plt.plot(np.linspace(1, len(data[0]), len(data[0])), data[0], label='real data')
            plt.plot(np.linspace(self.window_size, self.window_size + steps, steps), forecast.mean(axis=2)[:, 0],
                     label='forecasted data')
            plt.legend()
            plt.show()

        print('Forecast error:', mean_squared_error(time_series[0, -len(forecast):], forecast.mean(axis=2)[:, 0]))
        print('Forecast standard deviation', np.mean(forecast.std(axis=2)[:, 0], axis=0))
        return forecast


if __name__ == '__main__':
    rnn = RNN()
    rnn.build_model()
    rnn.train(epochs=500, batch_size=512)
    rnn.monte_carlo_forecast(generate_sine_data(5000), steps=100, mc_forward_passes=5000)
