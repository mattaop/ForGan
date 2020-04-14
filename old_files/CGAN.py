import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import keras.backend as K
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm

from data.generate_sine import generate_sine_data
from utility.split_data import split_sequence


class GAN:
    def __init__(self):
        self.window_size = 24
        self.forecasting_horizon = 12
        self.noise_vector_size = 124  # Try larger vector

        optimizer = Adam(0.0002, 0.5)
        loss_function = self.wasserstein_loss

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=loss_function, optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.noise_vector_size,))
        time_series = Input(shape=(self.window_size, 1))

        forecast = self.generator([time_series, z])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity

        self.expand_dims = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=-1))

        valid = self.discriminator([time_series, self.expand_dims(forecast)])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(inputs=[time_series, z], outputs=valid)
        self.combined.compile(loss=loss_function, optimizer=optimizer)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)

        hist = LSTM(64, return_sequences=False)(historic_inp)
        hist = ReLU()(hist)

        x = Concatenate(axis=1)([hist, noise_inp])
        # x = Dense(64)(x)
        # x = ReLU()(x)
        prediction = Dense(self.forecasting_horizon, activation='relu')(x)

        model = Model(inputs=[historic_inp, noise_inp], outputs=prediction)
        model.summary()
        return model

    def build_discriminator(self):

        historic_shape = (self.window_size, 1)
        future_shape = (self.forecasting_horizon, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        historic = LSTM(64, return_sequences=False)(historic_inp)
        historic = LeakyReLU(alpha=0.2)(historic)

        future = LSTM(64, return_sequences=False)(future_inp)
        future = LeakyReLU(alpha=0.2)(future)

        x = Concatenate(axis=1)([historic, future])
        # x = Dense(64)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model

    def train(self, epochs, batch_size=128, data_samples=5000, save_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        # time_series = generate_arp_data(5, 600, 500)
        time_series = generate_sine_data(data_samples)
        x_train, y_train = split_sequence(time_series, self.window_size, self.forecasting_horizon)
        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)
        forecast_mse = np.zeros(epochs)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            historic_time_series = x_train[idx]
            future_time_series = y_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_vector_size))  # Normalisere til 1

            # Generate a half batch of new images
            gen_forecasts = self.generator.predict([historic_time_series, noise])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([historic_time_series, future_time_series],
                                                            np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch([historic_time_series, tf.keras.backend.expand_dims(gen_forecasts, axis=-1)], np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.noise_vector_size))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            historic_time_series = x_train[idx]
            # Train the generator
            g_loss = self.combined.train_on_batch([historic_time_series, noise], valid_y)

            # Measure forecast MSE of generator
            forecast_mse[epoch] = mean_squared_error(future_time_series[:, :, 0], gen_forecasts)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, forecast mse: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss, forecast_mse[epoch]))

            # If at save interval => save generated image samples
            # if epoch % save_interval == 0:
                # self.save_imgs(epoch)
        plt.figure()
        plt.plot(np.linspace(1, epochs, epochs), forecast_mse, label='Training loss generator')
        plt.legend()
        plt.show()

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("generated_samples/mnist_%d.png" % epoch)
        plt.close()

    def forecast(self, steps=1):
        # time_series = generate_arp_data(5, 600, self.window_size+self.forecasting_horizon+steps-1)
        time_series = generate_sine_data(self.window_size+self.forecasting_horizon+steps-1)
        time_series = np.expand_dims(time_series, axis=0)
        forecast = np.zeros([steps, self.forecasting_horizon])
        for i in range(steps):
            noise = np.random.normal(0, 1, (1, self.noise_vector_size))
            forecast[i] = self.generator.predict([time_series[:, i:self.window_size+i], noise])[0]
        plt.figure()
        plt.plot(np.linspace(1, len(time_series[0]), len(time_series[0])), time_series[0], label='real data')
        plt.plot(np.linspace(self.window_size, self.window_size+self.forecasting_horizon+steps-1,
                             self.forecasting_horizon+steps-1), forecast, label='forecasted data')
        plt.legend()
        plt.show()
        print('Forecast error:', mean_squared_error(time_series[0, -len(forecast):], forecast))

    def monte_carlo_forecast(self, steps=1, mc_forward_passes=500):
        # time_series = generate_arp_data(5, 600, self.window_size+self.forecasting_horizon+steps-1)
        time_series = generate_sine_data(self.window_size + self.forecasting_horizon + steps - 1)
        time_series = np.expand_dims(time_series, axis=0)
        forecast = np.zeros([steps, self.forecasting_horizon, mc_forward_passes])
        for i in tqdm(range(steps)):
            for j in range(mc_forward_passes):
                noise = np.random.normal(0, 1, (1, self.noise_vector_size))
                forecast[i, :, j] = self.generator.predict([time_series[:, i:self.window_size + i], noise])[0]
        plt.figure()
        plt.plot(np.linspace(1, len(time_series[0]), len(time_series[0])), time_series[0], label='real data')
        plt.plot(np.linspace(self.window_size, self.window_size + steps, steps), forecast.mean(axis=2)[:, 0], label='forecasted data')
        plt.legend()
        plt.show()
        print('Forecast error:', mean_squared_error(time_series[0, -len(forecast):], forecast.mean(axis=2)[:, 0]))
        print('Forecast standard deviation', np.mean(forecast.std(axis=2)[:, 0], axis=0))

        plt.hist(forecast[0, 0], color='blue', edgecolor='black',
                 bins=int(50), density=True)
        plt.title('Histogram of predictions')
        plt.xlabel('Predicted value')
        plt.ylabel('Density')
        plt.axvline(forecast[0, 0].mean(), color='b', linewidth=1)
        plt.show()

    # calculate the kl divergence
    def kl_divergence(self, p, q):
        print('KL-divergence:', sum(p[i] * np.log2(p[i] / q[i]) for i in range(len(p))))


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, save_interval=200)
    # gan.forecast(steps=100)
    gan.monte_carlo_forecast(steps=100, mc_forward_passes=1000)
