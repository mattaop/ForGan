import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.datasets import mnist
from keras import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm


from generate_arp import generate_arp_data
from generate_sine import generate_sine_data
from split_data import split_sequence


class GAN:
    def __init__(self):
        self.window_size = 24
        self.forecasting_horizon = 1
        self.noise_vector_size = 24

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

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
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (self.noise_vector_size,)
        historic_shape = (self.window_size, 1)

        noise_inp = Input(shape=noise_shape)
        historic_inp = Input(shape=historic_shape)

        hist = SimpleRNN(64, return_sequences=False)(historic_inp)
        hist = LeakyReLU(alpha=0.2)(hist)

        # noise_layer = SimpleRNN(64, return_sequences=True)(noise_inp)
        # noise_layer = LeakyReLU(alpha=0.2)(noise_layer)

        x = Concatenate(axis=1)([hist, noise_inp])
        # x = SimpleRNN(64, return_sequences=False)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        #x = Dense(64)(x)
        #x = LeakyReLU(alpha=0.2)(x)
        prediction = Dense(self.forecasting_horizon, activation='tanh')(x)

        # model = Sequential()

        # model.add(SimpleRNN(64, input_shape=noise_shape))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(64))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(self.forecasting_horizon, activation='tanh'))

        # model.summary()

        # noise = Input(shape=noise_shape)
        # prediction = model(noise)
        model = Model(inputs=[historic_inp, noise_inp], outputs=prediction)
        model.summary()
        return model

    def build_discriminator(self):

        historic_shape = (self.window_size, 1)
        future_shape = (self.forecasting_horizon, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        x = Concatenate(axis=1)([historic_inp, future_inp])

        x = SimpleRNN(64, return_sequences=False)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #x = Dense(64)(x)
        #x = LeakyReLU(alpha=0.2)(x)
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

            noise = np.random.normal(0, 1, (half_batch, self.noise_vector_size))

            # Generate a half batch of new images
            gen_forecasts = self.generator.predict([historic_time_series, noise])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([historic_time_series, future_time_series],
                                                            np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch([historic_time_series, tf.keras.backend.expand_dims(gen_forecasts, axis=-1)],
                                                            np.zeros((half_batch, 1)))
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
        plt.plot(np.linspace(self.window_size, self.window_size + self.forecasting_horizon + steps - 1,
                             self.forecasting_horizon + steps - 1), forecast.mean(axis=2), label='forecasted data')
        plt.legend()
        plt.show()
        print('Forecast error:', mean_squared_error(time_series[0, -len(forecast):], forecast.mean(axis=2)))
        print('Forecast standard deviation', np.mean(forecast.std(axis=2), axis=0))

        plt.hist(forecast[0, 0], color='blue', edgecolor='black',
                 bins=int(50), density=True)
        plt.title('Histogram of predictions')
        plt.xlabel('Predicted value')
        plt.ylabel('Density')
        plt.axvline(forecast[0, 0].mean(), color='b', linewidth=1)
        plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=500, batch_size=32, save_interval=200)
    # gan.forecast(steps=100)
    gan.monte_carlo_forecast(steps=100, mc_forward_passes=5000)
