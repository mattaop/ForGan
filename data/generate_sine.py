import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import scipy.stats


def generate_sine_data(num_points=2000, plot=False):
    np.random.seed(0)
    x = np.linspace(1, num_points, num_points)
    mean = 0
    std = 0.1
    noise = np.random.normal(mean, std, num_points)
    scale = scipy.stats.norm.ppf(0.75)*std*1+1
    scale = 1

    y = np.sin(x*np.pi/6)/scale + noise
    if plot:
        plt.hist(noise, color='blue', edgecolor='black',
                 bins=int(50), density=True)
        plt.title('Histogram of predictions')
        plt.xlabel('Predicted value')
        plt.ylabel('Density')
        plt.axvline(noise.mean(), color='b', linewidth=1)
        plt.show()
    print('Noise standard deviation:', noise.std())

    plt.figure()
    plt.title('Sine Curve with Gaussian Noise')
    plt.plot(x, y)
    plt.show()
    return np.expand_dims(y, axis=-1)


if __name__ == '__main__':
    generate_sine_data(200)
