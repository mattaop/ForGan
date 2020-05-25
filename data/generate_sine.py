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

    y = np.sin(x*np.pi/6)/(scipy.stats.norm.ppf(0.75)*2) + noise
    if plot:
        plt.hist(noise, color='blue', edgecolor='black',
                 bins=int(50), density=True)
        plt.title('Histogram of predictions')
        plt.xlabel('Predicted value')
        plt.ylabel('Density')
        plt.axvline(noise.mean(), color='b', linewidth=1)
        plt.show()
    print('Noise standard deviation:', noise.std())

    return np.expand_dims(y, axis=-1)


if __name__ == '__main__':
    print(generate_sine_data(500))
