import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_sine_data(num_points=2000, plot=False):
    noise_distribution = 'gaussian'
    x = np.linspace(1, num_points, num_points)
    if noise_distribution.lower() in ['gaussian', 'normal']:
        print('Noise distribution: Gaussian')
        noise = np.random.normal(0, 0.1, num_points)
    elif noise_distribution.lower() == 'uniform':
        print('Noise distribution: Uniform')
        noise = np.random.uniform(-0.2, 0.2, num_points)
    elif noise_distribution.lower() == 'gamma':
        print('Noise distribution: Gamma')
        noise = np.random.gamma(2, 2, num_points)/4
    elif noise_distribution.lower() == 'bimodal':
        print('Noise distribution: Bimodal')
        y1 = np.random.normal(-0.3, 0.1, num_points)
        y2 = np.random.normal(0.3, 0.1, num_points)
        w = np.random.binomial(1, 0.5, num_points)  # 50:50 random choice
        noise = w * y1 + (1 - w) * y2
    else:
        print('Noise distribution: None')
        noise = np.zeros(num_points)

    y = np.sin(x*np.pi/(6*6)) + noise
    if plot:
        plt.hist(noise, color='blue', edgecolor='black',
                 bins=int(50), density=True)
        plt.title('Histogram of predictions')
        plt.xlabel('Predicted value')
        plt.ylabel('Density')
        plt.axvline(noise.mean(), color='b', linewidth=1)
        plt.show()

    return np.expand_dims(y, axis=-1)


if __name__ == '__main__':
    generate_sine_data(5000)
