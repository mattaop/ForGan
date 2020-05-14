import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_noise(num_points=2000, plot=False):
    noise_distribution = 'normal'
    if noise_distribution.lower() in ['gaussian', 'normal']:
        print('Noise distribution: Gaussian')
        noise = np.random.normal(0, 0.1, (num_points, 1))
    elif noise_distribution.lower() == 'uniform':
        print('Noise distribution: Uniform')
        noise = np.random.uniform(0.2, 1, (num_points, 1))
    elif noise_distribution.lower() == 'gamma':
        print('Noise distribution: Gamma')
        noise = np.random.gamma(2, 2, (num_points, 1))/4
    elif noise_distribution.lower() == 'bimodal':
        print('Noise distribution: Bimodal')
        y1 = np.random.normal(-0.5, 0.2, (num_points, 1))
        y2 = np.random.uniform(1, 2, (num_points, 1))
        w = np.random.binomial(1, 0.5, (num_points, 1))  # 50:50 random choice
        noise = np.multiply(w,  y1) + np.multiply(1 - w, y2)
    else:
        print('Noise distribution: None')
        noise = np.zeros([num_points, 1])

    if plot:
        plt.hist(noise, color='blue', edgecolor='black',
                 bins=int(50), density=True)
        plt.title('Histogram of predictions')
        plt.xlabel('Predicted value')
        plt.ylabel('Density')
        plt.axvline(noise.mean(), color='b', linewidth=1)
        plt.show()

    return noise


if __name__ == '__main__':
    generate_noise(5000)
