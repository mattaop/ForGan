import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_sine_data(num_points=2000):
    x = np.linspace(1, num_points, num_points)
    uniform_noise = np.random.uniform(-0.2, 0.2, num_points)
    gaussian_noise = np.random.normal(0, 0.1, num_points)
    gamma_noise = np.random.gamma(2, 2, num_points)/4

    # bimodal dist
    y1 = np.random.normal(-0.3, 0.1, num_points)
    y2 = np.random.normal(0.3, 0.1, num_points)
    w = np.random.binomial(1, 0.5, num_points)  # 50:50 random choice
    bimodal_noise = w * y1 + (1 - w) * y2

    y = np.sin(x*0.1) + gaussian_noise

    plt.hist(bimodal_noise, color='blue', edgecolor='black',
             bins=int(50), density=True)
    plt.title('Histogram of predictions')
    plt.xlabel('Predicted value')
    plt.ylabel('Density')
    plt.axvline(bimodal_noise.mean(), color='b', linewidth=1)
    plt.show()

    return pd.DataFrame({'y': y})


if __name__ == '__main__':
    generate_sine_data(5000)
