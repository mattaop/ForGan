import numpy as np
import pandas as pd


def generate_coefficients(p, stable_condition=True):
    filter_stable = False
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while not filter_stable:
        true_theta = np.random.random(p) - 0.5
        coefficients = np.append(1, -true_theta)
        # check if magnitude of all poles is less than one
        if np.max(np.abs(np.roots(coefficients))) < 1 or not stable_condition:
            filter_stable = True
    return true_theta


def generate_arp_data(p=2, burn_in=600, num_points=2000):
    arp_sequence = np.zeros(p+num_points+burn_in)
    arp_sequence[:p] = np.random.normal(0, 0.2, p)
    coefficients = generate_coefficients(p, stable_condition=True)
    for i in range(p, num_points+burn_in+p):
        for j in range(p):
            arp_sequence[i] += arp_sequence[i-j-1]*coefficients[j]

    arp_sequence += np.random.normal(0, 1, len(arp_sequence))
    return pd.DataFrame({'y': arp_sequence[(p+burn_in):]})
