
"""Ce script est un exemple de matplotlib"""

import numpy as np


def moving_average(x, n, type_='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type_ == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a_liss = np.convolve(x, weights, mode='full')[:len(x)]
    a_liss[:n] = a_liss[n]

    return a_liss


def get_a_b(x1, y1, x2, y2):
    a = (y1 - y2)/(x1 - x2)
    b = y1 - a*x1
    return a, b
