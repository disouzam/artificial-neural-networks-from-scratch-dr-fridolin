"""
Defines a single neuron
"""

import numpy as np


def single_neuron(inputs):
    """
    Implements a single neuron
    """
    input_length = len(inputs)

    weights = np.random.rand(1, input_length)
    bias = np.random.rand(1, 1)

    out = np.dot(weights, inputs) + bias

    return out
