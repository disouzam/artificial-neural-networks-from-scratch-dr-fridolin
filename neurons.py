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


def threeNeurons_OneLayer(inputs):
    """
    Implements a three neurons in a single layer
    """
    input_length = len(inputs)

    weights = np.random.rand(3, input_length)
    bias = np.random.rand(3, 1)

    out = np.dot(weights, inputs) + bias

    return out


def NNeurons_OneLayer(inputs, n_neurons):
    """
    Implements an arbitrary number of neurons in a single layer
    """
    input_length = len(inputs)

    weights = np.random.rand(n_neurons, input_length)
    bias = np.random.rand(n_neurons, 1)

    out = np.dot(weights, inputs) + bias

    return out
