"""
Layers of a neural network
"""

import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weigths = np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weigths) + self.biases
