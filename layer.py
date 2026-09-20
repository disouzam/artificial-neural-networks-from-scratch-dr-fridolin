"""
Layers of a neural network
"""

import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weigths = np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weigths) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weigths.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
