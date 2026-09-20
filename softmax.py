import numpy as np

from loss import LossCategoricalCrossEntropy


class ActivationSoftmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for i, single_output, single_values in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )

            self.dinputs[i] = np.dot(jacobian_matrix, single_values)


class CalcSoftmaxLossGrad:
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        self.loss.forward(self.output, y_true)

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        number_of_samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(number_of_samples), y_true] -= 1
        self.dinputs = self.dinputs / number_of_samples
