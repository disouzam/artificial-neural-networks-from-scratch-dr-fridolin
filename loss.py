import numpy as np


class Loss:
    def calculate(self, output, y):
        """
        Arguments:
            output: provided by softmax layer
            y: vector / matrix with actual class values
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):
    def forward(self, prob_pred, y_true):
        s = prob_pred.shape
        samples = s[0]
        y_pred_clipped = np.clip(prob_pred, 1e-7, 1 - 1e-7)

        y_shape = y_true.shape

        if len(y_shape) == 1:
            # Sparse y vector
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_shape) == 2:
            # y vector from one hot enconding
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
