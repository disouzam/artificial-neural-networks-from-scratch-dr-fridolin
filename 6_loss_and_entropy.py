import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo  # noqa: F401

    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from nnfs.datasets import spiral_data

    from activation_functions import ActivationReLU
    from layer import LayerDense
    from loss import LossCategoricalCrossEntropy
    from softmax import ActivationSoftmax

    return (
        ActivationReLU,
        ActivationSoftmax,
        LayerDense,
        LossCategoricalCrossEntropy,
        np,
        plt,
        spiral_data,
    )


@app.cell
def _(np, spiral_data):
    number_of_classes = 3
    [x, y] = spiral_data(samples=200, classes=number_of_classes)

    print(f"Classes = {np.unique(y)}\n\n")
    print(f"First 5 values of x: \n{x[0:5, :]}")
    return number_of_classes, x, y


@app.cell
def _(np, x, y):
    # Additional inspection of data

    print(f"Shape of y: {y.shape}")
    unique_y, counts_y = np.unique(y, return_counts=True)
    print(dict(zip(unique_y, counts_y)))

    print("\n------\n")

    print(f"Shape of x: {x.shape}")
    return


@app.cell
def _(np, plt, x, y):
    idx0 = np.argwhere(y == 0)
    idx1 = np.argwhere(y == 1)
    idx2 = np.argwhere(y == 2)

    plt.scatter(x[idx0, 0], x[idx0, 1], c="black")
    plt.scatter(x[idx1, 0], x[idx1, 1], c="blue")
    plt.scatter(x[idx2, 0], x[idx2, 1], c="red")

    plt.show()
    return


@app.cell
def _(
    ActivationReLU,
    ActivationSoftmax,
    LayerDense,
    LossCategoricalCrossEntropy,
    np,
    number_of_classes,
    x,
    y,
):
    S_softmax = x.shape

    neurons_in_first_layer = 5
    neurons_in_second_layer = number_of_classes

    dense1 = LayerDense(S_softmax[1], neurons_in_first_layer)
    dense2 = LayerDense(neurons_in_first_layer, neurons_in_second_layer)
    activation1 = ActivationReLU()
    activation2 = ActivationSoftmax()
    loss_function = LossCategoricalCrossEntropy()

    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    softmax_output = activation2.output

    predictions = np.argmax(softmax_output, axis=1)

    print(predictions)

    loss = loss_function.calculate(softmax_output, y)

    print(loss)
    return


if __name__ == "__main__":
    app.run()
