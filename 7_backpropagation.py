import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo  # noqa: F401

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from nnfs.datasets import spiral_data

    from activation_functions import ActivationReLU
    from layer import LayerDense
    from softmax import CalcSoftmaxLossGrad

    return (
        ActivationReLU,
        CalcSoftmaxLossGrad,
        LayerDense,
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
    CalcSoftmaxLossGrad,
    LayerDense,
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
    loss_function = CalcSoftmaxLossGrad()

    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_function.forward(dense2.output, y)

    predictions = np.argmax(loss_function.output, axis=1)
    print(predictions)

    y_recalc = y
    if len(y.shape) == 2:
        y_recalc = np.argmax(y, axis=1)
    return (
        S_softmax,
        activation1,
        dense1,
        dense2,
        loss_function,
        neurons_in_first_layer,
        neurons_in_second_layer,
        predictions,
        y_recalc,
    )


@app.cell
def _(np, predictions, y_recalc):
    accuracy = np.mean(predictions == y_recalc)
    print(accuracy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward propagation
    """)
    return


@app.cell
def _(activation1, dense1, dense2, loss_function, y):
    loss_function.backward(loss_function.output, y)
    dense2.backward(loss_function.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    return


@app.cell
def _(dense1):
    dense1.dbiases
    return


@app.cell
def _(dense1):
    dense1.dweights
    return


@app.cell
def _(dense1):
    dense1.biases
    return


@app.cell
def _(dense1):
    dense1.weigths
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Complete example with a loop to optimize weights and biases
    """)
    return


@app.cell
def _(
    ActivationReLU,
    CalcSoftmaxLossGrad,
    LayerDense,
    S_softmax,
    neurons_in_first_layer,
    neurons_in_second_layer,
    np,
    x,
    y,
):
    accuracies = []

    dense1_complete = LayerDense(S_softmax[1], neurons_in_first_layer)
    dense2_complete = LayerDense(neurons_in_first_layer, neurons_in_second_layer)
    activation1_complete = ActivationReLU()
    loss_function_complete = CalcSoftmaxLossGrad()

    for i in range(1000):
        dense1_complete.forward(x)
        activation1_complete.forward(dense1_complete.output)
        dense2_complete.forward(activation1_complete.output)
        loss_function_complete.forward(dense2_complete.output, y)

        predictions_complete = np.argmax(loss_function_complete.output, axis=1)

        y_recalc_complete = y
        if len(y.shape) == 2:
            y_recalc_complete = np.argmax(y, axis=1)

        accuracy_complete = np.mean(predictions_complete == y_recalc_complete)
        print(accuracy_complete)

        accuracies.append(accuracy_complete)

        loss_function_complete.backward(loss_function_complete.output, y)
        dense2_complete.backward(loss_function_complete.dinputs)
        activation1_complete.backward(dense2_complete.dinputs)
        dense1_complete.backward(activation1_complete.dinputs)

        dense1_complete.weigths -= 0.01 * dense1_complete.dweights
        dense2_complete.weigths -= 0.01 * dense2_complete.dweights

        dense1_complete.biases -= 0.01 * dense1_complete.dbiases
        dense2_complete.biases -= 0.01 * dense2_complete.dbiases
    return (accuracies,)


@app.cell
def _(accuracies):
    accuracies[0]
    return


@app.cell
def _(accuracies):
    accuracies[-1]
    return


@app.cell
def _(accuracies, plt):
    plt.scatter(range(len(accuracies)), accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
