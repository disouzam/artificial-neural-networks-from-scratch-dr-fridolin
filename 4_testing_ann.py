import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo  # noqa: F401

    return (mo,)


@app.cell
def _():
    import numpy as np

    from activation_functions import ActivationReLU
    from layer import LayerDense

    return ActivationReLU, LayerDense, np


@app.cell
def _(LayerDense, np):
    input = np.array([[1, 2, 5, 7], [4, 5, 7, 8]])

    layer1 = LayerDense(input.shape[1], 5)
    return input, layer1


@app.cell
def _(layer1):
    layer1.weigths
    return


@app.cell
def _(layer1):
    layer1.biases
    return


@app.cell
def _(input, layer1):
    layer1.forward(input)
    layer1.output
    return


@app.cell
def _(LayerDense, layer1):
    layer2 = LayerDense(layer1.output.shape[1], 2)
    layer2.forward(layer1.output)
    layer2.output
    return


@app.cell
def _(ActivationReLU):
    activation_1 = ActivationReLU()

    test_input = [1, -6, 0, 7, -8]

    activation_1.forward(test_input)
    activation_1.output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Complete example

    Adapted from https://github.com/DrBigMau/CNN-from-Scratch-in-Python/blob/main/3%20activation%20functions/Test%20My%20ANN.py
    """)
    return


@app.cell
def _(ActivationReLU, LayerDense, np):
    input_for_activation_function = [[1, -2, -5, 7], [4, -5, 7, -8]]
    input_for_activation_function = np.array(input_for_activation_function)
    S = input_for_activation_function.shape

    neurons_in_first_layer = 5
    neurons_in_second_layer = 6

    dense1 = LayerDense(S[1], neurons_in_first_layer)
    dense2 = LayerDense(neurons_in_first_layer, neurons_in_second_layer)
    activation1 = ActivationReLU()

    dense1.forward(input_for_activation_function)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    return activation1, dense1, dense2


@app.cell
def _(dense1):
    dense1.output
    return


@app.cell
def _(activation1):
    activation1.output
    return


@app.cell
def _(dense2):
    out = dense2.output
    out
    return (out,)


@app.cell
def _(out):
    out.shape
    return


if __name__ == "__main__":
    app.run()
