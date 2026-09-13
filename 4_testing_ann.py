import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo  # noqa: F401

    return


@app.cell
def _():
    import numpy as np

    from layer import LayerDense

    return LayerDense, np


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


if __name__ == "__main__":
    app.run()
