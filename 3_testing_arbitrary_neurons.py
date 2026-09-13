import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np

    from neurons import NNeurons_OneLayer

    return NNeurons_OneLayer, np


@app.cell
def _(NNeurons_OneLayer, np):
    input = np.array([[1], [2], [4], [5]])

    output = NNeurons_OneLayer(input, 15)

    print(output)
    return


if __name__ == "__main__":
    app.run()
