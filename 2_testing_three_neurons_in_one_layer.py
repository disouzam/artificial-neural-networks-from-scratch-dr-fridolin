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

    from neurons import threeNeurons_OneLayer

    return np, threeNeurons_OneLayer


@app.cell
def _(np, threeNeurons_OneLayer):
    input = np.array([[1], [2], [4], [5]])

    output = threeNeurons_OneLayer(input)

    print(output)
    return


if __name__ == "__main__":
    app.run()
