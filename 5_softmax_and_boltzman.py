import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from nnfs.datasets import spiral_data

    return np, plt, spiral_data


@app.cell
def _(np, spiral_data):
    [x, y] = spiral_data(samples=600, classes=3)

    print(f"Classes = {np.unique(y)}\n\n")
    print(f"First 5 values of x: \n{x[0:5, :]}")
    return x, y


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


if __name__ == "__main__":
    app.run()
