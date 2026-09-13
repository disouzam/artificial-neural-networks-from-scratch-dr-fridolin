import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo  # noqa: F401

    return


@app.cell
def _():
    from neurons import single_neuron

    return (single_neuron,)


@app.cell
def _(single_neuron):
    input = [1, 2, 4, 5]

    output = single_neuron(input)

    print(output)


if __name__ == "__main__":
    app.run()
