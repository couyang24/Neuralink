"""Testing layer"""
import numpy as np

from neuralink.layer import Layer


def test_layer():
    np.random.seed(1)
    X = np.random.randn(5, 3)
    Y = np.random.randn(2, 3)
    expected_output = (5, 4, 2)

    output = Layer().determine(X, Y)

    assert type(output) == tuple, "Output must be a tuple"
    assert (
        output == expected_output
    ), f"Wrong result. Expected {expected_output} got {output}"

    X = np.random.randn(7, 5)
    Y = np.random.randn(5, 5)
    expected_output = (7, 4, 5)

    output = Layer().determine(X, Y)

    assert type(output) == tuple, "Output must be a tuple"
    assert (
        output == expected_output
    ), f"Wrong result. Expected {expected_output} got {output}"
