"""test cost"""
import numpy as np

from neuralink.cost import Cost

from .utils import single_test


def test_compute_cost1():
    np.random.seed(1)
    Y = np.random.randn(1, 5) > 0
    A2 = np.array([[0.5002307, 0.49985831, 0.50023963, 0.25, 0.7]])

    expected_output = 0.5447066599017815
    output = Cost().compute(A2, Y)

    assert type(output) == float, "Wrong type. Float expected"
    assert np.isclose(
        output, expected_output
    ), f"Wrong value. Expected: {expected_output} got: {output}"


def test_compute_cost2():
    Y = np.asarray([[1, 1, 0]])
    AL = np.array([[0.8, 0.9, 0.4]])
    expected_output = np.array(0.27977656)

    test_cases = [
        {
            "name": "equation_output_check",
            "input": [AL, Y],
            "expected": expected_output,
            "error": "Wrong output",
        }
    ]

    single_test(test_cases, Cost().compute)


def test_compute_cost_with_regularization():
    np.random.seed(1)
    Y = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    A3 = np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]])
    lambd = 0.1
    expected_output = np.float64(1.7864859451590758)
    test_cases = [
        {
            "name": "shape_check",
            "input": [A3, Y, parameters, lambd],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [A3, Y, parameters, lambd],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    single_test(test_cases, Cost().compute)
