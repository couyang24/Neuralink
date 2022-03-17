"""test cost"""
import numpy as np

from deeplearning.cost import Cost

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
