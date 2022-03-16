"""test cost"""
from deeplearning.cost import Cost
import numpy as np


def test_compute_cost():
    np.random.seed(1)
    Y = np.random.randn(1, 5) > 0
    A2 = np.array([[0.5002307, 0.49985831, 0.50023963, 0.25, 0.7]])

    expected_output = 0.5447066599017815
    output = Cost().compute(A2, Y)

    assert type(output) == float, "Wrong type. Float expected"
    assert np.isclose(
        output, expected_output
    ), f"Wrong value. Expected: {expected_output} got: {output}"
