"""Testing propagate"""
from deeplearning.propagation import LogitPropagate
import numpy as np


def test_propagate():
    """Testing Propagate"""
    w, b = (
        np.array([[1.0], [2.0], [-1]]),
        2.5,
    )
    X = np.array([[1.0, 2.0, -1.0, 0], [3.0, 4.0, -3.2, 1], [3.0, 4.0, -3.2, -3.5]])
    Y = np.array([[1, 1, 0, 0]])

    expected_dw = np.array([[-0.03909333], [0.12501464], [-0.99960809]])
    expected_db = np.float64(0.288106326429569)
    expected_grads = {"dw": expected_dw, "db": expected_db}
    expected_cost = np.array(2.0424567983978403)
    expected_output = (expected_grads, expected_cost)

    grads, cost = LogitPropagate().propagate(w, b, X, Y)

    assert (
        type(grads["dw"]) == np.ndarray
    ), f"Wrong type for grads['dw']. {type(grads['dw'])} != np.ndarray"
    assert (
        grads["dw"].shape == w.shape
    ), f"Wrong shape for grads['dw']. {grads['dw'].shape} != {w.shape}"
    assert np.allclose(
        grads["dw"], expected_dw
    ), f"Wrong values for grads['dw']. {grads['dw']} != {expected_dw}"
    assert np.allclose(
        grads["db"], expected_db
    ), f"Wrong values for grads['db']. {grads['db']} != {expected_db}"
    assert np.allclose(
        cost, expected_cost
    ), f"Wrong values for cost. {cost} != {expected_cost}"
