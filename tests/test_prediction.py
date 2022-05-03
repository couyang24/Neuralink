"""Testing prediction"""
import numpy as np

from neuralink.prediction import LogitPredict


def test_predict():
    w = np.array([[0.3], [0.5], [-0.2]])
    b = -0.33333
    X = np.array([[1.0, -0.3, 1.5], [2, 0, 1], [0, -1.5, 2]])

    pred = LogitPredict().predict(w, b, X)

    assert type(pred) == np.ndarray, f"Wrong type for pred. {type(pred)} != np.ndarray"
    assert pred.shape == (
        1,
        X.shape[1],
    ), f"Wrong shape for pred. {pred.shape} != {(1, X.shape[1])}"
    assert np.bitwise_not(
        np.allclose(pred, [[1.0, 1.0, 1]])
    ), f"Perhaps you forget to add b in the calculation of A"
    assert np.allclose(
        pred, [[1.0, 0.0, 1]]
    ), f"Wrong values for pred. {pred} != {[[1., 0., 1.]]}"
