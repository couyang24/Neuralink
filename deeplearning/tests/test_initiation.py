"""test initiation"""
from deeplearning.initiation import ZeroInitialize, RandInitialize
import numpy as np


def test_zeroinitialize():
    """Testing ZeroInitialize"""
    dim = 3
    w, b = ZeroInitialize().initiate(dim)
    assert type(b) == float, f"Wrong type for b. {type(b)} != float"
    assert b == 0.0, "b must be 0.0"
    assert type(w) == np.ndarray, f"Wrong type for w. {type(w)} != np.ndarray"
    assert w.shape == (dim, 1), f"Wrong shape for w. {w.shape} != {(dim, 1)}"
    assert np.allclose(
        w, [[0.0], [0.0], [0.0]]
    ), f"Wrong values for w. {w} != {[[0.], [0.], [0.]]}"

    dim = 4
    w, b = ZeroInitialize().initiate(dim)
    assert type(b) == float, f"Wrong type for b. {type(b)} != float"
    assert b == 0.0, "b must be 0.0"
    assert type(w) == np.ndarray, f"Wrong type for w. {type(w)} != np.ndarray"
    assert w.shape == (dim, 1), f"Wrong shape for w. {w.shape} != {(dim, 1)}"
    assert np.allclose(
        w, [[0.0], [0.0], [0.0], [0.0]]
    ), f"Wrong values for w. {w} != {[[0.], [0.], [0.], [0.]]}"


def test_randinitialize():
    np.random.seed(2)
    n_x, n_h, n_y = 3, 5, 2

    expected_output = {
        "W1": np.array(
            [
                [-0.00416758, -0.00056267, -0.02136196],
                [0.01640271, -0.01793436, -0.00841747],
                [0.00502881, -0.01245288, -0.01057952],
                [-0.00909008, 0.00551454, 0.02292208],
                [0.00041539, -0.01117925, 0.00539058],
            ]
        ),
        "b1": np.array([[0.0], [0.0], [0.0], [0.0], [0.0]]),
        "W2": np.array(
            [
                [
                    -5.96159700e-03,
                    -1.91304965e-04,
                    1.17500122e-02,
                    -7.47870949e-03,
                    9.02525097e-05,
                ],
                [
                    -8.78107893e-03,
                    -1.56434170e-03,
                    2.56570452e-03,
                    -9.88779049e-03,
                    -3.38821966e-03,
                ],
            ]
        ),
        "b2": np.array([[0.0], [0.0]]),
    }

    parameters = RandInitialize().initiate(n_x, n_h, n_y)

    assert (
        type(parameters["W1"]) == np.ndarray
    ), f"Wrong type for W1. Expected: {np.ndarray}"
    assert (
        type(parameters["b1"]) == np.ndarray
    ), f"Wrong type for b1. Expected: {np.ndarray}"
    assert (
        type(parameters["W2"]) == np.ndarray
    ), f"Wrong type for W2. Expected: {np.ndarray}"
    assert (
        type(parameters["b2"]) == np.ndarray
    ), f"Wrong type for b2. Expected: {np.ndarray}"

    assert parameters["W1"].shape == expected_output["W1"].shape, f"Wrong shape for W1."
    assert parameters["b1"].shape == expected_output["b1"].shape, f"Wrong shape for b1."
    assert parameters["W2"].shape == expected_output["W2"].shape, f"Wrong shape for W2."
    assert parameters["b2"].shape == expected_output["b2"].shape, f"Wrong shape for b2."

    assert np.allclose(parameters["W1"], expected_output["W1"]), "Wrong values for W1"
    assert np.allclose(parameters["b1"], expected_output["b1"]), "Wrong values for b1"
    assert np.allclose(parameters["W2"], expected_output["W2"]), "Wrong values for W2"
    assert np.allclose(parameters["b2"], expected_output["b2"]), "Wrong values for b2"
