"""Testing propagate"""
import numpy as np

from deeplearning.propagation import (BackwardPropagate, ForwardPropagate,
                                      LogitPropagate)


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


def test_forward_propagate():
    """Testing ForwardPropagate"""
    np.random.seed(1)
    X = np.random.randn(2, 3)
    b1 = np.random.randn(4, 1)
    b2 = np.array([[-1.3]])

    parameters = {
        "W1": np.array(
            [
                [-0.00416758, -0.00056267],
                [-0.02136196, 0.01640271],
                [-0.01793436, -0.00841747],
                [0.00502881, -0.01245288],
            ]
        ),
        "W2": np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
        "b1": b1,
        "b2": b2,
    }
    expected_A1 = np.array(
        [
            [0.9400694, 0.94101876, 0.94118266],
            [-0.67151964, -0.62547205, -0.65709025],
            [0.29034152, 0.31196971, 0.33449821],
            [-0.22397799, -0.25730819, -0.2197236],
        ]
    )
    expected_A2 = np.array([[0.21292656, 0.21274673, 0.21295976]])

    expected_Z1 = np.array(
        [
            [1.7386459, 1.74687437, 1.74830797],
            [-0.81350569, -0.73394355, -0.78767559],
            [0.29893918, 0.32272601, 0.34788465],
            [-0.2278403, -0.2632236, -0.22336567],
        ]
    )

    expected_Z2 = np.array([[-1.30737426, -1.30844761, -1.30717618]])
    expected_cache = {
        "Z1": expected_Z1,
        "A1": expected_A1,
        "Z2": expected_Z2,
        "A2": expected_A2,
    }
    expected_output = (expected_A2, expected_cache)

    output = ForwardPropagate().propagate(X, parameters)

    assert type(output[0]) == np.ndarray, f"Wrong type for A2. Expected: {np.ndarray}"
    assert (
        type(output[1]["Z1"]) == np.ndarray
    ), f"Wrong type for cache['Z1']. Expected: {np.ndarray}"
    assert (
        type(output[1]["A1"]) == np.ndarray
    ), f"Wrong type for cache['A1']. Expected: {np.ndarray}"
    assert (
        type(output[1]["Z2"]) == np.ndarray
    ), f"Wrong type for cache['Z2']. Expected: {np.ndarray}"

    assert output[0].shape == expected_A2.shape, f"Wrong shape for A2."
    assert output[1]["Z1"].shape == expected_Z1.shape, f"Wrong shape for cache['Z1']."
    assert output[1]["A1"].shape == expected_A1.shape, f"Wrong shape for cache['A1']."
    assert output[1]["Z2"].shape == expected_Z2.shape, f"Wrong shape for cache['Z2']."

    assert np.allclose(output[0], expected_A2), "Wrong values for A2"
    assert np.allclose(output[1]["Z1"], expected_Z1), "Wrong values for cache['Z1']"
    assert np.allclose(output[1]["A1"], expected_A1), "Wrong values for cache['A1']"
    assert np.allclose(output[1]["Z2"], expected_Z2), "Wrong values for cache['Z2']"


def test_backward_propagate():
    """Testing BackwardPropagate"""
    np.random.seed(1)
    X = np.random.randn(3, 7)
    Y = np.random.randn(1, 7) > 0
    parameters = {
        "W1": np.random.randn(9, 3),
        "W2": np.random.randn(1, 9),
        "b1": np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        "b2": np.array([[0.0]]),
    }

    cache = {
        "A1": np.random.randn(9, 7),
        "A2": np.random.randn(1, 7),
        "Z1": np.random.randn(9, 7),
        "Z2": np.random.randn(1, 7),
    }

    expected_output = {
        "dW1": np.array(
            [
                [-0.24468458, -0.24371232, 0.15959609],
                [0.7370069, -0.64785999, 0.23669823],
                [0.47936123, -0.01516428, 0.01566728],
                [0.03361075, -0.0930929, 0.05581073],
                [0.52445178, -0.03895358, 0.09180612],
                [-0.17043596, 0.13406378, -0.20952062],
                [0.76144791, -0.41766018, 0.02544078],
                [0.22164791, -0.33081645, 0.19526915],
                [0.25619969, -0.09561825, 0.05679075],
            ]
        ),
        "db1": np.array(
            [
                [0.1463639],
                [-0.33647992],
                [-0.51738006],
                [-0.07780329],
                [-0.57889514],
                [0.28357278],
                [-0.39756864],
                [-0.10510329],
                [-0.13443244],
            ]
        ),
        "dW2": np.array(
            [
                [
                    -0.35768529,
                    0.22046323,
                    -0.29551566,
                    -0.12202786,
                    0.18809552,
                    0.13700323,
                    0.35892872,
                    -0.02220353,
                    -0.03976687,
                ]
            ]
        ),
        "db2": np.array([[-0.78032466]]),
    }

    output = BackwardPropagate().propagate(parameters, cache, X, Y)

    assert (
        type(output["dW1"]) == np.ndarray
    ), f"Wrong type for dW1. Expected: {np.ndarray}"
    assert (
        type(output["db1"]) == np.ndarray
    ), f"Wrong type for db1. Expected: {np.ndarray}"
    assert (
        type(output["dW2"]) == np.ndarray
    ), f"Wrong type for dW2. Expected: {np.ndarray}"
    assert (
        type(output["db2"]) == np.ndarray
    ), f"Wrong type for db2. Expected: {np.ndarray}"

    assert output["dW1"].shape == expected_output["dW1"].shape, f"Wrong shape for dW1."
    assert output["db1"].shape == expected_output["db1"].shape, f"Wrong shape for db1."
    assert output["dW2"].shape == expected_output["dW2"].shape, f"Wrong shape for dW2."
    assert output["db2"].shape == expected_output["db2"].shape, f"Wrong shape for db2."

    assert np.allclose(output["dW1"], expected_output["dW1"]), "Wrong values for dW1"
    assert np.allclose(output["db1"], expected_output["db1"]), "Wrong values for db1"
    assert np.allclose(output["dW2"], expected_output["dW2"]), "Wrong values for dW2"
    assert np.allclose(output["db2"], expected_output["db2"]), "Wrong values for db2"
