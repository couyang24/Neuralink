"""Testing modelling"""
from deeplearning.modelling import LogitModel, NNModel
import numpy as np


def test_logit_model():
    np.random.seed(0)

    expected_output = {
        "costs": [np.array(0.69314718)],
        "Y_prediction_test": np.array([[1.0, 1.0, 0.0]]),
        "Y_prediction_train": np.array([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]]),
        "w": np.array([[0.08639757], [-0.08231268], [-0.11798927], [0.12866053]]),
        "b": -0.03983236094816321,
    }

    # Use 3 samples for training
    b, Y, X = (
        1.5,
        np.array([1, 0, 0, 1, 0, 0, 1]).reshape(1, 7),
        np.random.randn(4, 7),
    )

    # Use 6 samples for testing
    x_test = np.random.randn(4, 3)
    y_test = np.array([0, 1, 0])

    model = LogitModel()
    model.train(X, Y, num_iterations=50, learning_rate=0.01)
    d = model.predict(x_test, y_test)

    assert (
        type(d["costs"]) == list
    ), f"Wrong type for d['costs']. {type(d['costs'])} != list"
    assert len(d["costs"]) == 1, f"Wrong length for d['costs']. {len(d['costs'])} != 1"
    assert np.allclose(
        d["costs"], expected_output["costs"]
    ), f"Wrong values for d['costs']. {d['costs']} != {expected_output['costs']}"

    assert (
        type(d["w"]) == np.ndarray
    ), f"Wrong type for d['w']. {type(d['w'])} != np.ndarray"
    assert d["w"].shape == (
        X.shape[0],
        1,
    ), f"Wrong shape for d['w']. {d['w'].shape} != {(X.shape[0], 1)}"
    assert np.allclose(
        d["w"], expected_output["w"]
    ), f"Wrong values for d['w']. {d['w']} != {expected_output['w']}"

    assert np.allclose(
        d["b"], expected_output["b"]
    ), f"Wrong values for d['b']. {d['b']} != {expected_output['b']}"

    assert (
        type(d["Y_prediction_test"]) == np.ndarray
    ), f"Wrong type for d['Y_prediction_test']. {type(d['Y_prediction_test'])} != np.ndarray"
    assert d["Y_prediction_test"].shape == (
        1,
        x_test.shape[1],
    ), f"Wrong shape for d['Y_prediction_test']. {d['Y_prediction_test'].shape} != {(1, x_test.shape[1])}"
    assert np.allclose(
        d["Y_prediction_test"], expected_output["Y_prediction_test"]
    ), f"Wrong values for d['Y_prediction_test']. {d['Y_prediction_test']} != {expected_output['Y_prediction_test']}"

    assert (
        type(d["Y_prediction_train"]) == np.ndarray
    ), f"Wrong type for d['Y_prediction_train']. {type(d['Y_prediction_train'])} != np.ndarray"
    assert d["Y_prediction_train"].shape == (
        1,
        X.shape[1],
    ), f"Wrong shape for d['Y_prediction_train']. {d['Y_prediction_train'].shape} != {(1, X.shape[1])}"
    assert np.allclose(
        d["Y_prediction_train"], expected_output["Y_prediction_train"]
    ), f"Wrong values for d['Y_prediction_train']. {d['Y_prediction_train']} != {expected_output['Y_prediction_train']}"


def test_nn_model_train():
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = np.random.randn(1, 3) > 0
    n_h = 4
    expected_output = {
        "W1": np.array(
            [
                [0.56305445, -1.03925886],
                [0.7345426, -1.36286875],
                [-0.72533346, 1.33753027],
                [0.74757629, -1.38274074],
            ]
        ),
        "b1": np.array([[-0.22240654], [-0.34662093], [0.33663708], [-0.35296113]]),
        "W2": np.array([[1.82196893, 3.09657075, -2.98193564, 3.19946508]]),
        "b2": np.array([[0.21344644]]),
    }

    np.random.seed(3)
    nnmodel = NNModel()
    nnmodel.train(X, Y, n_h, print_cost=False)
    output = nnmodel.parameters

    assert (
        type(output["W1"]) == np.ndarray
    ), f"Wrong type for W1. Expected: {np.ndarray}"
    assert (
        type(output["b1"]) == np.ndarray
    ), f"Wrong type for b1. Expected: {np.ndarray}"
    assert (
        type(output["W2"]) == np.ndarray
    ), f"Wrong type for W2. Expected: {np.ndarray}"
    assert (
        type(output["b2"]) == np.ndarray
    ), f"Wrong type for b2. Expected: {np.ndarray}"

    assert output["W1"].shape == expected_output["W1"].shape, f"Wrong shape for W1."
    assert output["b1"].shape == expected_output["b1"].shape, f"Wrong shape for b1."
    assert output["W2"].shape == expected_output["W2"].shape, f"Wrong shape for W2."
    assert output["b2"].shape == expected_output["b2"].shape, f"Wrong shape for b2."

    assert np.allclose(output["W1"], expected_output["W1"]), "Wrong values for W1"
    assert np.allclose(output["b1"], expected_output["b1"]), "Wrong values for b1"
    assert np.allclose(output["W2"], expected_output["W2"]), "Wrong values for W2"
    assert np.allclose(output["b2"], expected_output["b2"]), "Wrong values for b2"


def test_nn_model_predict():
    np.random.seed(1)
    X = np.random.randn(2, 3)
    parameters = {
        "W1": np.array(
            [
                [-0.00615039, 0.0169021],
                [-0.02311792, 0.03137121],
                [-0.0169217, -0.01752545],
                [0.00935436, -0.05018221],
            ]
        ),
        "W2": np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
        "b1": np.array(
            [[-8.97523455e-07], [8.15562092e-06], [6.04810633e-07], [-2.54560700e-06]]
        ),
        "b2": np.array([[9.14954378e-05]]),
    }
    expected_output = np.array([[True, False, True]])

    nnmodel = NNModel()
    nnmodel.parameters = parameters
    output = nnmodel.predict(X)

    assert np.array_equal(
        output, expected_output
    ), f"Wrong prediction. Expected: {expected_output} got: {output}"
