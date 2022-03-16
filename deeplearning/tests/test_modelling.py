"""Testing modelling"""
from deeplearning.modelling import Model
import numpy as np


def test_model():
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

    model = Model()
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
