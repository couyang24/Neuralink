"""test update parameters"""
import numpy as np

from deeplearning.parameters import Params


def test_update_params():
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

    grads = {
        "dW1": np.array(
            [
                [0.00023322, -0.00205423],
                [0.00082222, -0.00700776],
                [-0.00031831, 0.0028636],
                [-0.00092857, 0.00809933],
            ]
        ),
        "dW2": np.array(
            [[-1.75740039e-05, 3.70231337e-03, -1.25683095e-03, -2.55715317e-03]]
        ),
        "db1": np.array(
            [[1.05570087e-07], [-3.81814487e-06], [-1.90155145e-07], [5.46467802e-07]]
        ),
        "db2": np.array([[-1.08923140e-05]]),
    }

    expected_W1 = np.array(
        [
            [-0.00643025, 0.01936718],
            [-0.02410458, 0.03978052],
            [-0.01653973, -0.02096177],
            [0.01046864, -0.05990141],
        ]
    )
    expected_b1 = np.array(
        [[-1.02420756e-06], [1.27373948e-05], [8.32996807e-07], [-3.20136836e-06]]
    )
    expected_W2 = np.array([[-0.01041081, -0.04463285, 0.01758031, 0.04747113]])
    expected_b2 = np.array([[0.00010457]])

    expected_output = {
        "W1": expected_W1,
        "b1": expected_b1,
        "W2": expected_W2,
        "b2": expected_b2,
    }

    output = Params().update(parameters, grads)

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
