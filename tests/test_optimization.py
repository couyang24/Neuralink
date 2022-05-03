"""Testing optimization"""
import numpy as np

from neuralink.optimization import LogitOptimize


def test_optimize():
    w, b, X, Y = (
        np.array([[1.0], [2.0]]),
        2.0,
        np.array([[1.0, 2.0, -1.0], [3.0, 4.0, -3.2]]),
        np.array([[1, 0, 1]]),
    )
    expected_w = np.array([[-0.70916784], [-0.42390859]])
    expected_b = np.float64(2.26891346)
    expected_params = {"w": expected_w, "b": expected_b}

    expected_dw = np.array([[0.06188603], [-0.01407361]])
    expected_db = np.float64(-0.04709353)
    expected_grads = {"dw": expected_dw, "db": expected_db}

    expected_cost = [5.80154532, 0.31057104]
    expected_output = (expected_params, expected_grads, expected_cost)

    params, grads, costs = LogitOptimize().optimize(
        w, b, X, Y, num_iterations=101, learning_rate=0.1, print_cost=False
    )

    assert type(costs) == list, "Wrong type for costs. It must be a list"
    assert len(costs) == 2, f"Wrong length for costs. {len(costs)} != 2"
    assert np.allclose(
        costs, expected_cost
    ), f"Wrong values for costs. {costs} != {expected_cost}"

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

    assert (
        type(params["w"]) == np.ndarray
    ), f"Wrong type for params['w']. {type(params['w'])} != np.ndarray"
    assert (
        params["w"].shape == w.shape
    ), f"Wrong shape for params['w']. {params['w'].shape} != {w.shape}"
    assert np.allclose(
        params["w"], expected_w
    ), f"Wrong values for params['w']. {params['w']} != {expected_w}"

    assert np.allclose(
        params["b"], expected_b
    ), f"Wrong values for params['b']. {params['b']} != {expected_b}"
