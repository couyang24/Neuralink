"""test initiation"""
import numpy as np

from neuralink.initiation import (
    HeInitialize,
    RandDeepInitialize,
    RandInitialize,
    ZeroInitialize,
)

from .utils import multiple_test


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

    parameters = RandInitialize().initiate(n_x, n_h, n_y, 2)

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


def test_randinitialize2():
    np.random.seed(1)
    n_x, n_h, n_y = 3, 2, 1
    expected_W1 = np.array(
        [[0.01624345, -0.00611756, -0.00528172], [-0.01072969, 0.00865408, -0.02301539]]
    )
    expected_b1 = np.array([[0.0], [0.0]])
    expected_W2 = np.array([[0.01744812, -0.00761207]])
    expected_b2 = np.array([[0.0]])
    expected_output = {
        "W1": expected_W1,
        "b1": expected_b1,
        "W2": expected_W2,
        "b2": expected_b2,
    }
    test_cases = [
        {
            "name": "datatype_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Datatype mismatch.",
        },
        {
            "name": "equation_output_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, RandInitialize().initiate)


def test_randdeepinitialize():
    layer_dims = [5, 4, 3]
    expected_W1 = np.array(
        [
            [0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
            [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
            [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
            [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068],
        ]
    )
    expected_b1 = np.array([[0.0], [0.0], [0.0], [0.0]])
    expected_W2 = np.array(
        [
            [-0.01185047, -0.0020565, 0.01486148, 0.00236716],
            [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
            [-0.00768836, -0.00230031, 0.00745056, 0.01976111],
        ]
    )
    expected_b2 = np.array([[0.0], [0.0], [0.0]])
    expected_output = {
        "W1": expected_W1,
        "b1": expected_b1,
        "W2": expected_W2,
        "b2": expected_b2,
    }
    test_cases = [
        {
            "name": "datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Datatype mismatch",
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, RandDeepInitialize().initiate)


def test_initialize_parameters_random_deep():
    layer_dims = [3, 2, 1]
    expected_output = {
        "W1": np.array(
            [
                [17.88628473, 4.36509851, 0.96497468],
                [-18.63492703, -2.77388203, -3.54758979],
            ]
        ),
        "b1": np.array([[0.0], [0.0]]),
        "W2": np.array([[-0.82741481, -6.27000677]]),
        "b2": np.array([[0.0]]),
    }

    test_cases = [
        {
            "name": "datatype_check",
            "input": [layer_dims, 3, False, 0.1],
            "expected": expected_output,
            "error": "Datatype mismatch",
        },
        {
            "name": "shape_check",
            "input": [layer_dims, 3, False, 0.1],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims, 3, False, 0.1],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, RandDeepInitialize().initiate)


def test_initialize_parameters_he():

    layer_dims = [3, 1, 2]
    expected_output = {
        "W1": np.array([[1.46040903, 0.3564088, 0.07878985]]),
        "b1": np.array([[0.0]]),
        "W2": np.array([[-2.63537665], [-0.39228616]]),
        "b2": np.array([[0.0], [0.0]]),
    }

    test_cases = [
        {
            "name": "datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Datatype mismatch",
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, HeInitialize().initiate)
