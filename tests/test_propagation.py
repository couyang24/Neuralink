"""Testing propagate"""
import numpy as np

from neuralink.propagation import (
    BackwardPropagate,
    ForwardPropagate,
    LinearModelBackward,
    LinearModelForward,
    LogitPropagate,
    _linear_activation_backward,
    _linear_activation_forward,
    _linear_backward,
    _linear_forward,
)

from .utils import multiple_test


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


def test_linear_forward():
    np.random.seed(1)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    expected_cache = (A_prev, W, b)
    expected_Z = np.array([[3.26295337, -1.23429987]])
    expected_output = (expected_Z, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Datatype mismatch",
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, _linear_forward)


def test_linear_forward_activate():
    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    expected_linear_cache = (A_prev, W, b)
    expected_Z = np.array([[3.43896131, -2.08938436]])
    expected_cache = (expected_linear_cache, expected_Z)
    expected_A_sigmoid = np.array([[0.96890023, 0.11013289]])
    expected_A_relu = np.array([[3.43896131, 0.0]])

    expected_output_sigmoid = (expected_A_sigmoid, expected_cache)
    expected_output_relu = (expected_A_relu, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [A_prev, W, b, "sigmoid"],
            "expected": expected_output_sigmoid,
            "error": "Datatype mismatch with sigmoid activation",
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, "sigmoid"],
            "expected": expected_output_sigmoid,
            "error": "Wrong shape with sigmoid activation",
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, "sigmoid"],
            "expected": expected_output_sigmoid,
            "error": "Wrong output with sigmoid activation",
        },
        {
            "name": "datatype_check",
            "input": [A_prev, W, b, "relu"],
            "expected": expected_output_relu,
            "error": "Datatype mismatch with relu activation",
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, "relu"],
            "expected": expected_output_relu,
            "error": "Wrong shape with relu activation",
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, "relu"],
            "expected": expected_output_relu,
            "error": "Wrong output with relu activation",
        },
    ]

    multiple_test(test_cases, _linear_activation_forward)


def test_linear_model_forward():
    np.random.seed(6)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    expected_cache = [
        (
            (
                np.array(
                    [
                        [-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                        [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                        [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                        [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                        [0.07612761, -0.15512816, 0.63422534, 0.810655],
                    ]
                ),
                np.array(
                    [
                        [0.35480861, 1.81259031, -1.3564758, -0.46363197, 0.82465384],
                        [-1.17643148, 1.56448966, 0.71270509, -0.1810066, 0.53419953],
                        [-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143],
                        [
                            -0.02195668,
                            -2.12714455,
                            -0.83440747,
                            -0.46550831,
                            0.23371059,
                        ],
                    ]
                ),
                np.array([[1.38503523], [-0.51962709], [-0.78015214], [0.95560959]]),
            ),
            np.array(
                [
                    [-5.23825714, 3.18040136, 0.4074501, -1.88612721],
                    [-2.77358234, -0.56177316, 3.18141623, -0.99209432],
                    [4.18500916, -1.78006909, -0.14502619, 2.72141638],
                    [5.05850802, -1.25674082, -3.54566654, 3.82321852],
                ]
            ),
        ),
        (
            (
                np.array(
                    [
                        [0.0, 3.18040136, 0.4074501, 0.0],
                        [0.0, 0.0, 3.18141623, 0.0],
                        [4.18500916, 0.0, 0.0, 2.72141638],
                        [5.05850802, 0.0, 0.0, 3.82321852],
                    ]
                ),
                np.array(
                    [
                        [-0.12673638, -1.36861282, 1.21848065, -0.85750144],
                        [-0.56147088, -1.0335199, 0.35877096, 1.07368134],
                        [-0.37550472, 0.39636757, -0.47144628, 2.33660781],
                    ]
                ),
                np.array([[1.50278553], [-0.59545972], [0.52834106]]),
            ),
            np.array(
                [
                    [2.2644603, 1.09971298, -2.90298027, 1.54036335],
                    [6.33722569, -2.38116246, -4.11228806, 4.48582383],
                    [10.37508342, -0.66591468, 1.63635185, 8.17870169],
                ]
            ),
        ),
        (
            (
                np.array(
                    [
                        [2.2644603, 1.09971298, 0.0, 1.54036335],
                        [6.33722569, 0.0, 0.0, 4.48582383],
                        [10.37508342, 0.0, 1.63635185, 8.17870169],
                    ]
                ),
                np.array([[0.9398248, 0.42628539, -0.75815703]]),
                np.array([[-0.16236698]]),
            ),
            np.array([[-3.19864676, 0.87117055, -1.40297864, -3.00319435]]),
        ),
    ]
    expected_AL = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    expected_output = (expected_AL, expected_cache)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "The function should return a numpy array.",
        },
        {
            "name": "shape_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, LinearModelForward().propagate)


def test_linear_backward():
    np.random.seed(1)
    dZ = np.random.randn(3, 4)
    A = np.random.randn(5, 4)
    W = np.random.randn(3, 5)
    b = np.random.randn(3, 1)
    linear_cache = (A, W, b)
    expected_dA_prev = np.array(
        [
            [-1.15171336, 0.06718465, -0.3204696, 2.09812712],
            [0.60345879, -3.72508701, 5.81700741, -3.84326836],
            [-0.4319552, -1.30987417, 1.72354705, 0.05070578],
            [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
            [-2.52214926, 2.67882552, -0.67947465, 1.48119548],
        ]
    )
    expected_dW = np.array(
        [
            [0.07313866, -0.0976715, -0.87585828, 0.73763362, 0.00785716],
            [0.85508818, 0.37530413, -0.59912655, 0.71278189, -0.58931808],
            [0.97913304, -0.24376494, -0.08839671, 0.55151192, -0.10290907],
        ]
    )
    expected_db = np.array([[-0.14713786], [-0.11313155], [-0.13209101]])
    expected_output = (expected_dA_prev, expected_dW, expected_db)
    test_cases = [
        {
            "name": "datatype_check",
            "input": [dZ, linear_cache],
            "expected": expected_output,
            "error": "Data type mismatch",
        },
        {
            "name": "shape_check",
            "input": [dZ, linear_cache],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [dZ, linear_cache],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]
    multiple_test(test_cases, _linear_backward)


def test_linear_activation_backward():
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    expected_dA_prev_sigmoid = np.array(
        [[0.11017994, 0.01105339], [0.09466817, 0.00949723], [-0.05743092, -0.00576154]]
    )
    expected_dW_sigmoid = np.array([[0.10266786, 0.09778551, -0.01968084]])
    expected_db_sigmoid = np.array([[-0.05729622]])
    expected_output_sigmoid = (
        expected_dA_prev_sigmoid,
        expected_dW_sigmoid,
        expected_db_sigmoid,
    )

    expected_dA_prev_relu = np.array(
        [[0.44090989, 0.0], [0.37883606, 0.0], [-0.2298228, 0.0]]
    )
    expected_dW_relu = np.array([[0.44513824, 0.37371418, -0.10478989]])
    expected_db_relu = np.array([[-0.20837892]])
    expected_output_relu = (expected_dA_prev_relu, expected_dW_relu, expected_db_relu)

    test_cases = [
        {
            "name": "datatype_check",
            "input": [dA, linear_activation_cache, "sigmoid"],
            "expected": expected_output_sigmoid,
            "error": "Data type mismatch with sigmoid activation",
        },
        {
            "name": "shape_check",
            "input": [dA, linear_activation_cache, "sigmoid"],
            "expected": expected_output_sigmoid,
            "error": "Wrong shape with sigmoid activation",
        },
        {
            "name": "equation_output_check",
            "input": [dA, linear_activation_cache, "sigmoid"],
            "expected": expected_output_sigmoid,
            "error": "Wrong output with sigmoid activation",
        },
        {
            "name": "datatype_check",
            "input": [dA, linear_activation_cache, "relu"],
            "expected": expected_output_relu,
            "error": "Data type mismatch with relu activation",
        },
        {
            "name": "shape_check",
            "input": [dA, linear_activation_cache, "relu"],
            "expected": expected_output_relu,
            "error": "Wrong shape with relu activation",
        },
        {
            "name": "equation_output_check",
            "input": [dA, linear_activation_cache, "relu"],
            "expected": expected_output_relu,
            "error": "Wrong output with relu activation",
        },
    ]

    multiple_test(test_cases, _linear_activation_backward)


def test_linear_model_backward():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3, 2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    expected_dA1 = np.array(
        [
            [0.12913162, -0.44014127],
            [-0.14175655, 0.48317296],
            [0.01663708, -0.05670698],
        ]
    )
    expected_dW2 = np.array([[-0.39202432, -0.13325855, -0.04601089]])
    expected_db2 = np.array([[0.15187861]])
    expected_dA0 = np.array(
        [[0.0, 0.52257901], [0.0, -0.3269206], [0.0, -0.32070404], [0.0, -0.74079187]]
    )
    expected_dW1 = np.array(
        [
            [0.41010002, 0.07807203, 0.13798444, 0.10502167],
            [0.0, 0.0, 0.0, 0.0],
            [0.05283652, 0.01005865, 0.01777766, 0.0135308],
        ]
    )
    expected_db1 = np.array([[-0.22007063], [0.0], [-0.02835349]])
    expected_output = {
        "dA1": expected_dA1,
        "dW2": expected_dW2,
        "db2": expected_db2,
        "dA0": expected_dA0,
        "dW1": expected_dW1,
        "db1": expected_db1,
    }
    test_cases = [
        {
            "name": "datatype_check",
            "input": [AL, Y, caches],
            "expected": expected_output,
            "error": "Data type mismatch",
        },
        {
            "name": "shape_check",
            "input": [AL, Y, caches],
            "expected": expected_output,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [AL, Y, caches],
            "expected": expected_output,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, LinearModelBackward().propagate)
