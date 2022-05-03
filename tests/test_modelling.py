"""Testing modelling"""
import numpy as np

from neuralink.modelling import DeepNNModel, LogitModel, NNModel

from .utils import multiple_test


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


def test_deep_nn_model_train1():
    np.random.seed(1)
    n_x = 10
    n_h = 4
    n_y = 1
    num_examples = 10
    num_iterations = 2
    layers_dims = (n_x, n_h, n_y)
    learning_rate = 0.0075
    X = np.random.randn(n_x, num_examples)
    Y = np.random.randn(1, num_examples)

    expected_parameters = {
        "W1": np.array(
            [
                [
                    0.01624965,
                    -0.00610741,
                    -0.00528734,
                    -0.01072836,
                    0.008664,
                    -0.02301103,
                    0.01745639,
                    -0.00760949,
                    0.0031934,
                    -0.00248971,
                ],
                [
                    0.01462848,
                    -0.02057904,
                    -0.00326745,
                    -0.00383625,
                    0.01138176,
                    -0.01097596,
                    -0.00171974,
                    -0.00877601,
                    0.00043022,
                    0.00584423,
                ],
                [
                    -0.01098272,
                    0.01148209,
                    0.00902102,
                    0.00500958,
                    0.00900571,
                    -0.00683188,
                    -0.00123491,
                    -0.00937164,
                    -0.00267157,
                    0.00532808,
                ],
                [
                    -0.00693465,
                    -0.00400047,
                    -0.00684685,
                    -0.00844447,
                    -0.00670397,
                    -0.00014731,
                    -0.01113977,
                    0.00238846,
                    0.0165895,
                    0.00738212,
                ],
            ]
        ),
        "b1": np.array(
            [[1.10437111e-05], [1.78437869e-05], [3.74879549e-05], [-4.42988824e-05]]
        ),
        "W2": np.array([[-0.00200283, -0.00888593, -0.00751122, 0.01688162]]),
        "b2": np.array([[-0.00689018]]),
    }
    expected_costs = [np.array(0.69315968)]

    expected_output1 = (expected_parameters, expected_costs)

    expected_output2 = (
        {
            "W1": np.array(
                [
                    [
                        0.01640028,
                        -0.00585699,
                        -0.00542633,
                        -0.01069332,
                        0.0089055,
                        -0.02290418,
                        0.01765388,
                        -0.00754616,
                        0.00326712,
                        -0.00239159,
                    ],
                    [
                        0.01476737,
                        -0.02014461,
                        -0.0040947,
                        -0.0037457,
                        0.01221714,
                        -0.01054049,
                        -0.00164111,
                        -0.00872507,
                        0.00058592,
                        0.00615077,
                    ],
                    [
                        -0.01051621,
                        0.01216499,
                        0.009119,
                        0.0047126,
                        0.00894761,
                        -0.00672568,
                        -0.00134921,
                        -0.0096428,
                        -0.00253223,
                        0.00580758,
                    ],
                    [
                        -0.00728552,
                        -0.00461461,
                        -0.00638113,
                        -0.00831084,
                        -0.00654136,
                        -0.00053186,
                        -0.01052771,
                        0.00320719,
                        0.01643914,
                        0.00667123,
                    ],
                ]
            ),
            "b1": np.array([[0.00027478], [0.00034477], [0.00076016], [-0.00084497]]),
            "W2": np.array([[-0.00358725, -0.00911995, -0.00831979, 0.01615845]]),
            "b2": np.array([[-0.13451354]]),
        },
        [np.array(0.69315968)],
    )

    test_cases = [
        {
            "name": "datatype_check",
            "input": [
                X,
                Y,
                layers_dims,
                learning_rate,
                num_iterations,
                False,
                1,
                False,
            ],
            "expected": expected_output1,
            "error": "Datatype mismatch.",
        },
        {
            "name": "shape_check",
            "input": [
                X,
                Y,
                layers_dims,
                learning_rate,
                num_iterations,
                False,
                1,
                False,
            ],
            "expected": expected_output1,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [
                X,
                Y,
                layers_dims,
                learning_rate,
                num_iterations,
                False,
                1,
                False,
            ],
            "expected": expected_output1,
            "error": "Wrong output",
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, layers_dims, 0.1, 3, False, 1, False],
            "expected": expected_output2,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, DeepNNModel().train)


def test_deep_nn_model_train2():
    np.random.seed(1)
    n_x = 10
    n_y = 1
    num_examples = 10
    num_iterations = 2
    layers_dims = (n_x, 5, 6, n_y)
    learning_rate = 0.0075
    X = np.random.randn(n_x, num_examples)
    Y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 0]).reshape(1, 10)

    expected_parameters = {
        "W1": np.array(
            [
                [
                    0.51384638,
                    -0.19333098,
                    -0.16705238,
                    -0.33923196,
                    0.273477,
                    -0.72775498,
                    0.55170785,
                    -0.24077478,
                    0.10082452,
                    -0.07882423,
                ],
                [
                    0.46227786,
                    -0.65153639,
                    -0.10192959,
                    -0.12150984,
                    0.35855025,
                    -0.34787253,
                    -0.05455001,
                    -0.27767163,
                    0.01337835,
                    0.1843845,
                ],
                [
                    -0.34790478,
                    0.36200264,
                    0.28511245,
                    0.15868454,
                    0.284931,
                    -0.21645471,
                    -0.03877896,
                    -0.29584578,
                    -0.08480802,
                    0.16760667,
                ],
                [
                    -0.21835973,
                    -0.12531366,
                    -0.21720823,
                    -0.26764975,
                    -0.21214946,
                    -0.00438229,
                    -0.35316347,
                    0.07432144,
                    0.52474685,
                    0.23453653,
                ],
                [
                    -0.06060968,
                    -0.28061463,
                    -0.23624839,
                    0.53526844,
                    0.01597194,
                    -0.20136496,
                    0.06021639,
                    0.66414167,
                    0.03804666,
                    0.19528599,
                ],
            ]
        ),
        "b1": np.array(
            [
                [-2.16491028e-04],
                [1.50999130e-04],
                [8.71516045e-06],
                [5.57557615e-05],
                [-2.90746349e-05],
            ]
        ),
        "W2": np.array(
            [
                [0.13428358, -0.15747685, -0.51095667, -0.15624083, -0.09342034],
                [0.26226685, 0.3751336, 0.41644174, 0.12779375, 0.39573817],
                [-0.33726917, 0.56041154, 0.22939257, -0.1333337, 0.21851314],
                [-0.03377599, 0.50617255, 0.67960046, 0.97726521, -0.62458844],
                [-0.64581803, -0.22559264, 0.0715349, 0.39173682, 0.14112904],
                [-0.9043503, -0.13693179, 0.37026002, 0.10284282, 0.34076545],
            ]
        ),
        "b2": np.array(
            [
                [1.80215514e-07],
                [-1.07935097e-04],
                [1.63081605e-04],
                [-3.51202008e-05],
                [-7.40012619e-05],
                [-4.43814901e-05],
            ]
        ),
        "W3": np.array(
            [[-0.09079199, -0.08117381, 0.07667568, 0.16665535, 0.08029575, 0.04805811]]
        ),
        "b3": np.array([[0.0013201]]),
    }
    expected_costs = [np.array(0.70723944)]

    expected_output1 = (expected_parameters, expected_costs)
    expected_output2 = (
        {
            "W1": np.array(
                [
                    [
                        0.51439065,
                        -0.19296367,
                        -0.16714033,
                        -0.33902173,
                        0.27291558,
                        -0.72759069,
                        0.55155832,
                        -0.24095201,
                        0.10063293,
                        -0.07872596,
                    ],
                    [
                        0.46203186,
                        -0.65172685,
                        -0.10184775,
                        -0.12169458,
                        0.35861847,
                        -0.34804029,
                        -0.05461748,
                        -0.27787524,
                        0.01346693,
                        0.18463095,
                    ],
                    [
                        -0.34748255,
                        0.36202977,
                        0.28512463,
                        0.1580327,
                        0.28509518,
                        -0.21717447,
                        -0.03853304,
                        -0.29563725,
                        -0.08509025,
                        0.16728901,
                    ],
                    [
                        -0.21727997,
                        -0.12486465,
                        -0.21692552,
                        -0.26875722,
                        -0.21180188,
                        -0.00550575,
                        -0.35268367,
                        0.07489501,
                        0.52436384,
                        0.23418418,
                    ],
                    [
                        -0.06045008,
                        -0.28038304,
                        -0.23617868,
                        0.53546925,
                        0.01569291,
                        -0.20115358,
                        0.05975429,
                        0.66409149,
                        0.03819309,
                        0.1956102,
                    ],
                ]
            ),
            "b1": np.array(
                [
                    [-8.61228305e-04],
                    [6.08187689e-04],
                    [3.53075377e-05],
                    [2.21291877e-04],
                    [-1.13591429e-04],
                ]
            ),
            "W2": np.array(
                [
                    [0.13441428, -0.15731437, -0.51097778, -0.15627102, -0.09342034],
                    [0.2620349, 0.37492336, 0.4165605, 0.12801536, 0.39541677],
                    [-0.33694339, 0.56075022, 0.22940292, -0.1334017, 0.21863717],
                    [-0.03371679, 0.50644769, 0.67935577, 0.97680859, -0.62475679],
                    [-0.64579072, -0.22555897, 0.07142896, 0.3914475, 0.14104814],
                    [-0.90433399, -0.13691167, 0.37019673, 0.10266999, 0.34071712],
                ]
            ),
            "b2": np.array(
                [
                    [1.18811550e-06],
                    [-4.25510194e-04],
                    [6.56178455e-04],
                    [-1.42335482e-04],
                    [-2.93618626e-04],
                    [-1.75573157e-04],
                ]
            ),
            "W3": np.array(
                [
                    [
                        -0.09087434,
                        -0.07882982,
                        0.07821609,
                        0.16442826,
                        0.0783229,
                        0.04648216,
                    ]
                ]
            ),
            "b3": np.array([[0.00525865]]),
        },
        [np.array(0.70723944)],
    )

    test_cases = [
        {
            "name": "equation_output_check",
            "input": [X, Y, layers_dims, learning_rate, num_iterations],
            "expected": expected_output1,
            "error": "Wrong output",
        },
        {
            "name": "datatype_check",
            "input": [X, Y, layers_dims, learning_rate, num_iterations],
            "expected": expected_output1,
            "error": "Datatype mismatch.",
        },
        {
            "name": "shape_check",
            "input": [X, Y, layers_dims, learning_rate, num_iterations],
            "expected": expected_output1,
            "error": "Wrong shape",
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, layers_dims, 0.02, 3],
            "expected": expected_output2,
            "error": "Wrong output",
        },
    ]

    multiple_test(test_cases, DeepNNModel().train)
