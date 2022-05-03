"""test activation"""
import numpy as np

from neuralink.activation import Relu, Sigmoid, Tanh


def test_sigmoid():
    """Testing Sigmoid"""
    x = np.array([0, 2])
    output = Sigmoid().activate(x)
    assert type(output) == np.ndarray, "Wrong type. Expected np.ndarray"
    assert np.allclose(
        output, [0.5, 0.88079708]
    ), f"Wrong value. {output} != [0.5, 0.88079708]"
    output = Sigmoid().activate(1)
    assert np.allclose(output, 0.7310585), f"Wrong value. {output} != 0.7310585"

    np.testing.assert_allclose(
        Sigmoid().activate(np.array([-1, 0, 2])),
        np.array([0.26894142, 0.5, 0.88079708]),
    )


def test_tanh():
    """Testing Tanh"""
    np.testing.assert_allclose(
        Tanh().activate(np.array([-1, 0, 2])), np.array([-0.76159416, 0, 0.96402758])
    )


def test_relu():
    """Testing Relu"""
    np.testing.assert_allclose(
        Relu().activate(np.array([-1, 0, 2])), np.array([0, 0, 2])
    )
