"""Module contains activation functions"""
import numpy as np

from neuralink.base import Baseactivate


class Sigmoid(Baseactivate):
    def activate(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        return 1 / (1 + np.exp(-z))


class SigmoidBackward(Baseactivate):
    def activate(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert dZ.shape == Z.shape

        return dZ


class Tanh(Baseactivate):
    def activate(self, z):
        return np.tanh(z)


class Relu(Baseactivate):
    def activate(self, z):
        return z.clip(min=0)


class ReluBackward(Baseactivate):
    def activate(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

        assert dZ.shape == Z.shape

        return dZ
