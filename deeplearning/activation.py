"""Module contains activation functions"""
import numpy as np

from deeplearning.base import Baseactivate


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


class Tanh(Baseactivate):
    def activate(self, z):
        return np.tanh(z)


class Relu(Baseactivate):
    def activate(self, z):
        return z.clip(min=0)
