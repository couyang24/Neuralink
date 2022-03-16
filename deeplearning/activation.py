"""Module contains activation functions"""
from deeplearning.base import Baseactivate
import numpy as np


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
