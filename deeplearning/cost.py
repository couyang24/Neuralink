"""Module contains cost functions"""
import numpy as np

from deeplearning.base import Basecost


class Cost(Basecost):
    def compute(self, A2, Y):
        """
        Computes the cross-entropy cost

        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost

        """

        m = Y.shape[1]  # number of examples

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
        cost = -1 / m * np.sum(logprobs)

        cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
        # E.g., turns [[17]] into 17

        return cost
