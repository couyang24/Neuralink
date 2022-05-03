"""Module contains cost functions"""
import numpy as np

from neuralink.base import Basecost


class Cost(Basecost):
    def compute(self, AL, Y, parameters=None, lambd=0.1, deep=False):
        """
        Computes the cross-entropy cost

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]  # number of examples

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        cost = -1 / m * np.sum(logprobs)

        if deep:
            cost = np.array(np.squeeze(cost))
        else:
            cost = float(
                np.squeeze(cost)
            )  # makes sure cost is the dimension we expect.
        # E.g., turns [[17]] into 17

        if parameters is not None:
            W1 = parameters["W1"]
            W2 = parameters["W2"]
            W3 = parameters["W3"]
            L2_regularization_cost = (
                (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
                * lambd
                / 2
                / m
            )
            cost = cost + L2_regularization_cost

        return cost
