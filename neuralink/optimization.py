"""Module contains optimization functions"""
import copy

import numpy as np

from neuralink.base import Baseoptimize
from neuralink.propagation import LogitPropagate


class LogitOptimize(Baseoptimize):
    def optimize(
        self, w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False
    ):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """

        w = copy.deepcopy(w)
        b = copy.deepcopy(b)

        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = LogitPropagate().propagate(w, b, X, Y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

                # Print the cost every 100 training iterations
                if print_cost:
                    print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w, "b": b}

        grads = {"dw": dw, "db": db}

        return params, grads, costs
