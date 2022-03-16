"""Module contains propagate functions"""
from deeplearning.base import Basepropagate
from deeplearning.activation import Sigmoid, Tanh
import numpy as np


class LogitPropagate(Basepropagate):
    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """

        m = X.shape[1]

        # FORWARD PROPAGATION (FROM X TO COST)
        # compute activation
        A = Sigmoid().activate(np.dot(w.T, X) + b)
        # compute cost by using np.dot to perform multiplication.
        # And don't use loops for the sum.
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = 1 / m * np.dot(X, (A - Y).T)
        db = 1 / m * np.sum(A - Y)

        cost = np.squeeze(np.array(cost))

        grads = {"dw": dw, "db": db}

        return grads, cost

class ForwardPropagate(Basepropagate):
    def propagate(X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(W1, X)+b1
        A1 = Tanh().propagate(Z1)
        Z2 = np.dot(W2, A1)+b2
        A2 = Sigmoid().propagate(Z2)

        assert(A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache
