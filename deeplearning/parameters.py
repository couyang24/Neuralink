"""Module contains update parameters functions"""
from deeplearning.base import Baseparams
import copy


class Params(Baseparams):
    def update(self, parameters, grads, learning_rate=1.2):
        """
        Updates parameters using the gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary
        """
        # Retrieve a copy of each parameter from the dictionary "parameters". Use copy.deepcopy(...) for W1 and W2
        W1 = copy.deepcopy(parameters["W1"])
        b1 = copy.deepcopy(parameters["b1"])
        W2 = copy.deepcopy(parameters["W2"])
        b2 = copy.deepcopy(parameters["b2"])

        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        # Update rule for each parameter
        W1 = W1 - dW1 * learning_rate
        b1 = b1 - db1 * learning_rate
        W2 = W2 - dW2 * learning_rate
        b2 = b2 - db2 * learning_rate

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        return parameters
