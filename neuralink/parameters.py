"""Module contains update parameters functions"""
import copy

from neuralink.base import Baseparams


class Parameters(Baseparams):
    def update(self, params, grads, learning_rate=1.2):
        """
        Update parameters using gradient descent

        Arguments:
        params -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """
        parameters = params.copy()
        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l + 1)] = (
                parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            )
            parameters["b" + str(l + 1)] = (
                parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            )

        return parameters
