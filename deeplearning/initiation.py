"""Module contains initiation functions"""
import numpy as np

from deeplearning.base import Baseinitialize


class ZeroInitialize(Baseinitialize):
    def initiate(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias) of type float
        """

        w = np.zeros([dim, 1])
        b = float(0)
        return w, b


class RandInitialize(Baseinitialize):
    def initiate(self, n_x, n_h, n_y, seed=1):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        np.random.seed(seed)
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        return parameters


class RandDeepInitialize(Baseinitialize):
    def initiate(self, layer_dims, seed=3, deep=False):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(seed)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network
        demoninator = 100

        for l in range(1, L):
            if deep:
                demoninator = np.sqrt(layer_dims[l - 1])

            parameters["W" + str(l)] = (
                np.random.randn(layer_dims[l], layer_dims[l - 1]) / demoninator
            )
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

            assert parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
            assert parameters["b" + str(l)].shape == (layer_dims[l], 1)

        return parameters
