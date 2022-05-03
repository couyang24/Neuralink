"""Module contains layer functions"""
from neuralink.base import Baselayer


class Layer(Baselayer):
    def determine(self, X, Y, hidden_layer=4):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)

        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        n_x = X.shape[0]
        n_h = hidden_layer
        n_y = Y.shape[0]

        return (n_x, n_h, n_y)
