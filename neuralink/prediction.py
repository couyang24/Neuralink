"""Module contains predicition functions"""
import numpy as np

from neuralink.activation import Sigmoid
from neuralink.base import Basepredict


class LogitPredict(Basepredict):
    def predict(self, w, b, X):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        """

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = Sigmoid().activate(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):

            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0, i] > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0

        return Y_prediction
