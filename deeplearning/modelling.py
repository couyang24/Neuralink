"""Module contains modelling objects"""
from deeplearning.base import Basemodel
from deeplearning.cost import Cost
from deeplearning.initiation import ZeroInitialize, RandInitialize
from deeplearning.layer import Layer
from deeplearning.optimization import LogitOptimize
from deeplearning.parameters import Params
from deeplearning.prediction import LogitPredict
from deeplearning.propagation import ForwardPropagate, BackwardPropagate
import numpy as np


class LogitModel(Basemodel):
    def train(
        self, X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False
    ):
        """
        Builds the logistic regression model

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to True to print the cost every 100 iterations
        """
        # initialize parameters with zeros
        w, b = ZeroInitialize().initiate(X_train.shape[0])

        # Gradient descent
        params, grads, costs = LogitOptimize().optimize(
            w, b, X_train, Y_train, num_iterations, learning_rate, print_cost
        )

        self.params = params
        self.grads = grads
        self.costs = costs
        self.X_train = X_train
        self.Y_train = Y_train
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def predict(self, X_test, Y_test, print_cost=False):
        """
        Predict with the logistic regression model

        Arguments:
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        print_cost -- Set to True to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """
        # Retrieve parameters w and b from dictionary "params"
        w = self.params["w"]
        b = self.params["b"]

        # Predict test/train set examples
        Y_prediction_train = LogitPredict().predict(w, b, self.X_train)
        Y_prediction_test = LogitPredict().predict(w, b, X_test)

        # Print train/test Errors
        if print_cost:
            print(
                "train accuracy: {} %".format(
                    100 - np.mean(np.abs(Y_prediction_train - self.Y_train)) * 100
                )
            )
            print(
                "test accuracy: {} %".format(
                    100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
                )
            )

        d = {
            "costs": self.costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations,
        }

        return d


class NNModel(Basemodel):
    def train(self, X, Y, n_h, num_iterations=10000, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        """

        np.random.seed(3)
        n_x = Layer().determine(X, Y)[0]
        n_y = Layer().determine(X, Y)[2]

        # Initialize parameters
        parameters = RandInitialize().initiate(n_x, n_h, n_y)

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = ForwardPropagate().propagate(X, parameters)

            # Cost function. Inputs: "A2, Y". Outputs: "cost".
            cost = Cost().compute(A2, Y)

            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = BackwardPropagate().propagate(parameters, cache, X, Y)

            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = Params().update(parameters, grads, learning_rate=1.2)

            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        self.parameters = parameters

    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = ForwardPropagate().propagate(X, self.parameters)
        predictions = A2 > 0.5

        return predictions
