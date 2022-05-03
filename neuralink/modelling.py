"""Module contains modelling objects"""
import numpy as np

from neuralink.base import Basemodel
from neuralink.cost import Cost
from neuralink.initiation import RandDeepInitialize, RandInitialize, ZeroInitialize
from neuralink.layer import Layer
from neuralink.optimization import LogitOptimize
from neuralink.parameters import Parameters
from neuralink.prediction import LogitPredict
from neuralink.propagation import (
    BackwardPropagate,
    ForwardPropagate,
    LinearModelBackward,
    LinearModelForward,
)


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
    def train(self, X, Y, n_h, num_iterations=10000, print_cost=False, seed=3):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        """

        np.random.seed(seed)
        n_x = Layer().determine(X, Y)[0]
        n_y = Layer().determine(X, Y)[2]

        # Initialize parameters
        parameters = RandInitialize().initiate(n_x, n_h, n_y, seed)

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = ForwardPropagate().propagate(X, parameters)

            # Cost function. Inputs: "A2, Y". Outputs: "cost".
            cost = Cost().compute(A2, Y)

            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = BackwardPropagate().propagate(parameters, cache, X, Y)

            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = Parameters().update(parameters, grads, learning_rate=1.2)

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


class DeepNNModel(Basemodel):
    def train(
        self,
        X,
        Y,
        layers_dims,
        learning_rate=0.0075,
        num_iterations=3000,
        print_cost=False,
        seed=1,
        deep=True,
        lambd=None,
        keep_prob=None,
        parameters=None,
    ):
        """
        Implements a deep neural network: LINEAR->RELU->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations

        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """

        np.random.seed(seed)
        costs = []

        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        if parameters is None:
            parameters = RandDeepInitialize().initiate(
                layers_dims, seed=seed, deep=deep
            )

        # Loop (gradient descent)
        for i in range(num_iterations):
            AL, caches = LinearModelForward().propagate(
                X, parameters, keep_prob=keep_prob
            )

            cost = Cost().compute(AL, Y, deep=True)
            grads = LinearModelBackward().propagate(
                AL, Y, caches, lambd=lambd, keep_prob=keep_prob
            )
            parameters = Parameters().update(
                parameters, grads, learning_rate=learning_rate
            )

            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)

        self.parameters = parameters

        return parameters, costs

    def predict(self, X, y=None, parameters=None, keep_prob=None):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """
        if (parameters is None) and (self.parameters is not None):
            parameters = self.parameters
        elif (parameters is None) and (self.parameters is None):
            raise AttributeError(f"parameters is not provided.")

        m = X.shape[1]
        n = len(parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = LinearModelForward().propagate(
            X, parameters, keep_prob=keep_prob
        )

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        if y is not None:
            print("Accuracy: " + str(np.sum((p == y) / m)))

        return p
