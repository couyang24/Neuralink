"""Module contains propagate functions"""
import numpy as np

from neuralink.activation import Relu, ReluBackward, Sigmoid, SigmoidBackward, Tanh
from neuralink.base import Basepropagate


def _linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def _linear_backward(dZ, cache, lambd=None):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    lambd

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    if lambd is not None:
        regu = W * lambd / m
    else:
        regu = 0

    dW = 1 / m * np.dot(dZ, A_prev.T) + regu
    db = (
        1 / m * np.sum(dZ, axis=1, keepdims=True)
    )  # sum by the rows of dZ with keepdims=True
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def _linear_activation_forward(A_prev, W, b, activation, keep_prob=None):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    Z, linear_cache = _linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A = Sigmoid().activate(Z)
    elif activation == "relu":
        A = Relu().activate(Z)
    elif activation == "tanh":
        A = Tanh().activate(Z)
    else:
        raise NotImplementedError("Only sigmoid, tanh, and relu are implemented")

    if keep_prob is not None:
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = A * D
        A = A / keep_prob
        cache = (linear_cache, Z, D)
    else:
        cache = (linear_cache, Z)

    return A, cache


def _linear_activation_backward(dA, cache, activation, lambd=None, keep_prob=None):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # print(cache[2])
    if keep_prob is not None:
        linear_cache, activation_cache, D = cache
    else:
        linear_cache, activation_cache = cache

    if keep_prob is not None:
        dA = dA * D / keep_prob

    if activation == "relu":
        dZ = ReluBackward().activate(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = SigmoidBackward().activate(dA, activation_cache)

    dA_prev, dW, db = _linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db


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
        Z, _ = _linear_forward(X, w.T, b)

        # FORWARD PROPAGATION (FROM X TO COST)
        # compute activation
        A = Sigmoid().activate(Z)
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
    def propagate(self, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1, _ = _linear_forward(X, W1, b1)
        A1 = Tanh().activate(Z1)
        Z2, _ = _linear_forward(A1, W2, b2)
        A2 = Sigmoid().activate(Z2)

        assert A2.shape == (1, X.shape[1])

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2, cache


class BackwardPropagate(Basepropagate):
    def propagate(self, parameters, cache, X, Y, lambd=None):
        """
        Implement the backward propagation

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary
        """
        m = X.shape[1]

        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]

        if lambd is not None:
            regu1 = W1 * lambd / m
            regu2 = W2 * lambd / m
        else:
            regu1 = 0
            regu2 = 0

        # Backward propagation: calculate dW1, db1, dW2, db2.
        if "W3" in parameters:
            if lambd is not None:
                regu3 = W3 * lambd / m
            else:
                regu3 = 0
            W3 = parameters["W3"]
            A3 = cache["A3"]
            dZ3 = A3 - Y
            dW3 = 1 / m * np.dot(dZ3, A2.T) + regu3
            db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
            dZ2 = np.dot(W3.T, dZ3) * (1 - np.power(A2, 2))
            dW2 = 1 / m * np.dot(dZ2, X.T) + regu2
            db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
            dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
            dW1 = 1 / m * np.dot(dZ1, X.T) + regu1
            db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
            grads = {
                "dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2,
                "dW3": dW3,
                "db3": db3,
            }
        else:
            dZ2 = A2 - Y
            dW2 = 1 / m * np.dot(dZ2, A1.T) + regu2
            db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
            dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
            dW1 = 1 / m * np.dot(dZ1, X.T) + regu1
            db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

            grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return grads


class LinearModelForward(Basepropagate):
    def propagate(self, X, parameters, keep_prob=None, seed=1):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        keep_prob - probability of keeping a neuron active during drop-out, scalar

        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """
        np.random.seed(seed)

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        # The for loop starts at 1 because layer 0 is the input
        for l in range(1, L):
            A_prev = A
            A, cache = _linear_activation_forward(
                A_prev, parameters[f"W{l}"], parameters[f"b{l}"], "relu", keep_prob
            )
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = _linear_activation_forward(
            A, parameters[f"W{l+1}"], parameters[f"b{l+1}"], "sigmoid", keep_prob=None
        )
        caches.append(cache)

        return AL, caches


class LinearModelBackward(Basepropagate):
    def propagate(self, AL, Y, caches, lambd=None, keep_prob=None):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        lambd -- regularization hyperparameter, scalar with default None
        keep_prob - probability of keeping a neuron active during drop-out, scalar

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = _linear_activation_backward(
            dAL, current_cache, "sigmoid", lambd=lambd, keep_prob=None
        )
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = _linear_activation_backward(
                dA_prev_temp,
                current_cache,
                "relu",
                lambd=lambd,
                keep_prob=keep_prob,
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
