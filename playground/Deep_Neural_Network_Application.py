# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Deep Neural Network for Image Classification: Application
#
# - Build and train a deep L-layer neural network, and apply it to supervised learning

# ## Table of Contents
# - [1 - Packages](#1)
# - [2 - Load and Process the Dataset](#2)
# - [3 - Model Architecture](#3)
#     - [3.1 - 2-layer Neural Network](#3-1)
#     - [3.2 - L-layer Deep Neural Network](#3-2)
#     - [3.3 - General Methodology](#3-3)
# - [4 - Two-layer Neural Network](#4)
# - [5 - L-layer Neural Network](#5)
# - [6 - Results Analysis](#6)
# - [7 - Test with images](#7)

# <a name='1'></a>
# ## 1 - Packages

# - [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
# - [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.

# +
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

from deeplearning.modelling import DeepNNModel

# %matplotlib inline
plt.rcParams["figure.figsize"] = (5.0, 4.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# %load_ext autoreload
# %autoreload 2

np.random.seed(1)
# -

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075


# <a name='2'></a>
# ## 2 - Load and Process the Dataset
#
# **Problem Statement**:
#     - a training set of `m_train` images labelled as cat (1) or non-cat (0)
#     - a test set of `m_test` images labelled as cat and non-cat
#     - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).

# +
def load_data():
    train_dataset = h5py.File("../datasets/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("../datasets/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams["figure.figsize"] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation="nearest")
        plt.axis("off")
        plt.title(
            "Prediction: "
            + classes[int(p[0, index])].decode("utf-8")
            + " \n Class: "
            + classes[y[0, index]].decode("utf-8")
        )


# -

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print(
    "y = "
    + str(train_y[0, index])
    + ". It's a "
    + classes[train_y[0, index]].decode("utf-8")
    + " picture."
)

# +
# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

# +
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(
    train_x_orig.shape[0], -1
).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
# -

# <a name='3'></a>
# ## 3 - Model Architecture

# <a name='3-1'></a>
# ### 3.1 - 2-layer Neural Network
#
# Build a deep neural network to distinguish cat images from non-cat images!
#
# Build two different models:
#
# - A 2-layer neural network
# - An L-layer deep neural network
#
# Then, compare the performance of these models, and try out some different values for $L$. 
#
# - The input is a (64,64,3) image which is flattened to a vector of size $(12288,1)$. 
# - The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ of size $(n^{[1]}, 12288)$.
# - Then, add a bias term and take its relu to get the following vector: $[a_0^{[1]}, a_1^{[1]},..., a_{n^{[1]}-1}^{[1]}]^T$.
# - Repeat the same process.
# - Multiply the resulting vector by $W^{[2]}$ and add the intercept (bias). 
# - Finally, take the sigmoid of the result. If it's greater than 0.5, classify it as a cat.
#
# <a name='3-2'></a>
# ### 3.2 - L-layer Deep Neural Network
#
# It's pretty difficult to represent an L-layer deep neural network using the above representation. However, here is a simplified network representation:
#
# - The input is a (64,64,3) image which is flattened to a vector of size (12288,1).
# - The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ and then you add the intercept $b^{[1]}$. The result is called the linear unit.
# - Next, take the relu of the linear unit. This process could be repeated several times for each $(W^{[l]}, b^{[l]})$ depending on the model architecture.
# - Finally, take the sigmoid of the final linear unit. If it is greater than 0.5, classify it as a cat.
#
# <a name='3-3'></a>
# ### 3.3 - General Methodology
#
# 1. Initialize parameters / Define hyperparameters
# 2. Loop for num_iterations:
#     a. Forward propagation
#     b. Compute cost function
#     c. Backward propagation
#     d. Update parameters (using parameters, and grads from backprop) 
# 3. Use trained parameters to predict labels

# <a name='4'></a>
# ## 4 - Two-layer Neural Network

model = DeepNNModel()
parameters, costs = model.train(
    train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2, print_cost=False
)
print("Cost after first iteration: " + str(costs[0]))

parameters, costs = model.train(
    train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True
)
plot_costs(costs, learning_rate)

parameters

predictions_train = model.predict(train_x, train_y)

predictions_test = model.predict(test_x, test_y)

# <a name='5'></a>
# ## 5 - L-layer Neural Network

### CONSTANTS ###
layers_dims = [12288, 64, 32, 16, 1]  #  4-layer model

# +
model = DeepNNModel()
parameters, costs = model.train(
    train_x,
    train_y,
    layers_dims,
    num_iterations=1,
    print_cost=False,
    lambd=6,
    keep_prob=0.5,
    parameters=parameters,
)

print("Cost after first iteration: " + str(costs[0]))
# -

model = DeepNNModel()
parameters, costs = model.train(
    train_x,
    train_y,
    layers_dims,
    num_iterations=1400,
    print_cost=True,
    lambd=4,
    keep_prob=0.94,
    parameters=parameters,
)

pred_train = model.predict(train_x, train_y, parameters)

pred_test = model.predict(test_x, test_y, parameters)

# <a name='6'></a>
# ##  6 - Results Analysis
#
# First, take a look at some images the L-layer model labeled incorrectly. This will show a few mislabeled images. 

print_mislabeled_images(classes, test_x, test_y, pred_test)


# **A few types of images the model tends to do poorly on include:** 
# - Cat body in an unusual position
# - Cat appears against a background of a similar color
# - Unusual cat color and species
# - Camera Angle
# - Brightness of the picture
# - Scale variation (cat is very large or small in image) 

# <a name='7'></a>
# ## 7 - Test with images

def test_image(my_image, parameters):

    fname = "../images/" + my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.0
    image = image.reshape((1, num_px * num_px * 3)).T

    model = DeepNNModel()
    my_predicted_image = model.predict(image, parameters=parameters)

    print(
        "y = "
        + str(np.squeeze(my_predicted_image))
        + ', your L-layer model predicts a "'
        + classes[
            int(np.squeeze(my_predicted_image)),
        ].decode("utf-8")
        + '" picture.'
    )


test_image("my_image.jpg", parameters)

test_image("my_image2.jpg", parameters)

test_image("1561040958920.jpg", parameters)


