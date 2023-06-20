import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import math

# Layer 1 [3 neurons]
# INPUTS
# X = [[5.6, 7.8, 9.5, 10.8],
#      [-2.5, 3.2, 4.9, 9],
#      [10.5, -12.8, 0.98, -5.26]]

# weights = [[-1.001, 0.88, -0.598, 0.784],
#            [0.99, 0.65, 0.598, -0.84],
#            [0.68, -0.88, 0.33, 1.2]]

# bias = [1.22, 2, 2.3]

# layer1 = np.dot(X, np.array(weights).T) + bias
# print(layer1)

# # Secound layer[3 neurons]
# inputs2 = layer1

# weights2 = [[0.3, -0.88, -0.8],
#             [1.29, -1.65, 0.98],
#             [0.88, -0.88, -0.73]]

# bias2 = [3, 2.1, 0.9]

# layer2 = np.dot(inputs2, np.array(weights2).T) + bias2
# print(layer2)

np.random.seed(42)
# X = [[5.6, 7.8, 9.5, 10.8],
#      [-2.5, 3.2, 4.9, 9.01],
#      [10.5, -12.8, 0.98, -5.26]]

# Data Generator[this function gives a spiral dataset]

# https://cs231n.github.io/neural-networks-case-study/


def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


X, y = spiral_data(100, 3)


class Dense_Layer:
    def __init__(self, n_input, neurons, weights):
        self.weights = weights * np.random.randn(n_input, neurons)
        self.bias = np.array([0.23, -0.25, 0.45, 0.93, -0.73])

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

# Activation function-1


class Activation_ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)


layer_1 = Dense_Layer(2, 5, 0.9)
output_1 = layer_1.forward(X)
activation_relu = Activation_ReLU()
activation_relu.forward(output_1)
# print(activation_relu.output)

# Activation function-2
# SoftMax[e^x + Normalisation] [its for output layer.because relu will ignore any '-ve' value but we need that
#  value in our final layer to evalute model performence and properly correct the bias  ]

E = math.e  # 2.718281828459045
exp_val = []
for i in output_1:
    exp_val.append(E**i)

# anothe way
# exp_val = np.exp(output_1)

# Probability distributation(normalising values)
norm_values = exp_val / np.sum(exp_val)


# print(exp_val)
print(norm_values)


# layer_2 = Dense_Layer(5, 3, 1.2)
# output_2 = layer_2.forward(output_1)
