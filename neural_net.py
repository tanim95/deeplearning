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
X_demo = np.array([[5.6, 7.8, 9.5, 10.8],
                  [-2.5, 3.2, 4.9, 9.01],
                   [10.5, -12.8, 0.98, -5.26]])

# print(list(range(5)))
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


class Dense_Layer:
    def __init__(self, n_input, neurons):
        self.weights = np.random.randn(n_input, neurons)
        self.bias = np.zeros(neurons)

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

# Activation function-1


class Activation_ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)
# Activation function-2
# SoftMax[e^x + Normalisation] [its for output layer.because relu will ignore any '-ve' value but we need that
#  value in our final layer to evalute model performence and properly correct the bias  ]


class Activation_SoftMax:
    def forward(self, input):
        # substracting max value so that e^largeNumber won't cause explode of output!
        exp_val = np.exp(input - np.max(input, axis=1, keepdims=True))
        probability = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probability
# Loss Function


class Loss:
    def calculate(self, y, output):
        loss = self.forward(y, output)
        mean_loss = np.mean(loss)
        return mean_loss


class CategoricalCorssEntropy(Loss):
    def forward(self, y_true, y_pred):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, math.e ** -7, 1 - math.e ** -7)
        if len(y_true.shape) == 1:  # if the y = [0,1] this shape
            confidence = y_pred_clip[list(range(samples)), y_true]
        if len(y_true.shape) == 2:  # if the y = [[1,0],[0,1]],one-hot encoded
            confidence = np.sum(y_pred_clip * y_true, axis=1)
        likelihood = -np.log(confidence)
        return likelihood


X, y = spiral_data(100, 3)

layer_1 = Dense_Layer(2, 3)
activation_relu = Activation_ReLU()
layer_2 = Dense_Layer(3, 3)
activation_smax = Activation_SoftMax()
output_1 = layer_1.forward(X)
output_2 = layer_2.forward(output_1)

activation_relu.forward(output_1)
activation_smax.forward(output_2)
# print(activation_smax.output)

loss_function = CategoricalCorssEntropy()
loss = loss_function.calculate(y, activation_smax.output)
print(loss)
