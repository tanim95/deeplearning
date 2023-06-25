import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys


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
    for n in range(classes):
        ix = range(points*n, points*(n+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(n*4, (n+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = n
    return X, y


X, y = spiral_data(100, 3)
# SHOWING THE DATA
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Spiral Data')
plt.colorbar()
# plt.show()


class Dense_Layer:
    def __init__(self, n_input, neurons):
        self.weights = np.random.randn(n_input, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, input):
        self.inputs = input
        self.output = np.dot(input, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# Activation function-1


class Activation_ReLU:
    def forward(self, input):
        # remember inputs
        self.input = input
        self.output = np.maximum(0, input)

    def backward(self, dvalues):
        # as we need to modify original variable first make a copy
        self.dinputs = dvalues.copy()
        # zero gradient where input values negative
        self.dinputs[self.input <= 0] = 0


# Activation function-2
# SoftMax[e^x + Normalisation] [its for output layer.because relu will ignore any '-ve' value but we need that
#  value in our final layer to evalute model performence and properly correct biases  ]


class Activation_SoftMax:
    def forward(self, input):
        self.inputs = input
        # substracting max value so that e^largeNumber won't cause explode of output!
        exp_val = np.exp(input - np.max(input, axis=1, keepdims=True))
        probability = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probability

    def backward(self, dvalues):
        self.dinput = np.empty_like(dvalues)
        # ENUMURATING OUTPUT AND GRADIANT
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # FLATTENING OUTPUT ARRAY
            single_output = single_output.reshape(-1, 1)
            # CALCULATING JACOBIAN MATRIX OF THE OUTPUT
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            # CALCULATING SAMPLE-WISE GRADIANT AND ADD IT TO THE SMAPEL GRADIENTS
            self.dinput[index] = np.dot(jacobian_matrix, single_dvalues)


# Loss Function
class Loss:
    def calculate(self, output, y):
        loss = self.forward(output, y)
        mean_loss = np.mean(loss)
        return mean_loss


class CategoricalCorssEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, math.e ** -7, 1 - math.e ** -7)
        if len(y_true.shape) == 1:  # if the y = [0,1] this shape
            confidence = y_pred_clip[list(range(samples)), y_true]
        if len(y_true.shape) == 2:  # if the y = [[1,0],[0,1]],one-hot encoded
            confidence = np.sum(y_pred_clip * y_true, axis=1)

        # LOSSES
        likelihood = -np.log(confidence)
        return likelihood

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        # IF LABELS ARE SPARSE TURN THEM INTO ONE HOT ENCODER
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # CALCULATE GRADIENT
        self.dinput = -y_true / dvalues
        # NORMALISE GRADIENT
        self.dinput = self.dinput / samples


class Activation_Softmax_Loss_CategoricalCrossEntropy():
    # CREATING ACTIVATION AND LOSS FUNCTION OBJECTS
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = CategoricalCorssEntropy()

    def forward(self, inputs, y_true):
        # OUTPUT LAYER ACTIVATION
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # IF LABELS ARE ONE-HOT ENCODED TURN THEM INTO DISCRETE VALUES
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        # CALCULATE GRADIENT(penalising the correct class by substracting 1)
        self.dinputs[range(samples), y_true] -= 1
        # NORMALISE GRADIENT
        self.dinputs = self.dinputs / samples

# OPTIMISER(Stochastic Gradient Descent)


# SGD optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # CALLING ONCE BEFORE ANY PARAMTER UPDATE
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # UPDATING PARAMETERS
    def update_params(self, layer):
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Building weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Building bias updates
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Updating weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Calling once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# /////////////////TRAINING IN LOOP ///////////////////////////////////////////////////
layer_1 = Dense_Layer(2, 64)  # Input Layer
activation_relu = Activation_ReLU()
layer_2 = Dense_Layer(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimiser = Optimizer_SGD(decay=1e-3, momentum=0.7)

# sys.exit()
for epoch in range(10001):

    layer_1.forward(X)
    activation_relu.forward(layer_1.output)
    layer_2.forward(activation_relu.output)
    loss = loss_activation.forward(layer_2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    # if len(y.shape) == 2:
    #     y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 500:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimiser.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    layer_2.backward(loss_activation.dinputs)
    activation_relu.backward(layer_2.dinputs)
    layer_1.backward(activation_relu.dinputs)

    # Update weights and biases
    optimiser.pre_update_params()
    optimiser.update_params(layer_1)
    optimiser.update_params(layer_2)
    optimiser.post_update_params()


# WE GET [epoch: 10000, acc: 0.940, loss: 0.135, lr: 0.09091735612328393]
# AND , WE GET, [epoch: 100000, acc: 0.953, loss: 0.106, lr: 0.009901088129585442]
