import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

# Layer 1 [3 neurons]
# INPUTS
X = [[5.6, 7.8, 9.5, 10.8],
     [-2.5, 3.2, 4.9, 9],
     [10.5, -12.8, 0.98, -5.26]]

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

np.random.seed(100)
print(np.random.randn(4, 3))


class Dense_Layer:
    def __init__(self, input, neurons):
        self.weights = np.random.randn(input, neurons)

    def forward(self):
        pass
