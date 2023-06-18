import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


input = [5.6, 7.8, 9.5, 10.8]
weights1 = [-1.001, 0.88, -0.598, 0.784]
weights2 = [0.99, 0.65, 0.598, -0.84]
weights3 = [0.68, -0.88, 0.33, 1.2]
bias1 = 1.22
bias2 = 2
bias3 = 2.3

# for last fully connected layer which has three output neuron and takes input from 4 hidden neuron


output = [input[0]*weights1[0]+input[1]*weights1[1] + input[2]*weights1[2]+input[3]*weights1[3] + bias1,
          input[0]*weights2[0]+input[1]*weights2[1] +
          input[2]*weights2[2]+input[3]*weights2[3] + bias2,
          input[0]*weights3[0]+input[1]*weights3[1] + input[2]*weights3[2]+input[3]*weights3[3] + bias3]
value = [round(num, 4) for num in output]
print(value)
