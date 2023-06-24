# Neural Network

This repository contains an implementation of a simple neural network using Python and NumPy. The neural network is designed to perform classification tasks on spiral data.

## Overview

The neural network consists of the following components:

1. `Dense_Layer`: Represents a fully connected layer in the network. It includes methods for forward propagation and backward propagation.

2. `Activation_ReLU`: Implements the ReLU activation function. It applies the ReLU function element-wise to the input data.

3. `Activation_SoftMax`: Implements the Softmax activation function. It computes the Softmax probabilities for the input data.

4. `Loss`: Represents a loss function for evaluating the network's performance. It includes a method for calculating the loss.

5. `CategoricalCrossEntropy`: Inherits from the `Loss` class and implements the categorical cross-entropy loss function.

6. `Activation_Softmax_Loss_CategoricalCrossEntropy`: Combines the Softmax activation function and the categorical cross-entropy loss function.

7. `Optimizer_SGD`: Implements the Stochastic Gradient Descent (SGD) optimizer with support for learning rate decay and momentum.

## Usage

To use the neural network, follow these steps:

1. Import the necessary classes from the code.

2. Create an instance of the `Dense_Layer` class to define the network architecture.

3. Choose an activation function and loss function based on your task and create instances of those classes.

4. Instantiate the `Optimizer_SGD` class with the desired learning rate, decay, and momentum values.

5. Iterate through the desired number of epochs and perform the following steps:

   - Forward propagate the input data through the layers.
   - Calculate the loss using the loss function.
   - Backward propagate the gradients through the layers.
   - Update the parameters using the optimizer.

6. Monitor the accuracy and loss during training to evaluate the network's performance.

## Example

```python
# Import the necessary classes
from neural_network import Dense_Layer, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossEntropy, Optimizer_SGD

# Create the network architecture
layer_1 = Dense_Layer(2, 64)
activation_relu = Activation_ReLU()
layer_2 = Dense_Layer(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    # Perform forward propagation
    layer_1.forward(X)
    activation_relu.forward(layer_1.output)
    layer_2.forward(activation_relu.output)
    loss = loss_activation.forward(layer_2.output, y)

    # Perform backward propagation
    loss_activation.backward(loss_activation.output, y)
    layer_2.backward(loss_activation.dinputs)
    activation_relu.backward(layer_2.dinputs)
    layer_1.backward(activation_relu.dinputs)

    # Update the parameters
    optimizer.pre_update_params()
    optimizer.update_params(layer_1)
    optimizer.update_params(layer_2)
    optimizer.post_update_params()

    # Evaluate accuracy and loss
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    print(f"Epoch: {epoch+1}  Accuracy: {accuracy}  Loss: {loss}")

Dependencies:
-NumPy: For numerical computing and array operations.
```
