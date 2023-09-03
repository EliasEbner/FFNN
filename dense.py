import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.inputs = np.empty((input_size, 1))
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.biases

    def backward(self, loss_gradients, learning_rate):
        weight_gradients = np.dot(loss_gradients, self.inputs.T)
        self.weights -= weight_gradients * learning_rate
        self.biases -= loss_gradients * learning_rate
        return np.dot(loss_gradients, self.weights.T)
