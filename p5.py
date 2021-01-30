import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# We're using nnfs to generate a random number with a pre-defined seed so
# we can follow-along and get the same results
nnfs.init()

# we use nnfs.datasets to create a spiral dataset of 100 points, with 3 classes
X, y = spiral_data(100, 3)

class Layer_Dense:
    # initialize our weights and biases, scaled within 1 and -1
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer_1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer_1.forward(X)
activation1.forward(layer_1.output)

print(activation1.output)
