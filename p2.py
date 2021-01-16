# This file models a single neuron and is meant to show the basics
# of neural network connections, and how values move from layer to layer

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print("Output: ", output)
