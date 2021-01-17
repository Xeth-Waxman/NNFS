import numpy as np

'''
We're going to be using np.random() to do things like
initialize our layers' weights and biases. However, because we want our work
to be reproducible, we're going to use a defined seed
'''
np.random.seed(0)

'''
now let's do things with numpy
an entire layer this time
'''

# let's batch our inputs to make this more really-real
X =    [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    # initialize our weights and biases, scaled within 1 and -1
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer_1 = Layer_Dense(4, 5)
layer_2 = Layer_Dense(5, 2)

layer_1.forward(X)
# print("Layer_1 output: ", layer_1.output)
layer_2.forward(layer_1.output)
print(layer_2.output)



'''
This is all obsolete
# this is now layer 1
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# and we introduce layer 2
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

#transpose the array, but to do so it needs to be a numpy array first
# this is now the output of layer 1
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# the output of layer 1 become the inputs to layer 2
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print("Layer 2 Output: ", layer2_outputs)
'''
