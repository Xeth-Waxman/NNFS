# This file now models a single neuron layer: three neurons, each with four input,
# to represent a 4-into-3 layer transfer
# NOTE: this code is awful, we could be using loops or (even better) numpy for
# code that is pythonic, or at least not kindergarten-quality. However, we WANT
# it to be kindergarten quality here so we can clearly demonstrate what is happening
# with this layer. neural networks are complex unless you make them simple.

# the putputs of the preceeding 'layer' of 4 neurons
inputs = [1, 2, 3, 2.5]

# the weights of the inputs per neuron
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

# the bias of the three neurons we're modeling
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
    inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
    inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print("Output: ", output)
