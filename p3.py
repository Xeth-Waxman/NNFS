## clean up code to make things more...adequate.

# the putputs of the preceeding 'layer' of 4 neurons
inputs = [1, 2, 3, 2.5]

# the weights of the inputs per neuron
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

# the biases of neurons 1, 2 and 3
biases = [2, 3, 0.5]

layer_ouputs = [] #output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 #init the output
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_ouputs.append(neuron_output)

print("Layer Outputs: ", layer_ouputs)
