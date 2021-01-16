import numpy as np

'''
now let's do things with numpy
'''

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(weights, inputs) + bias
print ("Output: ", output)
