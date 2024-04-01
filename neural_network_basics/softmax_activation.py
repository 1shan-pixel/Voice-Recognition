#working on softmax_activation functions

import nnfs 

from nnfs.datasets import spiral_data
nnfs.init() 

X, y = spiral_data(100, 3)

import numpy as np 
class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) # we want a 2d array with 1 row and n_neurons columns., i don't know how removin the brackets doesnt make this a 2d array or something but whatever. 
    def forward_pass(self,input):
        self.output = np.dot(input, self.weights) + self.biases 
    
class Activation_softmax: 
    def forwardpass(self,input):
        exp_values = np.exp(input - np.max(input, axis= 1 , keepdims = True))
        norm_values = exp_values /np.sum(exp_values, axis = 1, keepdims = True)
        self.output = norm_values
    


layer1 = Layer_Dense(2,5)
layer1.forward_pass(X)
activation1 = Activation_softmax()
activation1.forwardpass(layer1.output)
print(activation1.output)

                            