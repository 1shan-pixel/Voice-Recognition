#going more into activation functions , we will go through sigmoid and relu activation functions. 
import numpy as np 
import nnfs 

from nnfs.datasets import spiral_data  

nnfs.init()

X, y = spiral_data(100, 3)   



class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) # we want a 2d array with 1 row and n_neurons columns., i don't know how removin the brackets doesnt make this a 2d array or something but whatever. 
    def forward_pass(self,input):
        self.output = np.dot(input, self.weights) + self.biases 
    
class Activation_ReLU:
    def forward_pass(self,inputs):
        self.output = np.maximum(0,inputs) # basically an implementation of the relu activation function.


layer1 = Layer_Dense(2,5)

layer1.forward_pass(X)

activation1 = Activation_ReLU()
activation1.forward_pass(layer1.output)
print(activation1.output)