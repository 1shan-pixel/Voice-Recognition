import numpy as np

#coding multiple layers of a neural network 

inputs =[[1, 2, 3,4],
        [5, 6, 7,6],
        [9, 10, 11,6]]
weights= [
    [0.1, 0.2, 0.3, 0.9],
    [0.4, 0.5, 0.6, 0.7],
    [0.7, 0.8, 0.9, 0.5]
]

biases = [2, 3, 0.5]    
output = []

output = np.dot(inputs,np.array(weights).T) + biases 

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs , n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        pass 
    def forward_pass(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases


layer1 = Layer_Dense(4,4)
layer1.forward_pass(inputs)
layer2 = Layer_Dense(4,4)

layer2.forward_pass(layer1.outputs)
print(layer2.outputs)
