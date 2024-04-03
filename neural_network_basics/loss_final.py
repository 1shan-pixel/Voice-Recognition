#everything up until now. 

import numpy as np 

class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs ):
        self.output = np.dot(inputs, self.weights) + self.biases 
    

class softmax_activation: 
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1 , keepdims=True))
        norm_values = exp_values/ np.sum(exp_values , axis = 1, keepdims=True)
        self.output = norm_values

class Loss:

    def calculate(self,output,y):
        sample_losses = self.forward(self, output,y)
        data_loss = np.mean(sample_losses)
        return data_loss