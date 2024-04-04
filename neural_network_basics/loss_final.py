#everything up until now. 

import numpy as np 
import nnfs 

from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs ):
        self.output = np.dot(inputs, self.weights) + self.biases 

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class softmax_activation: 
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1 , keepdims=True))
        norm_values = exp_values/ np.sum(exp_values , axis = 1, keepdims=True)
        self.output = norm_values

class Loss:

    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Categorical_Crossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1-1e-7)
        if(len(y_true.shape) == 1):
            correct_values = y_pred_clipped[range(samples), y_true] 
        elif(len(y_true.shape) == 2):
            correct_values = np.sum(y_pred_clipped*y_true, axis = 1)

        correct_prob = -np.log(correct_values)

        return correct_prob
    

X,y = spiral_data(samples = 100, classes = 3)

dense1 = Layer_Dense(2,3)
dense2 = Layer_Dense(3,3) #taking output of previous as input of this and taking another 3 as the number of classes we have to predict. 

dense1.forward(X) #forward pass 

activation1 = Activation_ReLU() 
activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2 = softmax_activation()

activation2.forward(dense2.output)

loss = Categorical_Crossentropy()

loss_value = loss.calculate(activation2.output, y)




print(loss_value)
