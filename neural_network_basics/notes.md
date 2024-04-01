# Taken from the "Building Neural Networks from Scratch" Book 

Neural network is all about tuning the weights and biases of particular neurons , ultimately adjusting their activation. 

A tensor is an object that can be represented as an array. 

np.dot works on a pretty cool way , what the heck!

we need the concept of batch sizes, in order for the nerual network to properly learn. a batch size of 10 works way better than a batch size of 1. 

in the case of softmax activation function we need to add a small epsilon value to prevent 0 value in the case of logarithms. 

what does np.argmax do? 
