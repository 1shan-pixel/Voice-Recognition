#calculating loss through the simple process of categorical cross entropy , quite a efficient method to calculate loss. 
import numpy as np 

'''predicted = [0.8 , 0.5 , 0.1]

one_hot_encoding = [1,0,0]

loss = -(np.log(predicted[0])*one_hot_encoding[0] + np.log(predicted[1])*one_hot_encoding[1] + np.log(predicted[2])*one_hot_encoding[2])
print(loss)'''

output = np.array([[0.8, 0.5, 0.1], 
                   [0.2, 0.3, 0.5], 
                   [0.6, 0.4, 0.0]])

target = [0, 1, 1]

print(np.mean(output[(range(len(output)), target)])) # intelligent way to find the 





