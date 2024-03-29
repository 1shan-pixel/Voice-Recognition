#implementing np.dot to find the output activatoins. 

import numpy as np 
weights = [[1,2],[4,1],[4,2]]
biases = [2,3,5]
input_vals = [2,4]
output = []
output = np.dot(weights,input_vals)+biases 
print(output)