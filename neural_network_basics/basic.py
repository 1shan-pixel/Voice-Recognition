
weights = [[1,2],[4,1],[4,2]]
biases = [2,3,5]
input_vals = [2,4]

output_values = []
for neuron_weight, bias in zip(weights,biases):
    output = 0 
    for input_values, weights in zip(input_vals,neuron_weight):
        output = output + input_values* weights
    output = output + bias 
    output_values.append(output)        

print(output_values)
        


