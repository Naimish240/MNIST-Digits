# Sample code creating a neural network with the modules in 
# the folder code/Using Functions/
# Imports loadData.py and neuralnetwork.py from code/Using Functions/
# Currently supports only one hidden layer
# Will add support for multiple hidden layers soon

from __future__ import print_function
import loadData as load
import neuralnetwork as nn

file = load.folder_finder_gui()
# Assuming dataset is normalized by default
dataset = load.convert_csv(file)

input_layer, output_layer = dataset[0], dataset[1]
i_neurons = len(input_layer[0])
h_neurons = int(input("Enter the number of neurons in the hidden layer: "))

# For loop to find distinct elements in the output matrix
distinct_o = []
for i in output_layer:
    if i not in distinct_o:
        distinct_o.append(i)
o_neurons = len(distinct_o)

network = nn.init_network(i_neurons, h_neurons, o_neurons)
l_rate = 0.01
epoches = 5

nn.train(network, input_layer, l_rate, epoches, output_layer)