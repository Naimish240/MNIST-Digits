# Program to create a neural network with input layer,
# output layer, and one hidden layer.

from random import random

# Initialize the network
def init_network(inputs, hidden, outputs):
    # -------------------------------------------
    # INPUT:
    # -------------------------------------------
    # inputs  : number of inputs
    # hidden  : number of neurons in hidden layer
    # outputs : length of output layer
    # OUTPUT: 
    # -------------------------------------------
    # network : array of layers in the network
    # -------------------------------------------
    
    network = []
    hidden_layer = {}
    output_layer = {}
    
    # weights and bias for hidden layer
    h_weights = []
    for i in range(inputs):
        h_weights.append(random())
    hidden_layer['weights'] = h_weights
    hidden_layer['bias'] = random()
    network.append(hidden_layer)
    
    # weights and bias for output layer
    o_weights = []
    for i in range(hidden):
        o_weights.append(random())
    output_layer['weights'] = o_weights
    output_layer['bias'] = random()
    network.append(output_layer)
    
    # returns network
    return network
    
if __name__ == '__main__':
    network = init_network(5, 9, 2)
    
    for i in network:
        print(i)
