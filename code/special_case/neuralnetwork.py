# Program to create a neural network with input layer,
# output layer, and one hidden layer.

# import statements
from random import random
from math import exp

# Initialize the network
def init_network(inputs, hidden, outputs):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # inputs  : int : number of inputs
    # hidden  : int : number of neurons in hidden layer
    # outputs : int : length of output layer
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # network : array : layers in the network, sans input
    # ----------------------------------------------------
    
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

# Function to calculate dot product
def dot(weights, bias, inputs):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # weights : list  : contains weights from network
    # bias    : float : contains bias value from network
    # inputs  : list  : contains the input matrix
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # sum : float : value of dot product + bias
    # ----------------------------------------------------
    
    sum = 0
    
    # Calculates dot product
    for i in range(len(weights)):
        sum += weights[i] * input[i]
    
    # Adds bias
    sum += bias
    
    # Returns the value
    return sum
    
# Function to calculate activation of neurons
def sigmoid(val):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # val : float : value to calculate sigmoid for
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # x : float : sigmoid of input
    # ----------------------------------------------------
    
    x = 1.0 / (1 + exp(-val))
    return x
    
# Function to forward propagate through the network
def forward_propagation(network, inputs):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # network : list : the matrix which contains the network
    # inputs  : list : the inputs for the network 
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # output : list : returns the output layer
    # ----------------------------------------------------

    # loop to perform forward propagation
    for layer in network:
        # Stores output for layer
        output = []
        # Loop to calcualte outputs
        for neuron in layer:
            # Calcuating dot product
            dot_product = dot(neuron['weights'], neuron['bias'], inputs)
            # Applying sigmoid to output
            neuron['output'] = sigmoid(dot_product)
            # Adding the value to the output
            output.append(neuron['output'])
            
        # Updating output for next iteration
        inputs = output
    
    # Returning the output matrix
    return inputs
            
if __name__ == '__main__':
    network = init_network(5, 9, 2)
    
    for i in network:
        print(i)
