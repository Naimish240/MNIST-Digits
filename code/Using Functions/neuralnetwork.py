# Program to create a neural network with input layer,
# output layer, and one hidden layer.

# Import statements
from __future__ import print_function
from random import random
from math import exp
from tqdm import tqdm
from pickle import dump

'''
    Functions:
    ------------------------------------------------
    init_network(inputs, hidden, outputs)
        Initializes the network    
    ------------------------------------------------
    dot(weights, bias, inputs)
        Calcuates the dot product and adds bias 
    ------------------------------------------------
    sigmoid(val)
        Calcuates the sigmoid of input
    ------------------------------------------------
    transfer_derivative(val)
        Calcuates transfer derivative of value
    ------------------------------------------------
    forward_propagation(network, inputs)
        Performs forward propagation
    ------------------------------------------------
    backpropagate_error(network, output)
        Calculates backpropagtation error for neuron
    ------------------------------------------------
    update_weights(network, row, l_rate)
        Updates the weights and biases
    ------------------------------------------------
    train(network, l_rate, n_epoch, expected, log = False)
        Trains the network
    ------------------------------------------------
    predict(network, row)
        Inputs the row into the network and predicts
    ------------------------------------------------
    main()
        Main function for the program
    ------------------------------------------------
'''

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
    # network : list : layers in the network, sans input
    # ----------------------------------------------------
    
    network = []

    # weights and bias for hidden layers
    h = []
    for i in range(hidden):
        hidden_layer = {}
        h_weights = []
        for j in range(inputs):
            h_weights.append(random())
        hidden_layer['weights'] = h_weights
        hidden_layer['bias'] = random()
        h.append(hidden_layer)
    network.append(h)
    
    # weights and bias for output layer
    o = []
    for i in range(outputs):
        output_layer = {}
        o_weights = []
        for j in range(hidden):
            o_weights.append(random())
        output_layer['weights'] = o_weights
        output_layer['bias'] = random()
        o.append(output_layer)
    network.append(o)
    
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

    # Initializing the sum to equal the bias of the neuron    
    sum = bias
    
    # Calculates dot product
    for i in range(len(weights)):
        sum += (weights[i] * inputs[i])
    
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
    
def transfer_derivative(val):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # val : float : value to calculate derivative for
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # x : float : derivative of input
    # ----------------------------------------------------    
    x = val * (1 - val)
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

# Function to backpropagate error and store in neuron
def backpropagate_error(network, output):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # network : list : the matrix which contains the network
    # output  : list : the expceted output for the network 
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # output : no output, just modifies the network
    # ----------------------------------------------------
    
    # Backtracing our steps to calcuate error
    for i in range(len(network)-1, -1, -1):
        layer = network[i]
        errors = []
        
        # Verifying that i is not the last layer
        if i != (len(network) - 1):
            for j in range(len(layer)):
                error = 0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
                
        # If i is the last layer
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(output[j] - neuron['output'])

        # For loop to calculate delta
        for j in range(len(layer)):
            neuron = layer[j]
            # Calcualting delta
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Function to update weights and bias
def update_weights(network, row, l_rate):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # network : list  : the matrix which contains the network
    # row     : list  : the input for the network
    # l_rate  : float : the learning rate for the network
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # output : no output, just modifies the network
    # ----------------------------------------------------
    for i in range(len(network)):
        inputs = [neuron['output'] for neuron in network[i - 1]]
        if i == 0:
            inputs = row
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['bias'] += l_rate * neuron['delta']

# Function to train the network
def train(network, train, l_rate, n_epoch, expected, log = False):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # network  : list  : the matrix which contains the network
    # train    : list  : the input for the network
    # l_rate   : float : the learning rate for the network
    # n_epoch  : int   : number of epochs to train for
    # expected : list  : expected output list
    # log      : bool  : save the network after each epoch?
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # output : no output, just modifies the network
    # ----------------------------------------------------    
    for epoch in range(n_epoch):
        sum_error = 0
        # replace following line with "for i in range(len(train)):" to run without using tqdm
        for i in tqdm(range(len(train))):
            row = train[i]
            expected_val = expected[i]
            outputs = forward_propagation(network, row)
            for j in range(len(expected_val)):
                sum_error += (expected_val[j] - outputs[j]) **  2
            backpropagate_error(network, expected_val)
            update_weights(network, row, l_rate)
        print('> Epoch {} , l_rate = {} ,error = {}'.format(epoch + 1, l_rate, sum_error))

        # If logging is enabled, the network is stored after each iteration
        if log:
            # Creates .dat file, and loggs the network state onto the file
            with open("Epoch_{}_error_{}.dat".format(epoch, sum_error), 'wb') as fh:
                fh.dump(network)

# Function to predict value for an unknown input
def predict(network, row):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # network : list  : the matrix which contains the network
    # row     : list  : the input for the network
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # output : returns the prediction
    # ----------------------------------------------------
    outputs = forward_propagation(network, row)
    return outputs.index(max(outputs))

# Main function
def main():
    # Will finish this function later
    pass

if __name__ == '__main__':
    pass
