# Program to create a neural network with input layer,
# output layer, and one hidden layer.

# * UPDATE *
# Added support for multiple hidden layers! Yay!

# Import statements
from __future__ import print_function
from random import random, randint
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
    # inputs  : int  : number of inputs
    # hidden  : list : list of neurons in hidden layers
    # outputs : int  : length of output layer
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # network : list : layers in the network, sans input
    # ----------------------------------------------------
    
    network = []

    # weights and bias for hidden layers
    
    for i in range(len(hidden)):
        h = []
        for j in range(hidden[i]):
            hidden_layer = {}
            h_weights = []

            # For first hidden layer
            if i == 0:
                for k in range(inputs):
                    # Modified to have -ve initial weights too
                    h_weights.append(random() * (-1) * (randint(0, 10)))
            # For subsequent hidden layers
            else:
                for k in range(hidden[i-1]):
                    h_weights.append(random())
            
            # Adding weights and biases to the dictionary
            hidden_layer['weights'] = h_weights
            hidden_layer['bias'] = random()
            h.append(hidden_layer)

        # Adding the hidden layer to the
        network.append(h)
    
    # weights and bias for output layer
    o = []
    for i in range(outputs):
        output_layer = {}
        o_weights = []
        for j in range(hidden[-1]):
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
def sigmoid(gamma):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # gamma : float : value to calculate sigmoid for
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # x : float : sigmoid of input
    # ----------------------------------------------------    
    if gamma < 0:
        return 1 - 1/(1 + exp(gamma))
    else:
        return 1/(1 + exp(-gamma))
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
        inputs = []
        for neuron in network[i-1]:
            inputs.append(neuron['output'])

        if i == 0:
            inputs = row

        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['bias'] += l_rate * neuron['delta']

# Function to train the network
def train(network, training_dataset, l_rate, n_epoch, log = False, testing_dataset = False):
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

    train, expected = training_dataset[0], training_dataset[1]

    n_epoch = int(n_epoch)
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
    
        print("> Training result: ")
        print('> Epoch {} , l_rate = {} ,error = {}'.format(epoch + 1, l_rate, sum_error))

        if testing_dataset:
            testing(network, testing_dataset)

        # If logging is enabled, the network is stored after each iteration
        if log:
            # Creates .dat file, and loggs the network state onto the file
            with open("Epoch_{}_error_{}.dat".format(epoch, sum_error), 'wb') as fh:
                dump(network, fh)

# Function to test the network
def testing(network, testing_dataset):
    test, val = testing_dataset[0], testing_dataset[1]
    correct_guesses = 0
    # replace following line with "for i in range(len(train)):" to run without using tqdm
    for i in tqdm(range(len(test))):
        row = test[i]
        expected = val[i]
        network_guess = predict(network, row)
        # Prediction is correct
        if network_guess == expected:
            correct_guesses += 1
    accuracy = correct_guesses / len(test)
    accuracy *= 100
    print("> Testing result: ")
    print("> Accuracy : {}%".format(accuracy))

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
