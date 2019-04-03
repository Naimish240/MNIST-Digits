# Import statements
from __future__ import print_function
from random import random
from math import exp
from tqdm import tqdm
from pickle import dump
import numpy as np

key = {
    'weights' : 0,
    'bias'    : 1,
    'output'  : 2,
    'delta'   : 3
}

# Initialize the network
def init_network(inputs, hidden, outputs):
    network = []

    # weights and bias for hidden layers
    
    for i in range(len(hidden)):
        h = []
        for j in range(hidden[i]):
            hidden_layer = []
            h_weights = []

            # For first hidden layer
            if i == 0:
                for k in range(inputs):
                    h_weights.append(random())
            # For subsequent hidden layers
            else:
                for k in range(hidden[i-1]):
                    h_weights.append(random())
            
            # Adding weights and biases to the dictionary
            hidden_layer[key['weights']] = h_weights
            hidden_layer[key['bias']] = random()
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