# Import statements
from __future__ import print_function
from random import random
from math import exp
from tqdm import tqdm
from pickle import dump
import numpy as np      

class HiddenLayer(object):
    # Creates a hidden layer
    def __init__(self, neurons, neurons_in_prev):
        # Creates a layer of n neurons
        weight_matrix = []
        bias_matrix = []
        for i in neurons:
            weights = self.init_weights(neurons_in_prev)
            weight_matrix.append(weights)
            bias_matrix.append(random())

        self.weights = np.array(weight_matrix, dtype = np.float_)
        self.bias = np.array(bias_matrix, dtype = np.float_)

    def init_weights(self, neurons_in_prev):
        weights= []
        for i in len(neurons_in_prev):
            weights.append(random())
        
        return weights
        
    