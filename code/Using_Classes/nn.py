#!/usr/bin/env python

from __future__ import print_function
from tqdm import tqdm
from pickle import dump
import numpy as np      

class Network(object):
    def __init__(self, layers, activations):
        assert(len(layers) == len(activations) + 1)
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self.initialize()

    def initialize(self):
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i+1], self.layers[i]))
            self.biases.append(np.random.randn(self.layers[i+1], 1))



if __name__ =='__main__':
    layers = [1, 100, 1]
    activations = [1, 2]
    network = Network(layers, activations