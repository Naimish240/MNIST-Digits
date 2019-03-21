# Contains functions to parallelize the computations
# performed by the neuralnetwork.py script

# Work In Progress
# Pipeline under construction

# Import statements
from __future__ import print_function
import multiprocessing as mp

import random

cpu_count = mp.cpu_count()

def random_init(prev_len, curr_len):
    o = []
    for i in range(curr_len):
        output_layer = {}
        o_weights = []
        for j in range(prev_len):
            o_weights.append(random())
        output_layer['weights'] = o_weights
        output_layer['bias'] = random()
        o.append(output_layer)
    return o

def parallel_init(inputs, hidden, outputs):
    network = []
    # Initializing one layer at a time, while parallelizing
    # the initiation of weights and biases
    
    while True:
        continue

if __name__ == '__main__':
    # Prints number of cores
    print(cpu_count)
