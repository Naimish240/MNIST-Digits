# Import statements
from __future__ import print_function
from tqdm import tqdm
from pickle import dump
import numpy as np      
      
class NeuralNetwork(object):
    def __init__(self, layers, activations):
        assert(len(layers) == len(activations) + 1)
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []

    def initialize(self):
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i+1], layers[i]))
            self.biases.append(np.random.randn(layers[i+1], 1))

    def feedforward(self, x):
        # Returns the feed forward value for X
        a = np.copy(x)
        z_s = []
        a_s = [a]

        for i in range(len(self.weights)):
            act_fn = self.getActFn(self.activations[i])
            z_s.append(self.weights[i].dot(a) + self.biases[i])
            a = act_fn(z_s[-1])
            a_s.append(a)

        return (z_s, a_s) 
    
    @staticmethod
    def getActFn(name):
        if (name == 'sigmoid'):
            return lambda x : np.exp(x) / (1+np.exp(x))

        elif (name == 'relu'):
            
            def relu(x):
                y = np.copy(x)
                y[y<0] = 0
                return y
            
            return relu
            
        else:
            print("> Unknown Activaction function. Linear used instead.")
            return lambda x : x

    def backpropagation(self, y, z_s, a_s):
        dw = []     # dC/dW
        db = []     # dC/dB

        deltas = [None] * len(self.weights)     # delta = dC/dZ == error for layer
        
        # Adding delta of last layer
        # Using MSE loss function. replace (y - a_s[-1]) with derivative of other cost function
        # if you wish to use another loss function
        deltas[-1] = ((y-a_s[-1]) * (self.getDerActFn(self.activations[-1]))(z_s[-1]))

        # Backpropagation
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1]) * (self.getDerActFn(self.activations[i])(z_s[i]))
            
            batch_size = y.shape(1)

            db = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
            dw = [d.dot(a_s[i].T)/float(batch_size) for i, d in enumerate(deltas)]

            # Returns derivative with respect to weight matrix and biases
            return dw, db

    @staticmethod
    def getDerActFn(name):
        if (name == 'sigmoid'):
            sig = lambda x : np.exp(x) * (1-sig(x))

            return lambda x : sig(x) * (1-sig(x))

        elif (name == 'linear'):
            return lambda x : 1

        elif (name == 'relu'):

            def der_relu(x):
                y = np.copy(x)
                y[y>=0] = 1
                y[y<0] = 0

                return y
            
            return der_relu

        else:
            print("> Unknown activation function. Using Linear instead.")
            return lambda x : 1

    def train(self, x, y, batch_size, epochs, lr):
        # Updates weights and biases based on output

        for e in tqdm(range(epochs)):
            i = 0
            while (i<len(y)):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                i += batch_size

                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)

                self.weights = [w + lr * dweight for w, dweight in zip(self.weights, dw)]
                self.biases = [w + lr * dbias for w, dbias in zip(self.biases, db)]

                print("> Loss = {}".format(np.linalg.norm(a_s[-1]-y_batch)))
