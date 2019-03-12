#///////////////////////////////////////////#
# Simple neural network program in python   #
# To train on MNIST digit dataset           #
# Written without any external dependancies #
#///////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Import statements
from random import uniform      # To initialize weights and biases
from math import sqrt, exp      # For mathematical operations
from time import time           # To calculate time taken
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Creating the network as an array
# Global variable
network = []
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Class whose object is the layer
class Layer(object):
    # Constructor
    #----------------------------------------------------------------------#
    def __init__(self, index, wt, bias, length, output = []):
        # index     : Index of current layer
        # weight    : Weight matrix for layer
        # bias      : Bias value for the each neuron in layer
        # output    : Output matrix of the layer
        # x         : Matrix of neurons for the layer

	# length    : The number of neurons in the network

        self.index = index
        self.x = [] * length
        self.wt = []
        self.bias = [0] * length
        self.output = output
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Initialize values of weights
    def setWeight(self):
        # Calls the rand_init function and passes to it the length of the 
        # previous layer and length of the current layer 
        self.weight = self.rand_init(len(self.bias), len(self.index - 1))
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Function to find output
    def getOutput(self):
        for i in network[self.index - 1]:
            continue
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Function to randomly initialize weights and biases for hidden layer
    def rand_init(self):
        # x    : The number of neurons in layer
        # y    : Number of neurons in previous layer
        x = len(network[self.index].bias)
        y = len(network[self.index - 1].bias)
        
        weight = []           # Create weight matrix
        rt_x = 1/sqrt(x)      # Finds value of 1/root(x)
        
        # for loop to initialize the weights
        for i in range(x):
            l = []
            for j in range(y):
                rand = uniform(-rt_x,rt_x)  # Generates random number
                l.append(rand)              # Adds it to list
            weight.append(l)                # Adds list to weight matrix
            
        # Return list weights
        return weight
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Cost function
    def cost(self, last_n):
        # Using Euclidian distance
        # last_n    : Contains output of previous n iterations 
        # prev      : Contains the previous layer of the network
        
        prev = network[self.index - 1]
        
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Function to calculate the derivative of the sigmoid function
    def derivativeSigmoid(x):
        # x : variable to calcuate derivative for
        t = sigmoid(x)
        return t*(1-t)
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Function to find sigmoid
    def sigmoid(self, x):
        # x : variable to calcuate sigmoid for
        return (exp(x) / (exp(x) + 1))
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Function to return sigmoid matrix
    def sigmoidMatrix(self, l):
        # l   : Matrix to calculate sigmoid on
        # sig : Matrix storing the sigmoid values
        sig = []
        for i in range(len(self.bias)):
            x = sigmoid(l[i])
            sig.append(x)
        # Returns the sigmoid matrix
        return sig
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    def derivSigmoidMatrix(self, l):
        # l       : Matrix to calculate the derivative for
        # der_sig : Matrix to store the derivatives
        der_sig = []
        for i in range(len(self.bias)):
            x = derivativeSigmoid(l[i])
            der_sig.append(x)
        # Returns the derivative of sigmoid matrix
        return der_sig
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Function to calculate dot product matrix
    def dot_product(self):
        # wt   : weight matrix 
        # bias : bias value matrix
        # dot  : dot product matrix
        dot = []
        # for loop to calculate dot product
        for k in range(len(bias)):
            # Calculate the dot product
            dot_i = sum(i*j for i,j in zip(self.x[k], self.wt[k]))
            # Add bias
            dot_i += self.bias
            # Add dot product to the matrix
            dot.append(dot_i)
            
        # Return the product matrix
        return dot
    #----------------------------------------------------------------------#
#//////////////////////////////////////////////////////////////////////////#



#//////////////////////////////////////////////////////////////////////////#
def start(training, epoches, learning_rate, hidden_layers, act_fn = 'Sigmoid'):
    # training              : Input layer for the network
    # epoches               : Number of cycles to train for
    # learning_rate         : Learning rate of the network
    # hidden_layers         : Number of hidden layers in the network
    # act_fn                : Activation function used
    for i in range(epoches):
        continue
#//////////////////////////////////////////////////////////////////////////#

