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
import pickle                   # To save the model
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Creating the network as an array
# Global variable
network = []
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to get data from csv and return a dictionary
def convert_csv(fileName, val = 1):
    input_data = []
    output_data = []
    
    # Opening the csv file
    with open(fileName, 'r') as fh:
        lines = fh.readlines()
        
        for i in lines:
            i = i.strip('\n')                    # Removing new line char
            i = i.split(',')                     # Splitting with comma
            
            output_data.append(int(i[0]))        # Adding to output matrix
            input_data.append(i[1:])             # Adding to input strings
    # Edits input data to contain values from 0-1 instead of 0-val
    # Provided the value is not unity
    if val != 1:
        input_data = beautify(input_data, val)
    # Returns the input and output matricies
    return [input_data, output_data]
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Convert the values of input matrix from between 0-255 to 0-1
def beautify(arr, c):
    # arr : input matrix
    # c   : factor to divide by
    a = []
    # For loop to divide each element by c
    for i in arr:
        b = []
        for j in i:
            b.append(int(j)/c)
        a.append(b)
    # Returns double dimentional array
    return a
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
# Function to save the model after the current epoch
def saveModel(x):
    # x     : Epoch number
    with open('model_{}_{}'.format(x, time), 'wb') as fh:
        pickle.dump(network, fh)
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to load model from epoch x
def loadModel(x):
    # x     : File name
    data = None
    with open('model_{}'.format(x), 'wb') as fh:
        data = pickle.load(fh)
    return data
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

#//////////////////////////////////////////////////////////////////////////#
if __name__ == '__main__':
    # Load data
    a = convert_csv('mnist_train.csv',255)
    
    # Save to binary file
    # with open('training_dataset.bin', 'wb') as fh:
    #     pickle.dump(a, fh)

#//////////////////////////////////////////////////////////////////////////#
