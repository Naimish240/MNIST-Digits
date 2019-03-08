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
def convert_csv(fileName, val):
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
# Function to randomly initialize weights and biases for hidden layer
def rand_init(x, y):
    # x    : The number of neurons in layer
    # y    : Number of neurons in previous layer
    weight = []                         # Create weight matrix
    
    for i in range(x):
        l = []
        for j in range(y):
            rt_x = 1/sqrt(x)            # Finds value of 1/root(x)
            rand = uniform(-rt_x,rt_x)  # Generates random number
            l.append(rand)              # Adds it to list
        weight.append(l)                # Adds list to weight matrix
        
    # Return list weights
    return weight
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to calculate dot product
def dot_product(x, wt, b):
    # x    : input matrix  
    # wt   : weight matrix 
    # b    : bias value matrix
    dot = sum(i*j for i,j in zip(x,wt)) # Calculate dot product
    dot += b                            # Add bias
    
    # Return the product
    return dot
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to find sigmoid
def sigmoid(x):
    # x : variable to calcuate sigmoid for
    return (exp(x) / (exp(x) + 1))
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to calculate the derivative of the sigmoid function
def derivativeSigmoid(x):
    # x : variable to calcuate derivative for for
    t = sigmoid(x)
    return t*(1-t)
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
        self.index = index
        self.wt = []
        self.bias = [0] * length
        self.output = output
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Initialize values of weights
    def setWeight(self):
        # Calls the rand_init function and passes to it the length of the 
        # previous layer and length of the current layer 
        self.weight = rand_init(len(self.bias), len(self.index - 1))
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Function to find output
    def getOutput(self):
        for i in network[self.index - 1]:
            continue
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
def start(epoches, learning_rate, hidden_layers, act_fn = 'Sigmoid'):
    # epoches               : Number of cycles to train for
    # learning_rate         : Learning rate of the network
    # hidden_layers         : Number of hidden layers in the network
    # act_fn                : Activation function used
    for i in range(epoches):
        continue
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
if __name__ == '__main__':
    # Currently takes 13 seconds
    # Tries loading the input csv
    a = trainingData()
    t = 0
    for i in a[0]:
        if len(i) == 784:
            t+=1
        else:
            print(len(i))
    print(t)
    print(a[0][0],"\n",len(a[0][0]))
#//////////////////////////////////////////////////////////////////////////#
