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
# Function to get data from csv and return a dictionary
def trainingData():
    input_data = []
    output_data = []
    
    # Opening the csv file
    with open('mnist_train.csv', 'r') as fh:
        lines = fh.readlines()
        
        for i in lines:
            i = i.strip('\n')                    # Removing new line char
            i = i.split(',')                     # Splitting with comma
            
            output_data.append(int(i[0]))        # Adding to output matrix
            input_data.append(i[1:])             # Adding to input strings
    # Edits input data to contain values from 0-1 instead of 0-255
    input_data = beautify(input_data)
    # Returns the input and output matricies
    return [input_data, output_data]
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Convert the values of input matrix from between 0-255 to 0-1
def beautify(arr):
    # arr : input matrix
    a = []
    # For loop to divide each element by 255
    for i in arr:
        b = []
        for j in i:
            b.append(int(j)/255)
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
    return weights
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
# Class whose object is an array of neurons in the layer
class HiddenLayer(object):
    # Constructor
    #----------------------------------------------------------------------#
    def __init__(self, prev, wt, bias, length):
        # prev      : Output of the previous layer
        # weight    : Weight matrix for layer
        # bias      : Bias value for the each neuron in layer
        # output    : Output matrix of the layer
        self.prev = prev
        self.wt = []
        self.bias = [0] * length
        self.output = [] * length
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Initialize values of weights
    def setWeight(self):
        self.weight = rand_init(len(self.prev), len(self.output))
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    
#//////////////////////////////////////////////////////////////////////////#
if __name__ == '__main__':
    # Currently takes 13 seconds
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
