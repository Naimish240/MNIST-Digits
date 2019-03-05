#///////////////////////////////////////////#
# Simple neural network program in python   #
# To train on MNIST digit dataset           #
# Written without any external dependancies #
#///////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////
# Import statements
import random                   # To initialize weights and biases
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
# Function to randomly initialize weights and biases 
def rand_init(x, low, high):
    # x    : the number of neurons in layer
    # low  : the lower limit of random value
    # high : the upper limit for random values
    # W.I.P.
    pass
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to find sigmoid of x
# Sigmoid of x is given as
#       e^x
#    ---------  
#     e^x + 1  
def sigmoid(x):
    # x : variable to calcuate sigmoid for
    return (exp(x) / (exp(x) + 1))
#//////////////////////////////////////////////////////////////////////////#

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
