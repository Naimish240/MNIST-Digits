
# This part of the program exists to save the model
# and load one when required

'''
    Functions:
    def saveModel(network, x)
        Input : 
        ----------------------------------------------------
        x       : epoch number
        network : array of weights and biases at each layer
        ----------------------------------------------------
        
        Output: 
        ----------------------------------------------------
        Saves the array to a .dat file
        ----------------------------------------------------
        
        Purpose:
        ----------------------------------------------------
        This function exists to save the model
        ----------------------------------------------------
        
    ##########################################################
    
    def loadModel(x)
        Input : 
        ----------------------------------------------------
        x    : Name of file to load model from
        ----------------------------------------------------
        
        Output: 
        ----------------------------------------------------
        Returns the network
        ----------------------------------------------------
        
        Purpose:
        ----------------------------------------------------
        This function exists to load the model
        ----------------------------------------------------
                    
'''

#//////////////////////////////////////////////////////////////////////////#
import pickle                   # To save the model
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to save the model after the current epoch
def saveModel(network, x = 1):
    # x         : Epoch number
    # network   : Contains weights and biases of each layer
    with open('model_{}_{}.dat'.format(x, time), 'wb') as fh:
        pickle.dump(network, fh)
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to load model from epoch x
def loadModel(x):
    # x     : File name
    data = None
    with open(x, 'wb') as fh:
        data = pickle.load(fh)
    return data
#//////////////////////////////////////////////////////////////////////////#
