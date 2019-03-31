
# This part of the program exists to save the model
# and load one when required

'''
    Functions:
    
    
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
# Function to load model from epoch x
def loadModel(x):
    # x     : File name
    data = None
    with open(x, 'wb') as fh:
        data = pickle.load(fh)
    return data
#//////////////////////////////////////////////////////////////////////////#
