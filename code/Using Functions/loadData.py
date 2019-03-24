#///////////////////////////////////////////#
# Simple neural network program in python   #
# To train on MNIST digit dataset           #
# Written without any external dependancies #
#///////////////////////////////////////////#

# This program starts the execution

'''
    Functions:
    ##########################################################
    def folder_finder_gui()
        Input : 
        ----------------------------------------------------
        this function takes no inputs
        ----------------------------------------------------
        
        Output: 
        ----------------------------------------------------
        returns folder name
        ----------------------------------------------------
        
        Purpose:
        ----------------------------------------------------
        Selecting dataset through Tkinter GUI
        ----------------------------------------------------

    ##########################################################
    def convert_csv(fileName, val)
        Input : 
        ----------------------------------------------------
        fileName  : Name of the file being loaded as input 
        val       : Value to divide input by
        ----------------------------------------------------
        
        Output: 
        ----------------------------------------------------
        returns list containing input and output data
        ----------------------------------------------------
        
        Purpose:
        ----------------------------------------------------
        Reading the csv to the input and output layers
        ----------------------------------------------------
    ##########################################################
    def beautify_input(arr, c)
        Input : 
        ----------------------------------------------------
        arr  : The input array from convert_csv
        c    : Factor to diminish each value by
        ----------------------------------------------------
        
        Output: 
        ----------------------------------------------------
        Returns the input array diminished by a factor c
        ----------------------------------------------------
        
        Purpose:
        ----------------------------------------------------
        Divide each element by factor c
        ----------------------------------------------------
    ##########################################################
    def beautify_output(arr)
        Input : 
        ----------------------------------------------------
        arr  : The output expected for the image
        ----------------------------------------------------
        
        Output: 
        ----------------------------------------------------
        Returns the output vector
        ----------------------------------------------------
        
        Purpose:
        ----------------------------------------------------
        convert from 'int' to 'list'
        ----------------------------------------------------
    ##########################################################
    def load_model()
        Input : 
        ----------------------------------------------------
        No input
        ----------------------------------------------------
        Output: 
        ----------------------------------------------------
        Returns the selected model
        ----------------------------------------------------
        Purpose:
        ----------------------------------------------------
        to load a pre-trained model
        ----------------------------------------------------
    ##########################################################
'''

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
from pickle import load

#//////////////////////////////////////////////////////////////////////////#
# Function to select folder using gui
def folder_finder_gui():
    Tk().withdraw()                       # We don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()          # Show an "Open" dialog box and return the path to the selected file
    return filename
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to get data from csv and return a list
def convert_csv(fileName, val = 1, vectorize_output = True):
    input_data = []
    output_data = []
    
    # Opening the csv file
    with open(fileName, 'r') as fh:
        lines = fh.readlines()
        
        # Replace following line with "for i in lines:" to use without tqdm
        for i in tqdm(lines):
            i = i.strip('\n')                    # Removing new line char
            i = i.split(',')                     # Splitting with comma
            
            output_data.append(int(i[0]))        # Adding to output matrix
            input_data.append(i[1:])             # Adding to input strings

    # Edits input data to contain values from 0-1 instead of 0-val
    # Provided the value is not unity
    if val != 1:
        input_data = beautify_input(input_data, val)
    
    # Converts output from int to vector
    # For testing data only, not training
    if vectorize_output:
        output_data = beautify_output(output_data)
    # Returns the input and output matricies
    return [input_data, output_data]
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Convert the values of input matrix from between 0-255 to 0-1
def beautify_input(arr, c):
    # arr : input matrix
    # c   : factor to divide by
    print("> Normalizing the dataset")
    a = []
    # For loop to divide each element by c
    # Replace following line with "for i in arr:" to run without tqdm
    for i in tqdm(arr):
        b = []
        for j in i:
            b.append(int(j)/c)
        a.append(b)
    # Returns double dimentional array
    return a
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Convert the output value into the output layer
def beautify_output(arr):
    # arr : list containing all outputs
    a = []
    print("> Forming output vectors")
    # For loop to do the conversion
    # Replace following line with "for val in arr:" to run without tqdm 
    for val in tqdm(arr):
        # Creating empty list of zeros
        l = [0] * 10
        # Replacing the element with its value
        l[val] = 1
        # Adding it to the list
        a.append(l)

    # Returning the list of output vectors
    return a
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to load a trained model
def load_model():
    print("> Select the model to load")
    # Choosing file with Tkinter gui
    file = folder_finder_gui()
    network = None
    print("> Loading the model")
    # Opening the model .dat file
    with open(file, 'rb') as fh:
        network = load(fh)
    # Returning the network
    return network
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
if __name__ == '__main__':
    pass
    # Load data
    #a = convert_csv('mnist_train.csv',255)
    
    # Save to binary file
    # with open('training_dataset.bin', 'wb') as fh:
    #     pickle.dump(a, fh)

#//////////////////////////////////////////////////////////////////////////#
