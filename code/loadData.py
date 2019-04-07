#///////////////////////////////////////////#
# Simple neural network program in python   #
# To train on MNIST digit dataset           #
# Written without any external dependancies #
#///////////////////////////////////////////#

# This program starts the execution

'''
    Functions:
    ##########################################################
    def echo()
	Input:
	----------------------------------------------------
	This function takes 'command' as input
	----------------------------------------------------

	Output:
	----------------------------------------------------
	echoes message to console
	----------------------------------------------------

	Purpose:
	----------------------------------------------------
	To assist in selecting folder without gui
	----------------------------------------------------

    ##########################################################
    def commands()
	Input:
	----------------------------------------------------
	OPERATING_SYSTEM for win vs linux vs mac
	----------------------------------------------------

	Output:
	----------------------------------------------------
	Prints the help menu for the custom terminal
	----------------------------------------------------

	Purpose:
	----------------------------------------------------
	To display the help menu
	----------------------------------------------------

    ##########################################################
    def my_terminal()
	Input:
	----------------------------------------------------
	This function takes no inputs
	----------------------------------------------------

	Output:
	----------------------------------------------------
	Returns the file name selected
	----------------------------------------------------

	Purpose:
	----------------------------------------------------
	To select the file without Tkinter
	----------------------------------------------------

    ##########################################################
    def folder_finder()
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
        Selecting dataset through Tkinter GUI or custom 
	terminal
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
'''

from __future__ import print_function

# Getting the system information
from sysinfo import sysinfo

info = sysinfo(return_info = True)
PYTHON_VERSION = info['python_version']
OPERATING_SYSTEM = info['os']

from time import time
import os
    
try:
    from tqdm import tqdm

except:
    if 'linux' in OPERATING_SYSTEM.lower() and PYTHON_VERSION[0] == '3':
        os.system('pip3 install tqdm')
    else:
        os.system('pip install tqdm')

from pickle import load, dump

#//////////////////////////////////////////////////////////////////////////#
# Function to echo a message to the terminal
def echo(command):
    os.system('echo "> {}"'.format(command))
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to echo standard messages to terminal
def commands(OPERATING_SYSTEM):
    if 'windows' in OPERATING_SYSTEM.lower():
        cmd = 'dir'
    else:
        cmd = 'ls'

    print('-' * 78)
    echo('COMMANDS')
    echo("1. '{}'                : list the items in the current directory".format(cmd))
    echo("2. 'cd <name>           : changes the current working directory to <name>")
    echo("    use 'cd ..' to move into previous directory")
    echo("3. 'select <file_name>' : select the file (with proper extension)")
    echo("4. 'pwd'                : prints current working directory")
    echo("5. 'help'               : prints this help menu")
    echo("6. 'ctrl + c'           : exit without selecting the file")
    print('-' * 78)
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function for the terminal extension
def my_terminal():
    print('-' * 78)
    echo('Your system does not support Tkinter.')
    echo('Select the file you want to load by navigating through the terminal.')
    commands(OPERATING_SYSTEM)
    while True:
        cmd = input("> Enter your command: ")

        ch = cmd.split()
        print('-' * 78)

        if ch[0] == 'select':
            return ch[1]
        elif ch[0] == 'dir' or ch[0] == 'ls':
            try:
                os.system(cmd)
            except:
                echo("ERROR!!! WINDOWS DOES NOT SUPPORT THIS COMMAND!")
        elif ch[0] == 'cd':
            try:
                new_dir = ''
                for i in ch[1:]:
                    new_dir += (i + '\\ ')
                new_dir = new_dir[:-2]
                print(new_dir)
                os.chdir(new_dir)
                echo("CURRENT WORKING DIRECTORY : {}".format(os.getcwd()))
            except:
                echo("ERROR!!! THE DIRECTORY DOES NOT EXIST! TRY AGAIN!")
        elif ch[0] == 'pwd':
            echo("CURRENT WORKING DIRECTORY : {}".format(os.getcwd()))
        elif ch[0] == 'help':
            commands(OPERATING_SYSTEM)
        else:
            echo("ERROR!!! INVALID COMMAND! TRY AGAIN!")
            commands(OPERATING_SYSTEM)
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to select folder
def folder_finder():
    filename = None
    try:
        # (Un)comment to control gui
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()                       # We don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename()          # Show an "Open" dialog box and return the path to the selected file
    
    # select folder through terminal
    except:
        filename = my_terminal()
    
    return filename
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to get data from csv and return a list
# csv has first row as output and rest as inputs
def convert_csv(fileName, vectorize_output = True):
    input_data = []
    output_data = []

    # Opening the csv file
    with open(fileName, 'r') as fh:
        lines = fh.readlines()

        # Replace following line with "for i in lines:" to use without tqdm
        for i in tqdm(lines):
            i = i.strip('\n')                    # Removing new line char
            i = i.split(',')                     # Splitting with comma

            output_data.append(float(i[0]))      # Adding to output matrix
            input_data.append([float(i) for i in i[1:]])             # Adding to input strings

    # Edits input data to contain values from 0-1 instead of 0-val
    # Provided the value is not unity
    print("> Do you want to normalize the dataset? (Y/N) : ")
    val = input().lower()
    if 'y' in val:
        x = max(max(input_data))
        y = min(min(input_data))
        z = max(x, abs(y))
        print(z)
        input_data = beautify_input(input_data, z)
    
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
    c = int(input("> Enter the number of output neurons: "))
    print("> Forming output vectors")
    
    # For loop to do the conversion
    # If only one output neuron, then return the list of outputs as output
    if c == 1:
        for val in tqdm(arr):
            a.append(val)
    else:
        # Replace following line with "for val in arr:" to run without tqdm 
        for val in tqdm(arr):
            # Creating empty list of zeros
            l = [0] * c
            # Replacing the element with its value
            l[int(val)] = 1
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
    file = folder_finder()
    network = None
    print("> Loading the model")
    # Opening the model .dat file
    with open(file, 'rb') as fh:
        network = load(fh)
    # Returning the network
    return network
#//////////////////////////////////////////////////////////////////////////#

#//////////////////////////////////////////////////////////////////////////#
# Function to save the model after the current epoch
def saveModel(network, x = 1):
    # x         : Epoch number
    # network   : Contains weights and biases of each layer
    with open('model_{}_{}.dat'.format(x, time), 'wb') as fh:
        dump(network, fh)
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
