#///////////////////////////////////////////#
# Simple neural network program in python   #
# To train on MNIST digit dataset           #
# Written without any external dependancies #
#///////////////////////////////////////////#

# This program starts the execution

'''
    Functions:
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
    
    def beautify(arr, c)
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
                    
'''

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
if __name__ == '__main__':
    # Load data
    a = convert_csv('mnist_train.csv',255)
    
    # Save to binary file
    # with open('training_dataset.bin', 'wb') as fh:
    #     pickle.dump(a, fh)

#//////////////////////////////////////////////////////////////////////////#
